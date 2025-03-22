from turtle import forward
from httpx import head
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# hyper parameters
# batch_size = 32 #有多少个批次的数据同时进行训练，评估损失，更新参数
# block_size = 8
# max_iters = 5000
# eval_interval = 300
# learning_rate = 1e-3 # le-2 4.使用attention之后适当的降低学习率，因为attention模块无法承受很高的学习率
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# eval_iters = 200
# n_embed = 32 # 词嵌入的维度

# hyper parameters scaled
batch_size = 64 #有多少个批次的数据同时进行训练，评估损失，更新参数
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4 # 扩大规模后进一步降低学习率
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 384 # 词嵌入的维度
n_layer = 8 # 多少个block层
n_head = 6 # 多少个头 head_size = n_embed // n_heads=64
dropout = 0.2 # 随机失活的概率 因为网络扩大之后防止过拟合
#------------------------------------
print(f"Using device: {device}")
#------------------------------------

torch.manual_seed(1337)

current_file_path = os.path.dirname(__file__)

with open(os.path.join(os.path.dirname(current_file_path),'data/tinyshakespeare/input.txt'),'r',encoding='utf-8') as f:
    text = f.read()
    
# 统计所有出现的字符
chars = sorted(list(set(text)))
vocab_size = len(chars)

# build tokenizer
itos ={i:ch for i,ch in enumerate(chars)}
stoi ={ch:i for i,ch in enumerate(chars)}

encode = lambda s:[stoi[c] for c in s]
decode = lambda l:''.join([itos[i] for i in l])

# tensor所有的文本数据，并切分训练集和测试集
data = torch.tensor(encode(text),dtype=torch.long)
n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]

# 定义获取批数据函数
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    # 这里主要是如果cuda是可用的，那么将数据放在GPU上会运行的更快
    x,y = x.to(device),y.to(device)
    return x, y
  
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits,loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()    
    model.train()
    return out
  
  
class Head(nn.Module):
    '''单头自注意力模块，使用transformer算法将输入的B,T,C在T的维度实现注意力计算后再转换成B,T,head_size的输出'''
    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embed,head_size,bias=False)
        self.query = nn.Linear(n_embed,head_size,bias=False)
        self.value = nn.Linear(n_embed,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        
        # 9. 在计算token之间的亲密度的时候，增加dropout层，防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        B,T,C = x.shape
        # 1. 计算key,query,value
        k = self.key(x)  # B,T,head_size
        q = self.query(x)  # B,T,head_size
        wei = k @ q.transpose(-1,-2) * C**-0.5  # B,T,T
        # self.tril[:T,:T]对下三角的矩阵进行切片操作，只取:T,:T的部分
        wei = wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        wei = F.softmax(wei,dim=-1) # B,T,T
        # 9. 在计算token之间的亲密度的时候，增加dropout层，随机的阻止一些token进行通信
        wei = self.dropout(wei)
        
        v = self.value(x)  # B,T,head_size
        out = wei @ v  # B,T,head_size
        return out
        
class MultiHeadAttention(nn.Module):
    '''多头自注意力层，就是将多个头的注意力结果在C的维度cat起来'''
    def __init__(self,n_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        
        # 7. 增加最后一层残差连接的投影projection，因为x相当于原始的logits，而sa当前是使用了relu之后的激活值，
        #    所以需要一个全连接层再次将值投影到logits的空间。
        self.proj = nn.Linear(n_embed,n_embed)
    
    def forward(self,x):
        out =  torch.cat([h(x) for h in self.heads],dim=-1)
        out = self.proj(out)
        return out
class FeedForward(nn.Module):
    '''前馈层，连接在注意力层之后，让token在互相交流之后，能够独立思考的层'''
    def __init__(self,n_embed):
        super().__init__()
        self.net = nn.Sequential(
            # 8. 按照原始论文的设计，ffd层内部维数提高4倍再缩放回n_embed
            nn.Linear(n_embed,4 * n_embed),
            nn.ReLU(),
            # 7. 增加最后一层残差连接的投影projection，因为x相当于原始的logits，而sa当前是使用了relu之后的激活值，
            #    所以需要一个全连接层再次将值投影到logits的空间。
            nn.Linear(4 * n_embed,n_embed),
            # 9. 在返回残差通路之前，增加dropout层，防止过拟合
            nn.Dropout(dropout)
            )
    
    def forward(self,x):
        return self.net(x)  
    
    
class Block(nn.Module):
    def __init__(self,n_heads,n_embed):
        super().__init__()    
        head_size = n_embed // n_heads
        # self.block = nn.Sequential(
        #     MultiHeadAttention(n_heads=n_heads,head_size=head_size)
        #     FeedForward(n_embed)
        # )
        
        self.sa = MultiHeadAttention(n_heads=n_heads,head_size=head_size)
        self.ffwd = FeedForward(n_embed)
         # 8. layer norm 进一步帮助优化训练深层网络，相当于将每一个token的embed的值的数值分布都进行了标准正态分布的normalization
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self,x):
        # return self.block(x)
        
        # 7. x + 是使用了残差连接，sa和ffwd都是属于残差连接旁边的块，刚开始这些块对训练的贡献很小，相当于x=x+0，但是随着训练的进行，
        #    这些块的输出会逐渐变得更加重要，最终可以帮助模型更好地拟合数据，这就是残差连接的原理。
        # 8. 原始论文中，layer norm是位于multi head attention和feed-forward层之后，但是现在的做法更多的是reshuffle layer norm，在attention和ffd计算之前使用。
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    


class LayerNorm1d:
    '''层归一化，对比batch norm，层归一化只对每个样本,如按时间步T这个维度的方向进行归一化，而batch norm对整个batch，按B的方向进行归一化。
       这里我们使用nn的ln1，但是是如下实现原理。
    '''
    def __init__(self,dim,eps=1e-5):
        self.gamma = torch.ones(dim)  # (dim,)
        self.beta = torch.zeros(dim)
        self.eps = eps
        
    def __call__(self,x):
        # batch mean and variance
        xmean = x.mean(1,keepdim=True) # (batch,1)
        xvar = x.var(1,keepdim=True)   # (batch,1)

        xhat = (x-xmean)/torch.sqrt(xvar+self.eps)  # (batch,dim)
        self.out = self.gamma * xhat + self.beta  # (batch,dim)
        
        return self.out
    

# 定义模型
class TransformerLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. 词嵌入层，将输入的token转换为词向量
        self.token_embedding_table = nn.Embedding(vocab_size,n_embed)
        # 3.位置嵌入层，将位置信息编码到词向量中，使得每个token能知道它在句子中的位置信息
        self.position_embedding_table = nn.Embedding(block_size,n_embed)
        # 4. single-head self-attention层
        # self.sa_head = Head(n_embed) # self-attention head
        #self.sa_head = MultiHeadAttention(n_heads=4,head_size=n_embed//4) # 使用多头注意力，注意头数和每个头的维数
        # 5. feed-forward层，目的是为了让所有的token在互相交流之后，通过前馈层进行独立的思考，再将信息进行整理，如果没有ffwd层，直观上来说数据处理的太快了，模型没有在token交流后的思考过程
        #self.ffwd = FeedForward(n_embed)
        # 6.将4，5的multi head attention和feed-forward层堆叠起来作为一个block
        self.blocks = nn.Sequential(
            *[Block(n_heads=n_head,n_embed=n_embed) for _ in range(n_layer)],
            # 8. 在经过transformer所有计算之后，在输出logits线性层之前，使用layer norm进行标准化
            nn.LayerNorm(n_embed),
        )
        # 2.通常模型是主干网络backbone，最后的输出是head，表示预测结果，如二分类、多分类、回归等，也就是最后一层叫head
        self.lm_head = nn.Linear(n_embed,vocab_size) # lm_head是large language model head的缩写
        
    def forward(self,idx,targets=None):
        # 获取批次大小和给pos_emb使用的当前上下文长度(目前来看就是固定的block_size)
        B,T = idx.shape
        
        token_emb = self.token_embedding_table(idx)  #B,T,C
        # 3.使用位置嵌入层
        pos_emb = self.position_embedding_table(torch.arange(T,device=device))  #T,C
        # 3.将token嵌入和位置嵌入的信息相加作为输入，这样的聚合信息使得每个T既包含了token的语义信息，又包含了位置信息
        # 3.信息简单的相加在bigram model这样的模型中是很难体现效果的，因为这些token之间并没有产生信息交集，但是当引入了attention机制后，
        # 可以看到聚合了语义信息和位置信息开始发挥作用
        x = token_emb + pos_emb  #B,T,C 因为广播作用会将pos_emb扩展到B,T,C的形状，然后相加
        
        # 4.使用single-head self-attention层
        #x = self.sa_head(x) #B,T,C
        # 5.使用feed-forward层
        #x = self.ffwd(x) #B,T,C
        
        # 6.堆叠多个block （刚开始堆叠多层的时候效果并没有提升，这可能是因为我们开始接触到有一定deep的网络模型了，需要有一些方式来确保深层网络可持续优化）
        x = self.blocks(x) #B,T,C
        
        logits = self.lm_head(x)   # B,T,vocab_size
        
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            # 因为pytorch的cross_entropy函数要求输入(mini_batch,C)的格式，以bigram为例，C=vocab_size，实际有B*T个批次，所以需要将logits的形状变为(B*T,C)
            logits = logits.view(B*T,C)
            # targets也做相同的形状变换匹配输入格式
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        
        return logits,loss
    
    def generate(self,idx,max_new_tokens):
        # idx是shape为（B,T）的输入，每个T时间窗内为token的index
        for _ in range(max_new_tokens):
            # 4.使用了attention之后做的修改，只取最后block_size个token作为输入,因为限制了上下文长度
            idx_cond = idx[:,-block_size:] 
            logits,loss = self(idx_cond)
            # 取出最后一个时间步的输出作为输入的下一个token
            logits = logits[:,-1,:] # becomes shape (B,C)
            prob = F.softmax(logits,dim=-1) # becomes shape (B,C)
            # 随机采样一个token
            idx_next = torch.multinomial(prob,num_samples=1) # becomes shape (B,1)
            # 增加到序列中，作为下一次输入
            idx = torch.cat((idx,idx_next),dim=-1)
        return idx
      
# 构建模型实例
m = TransformerLanguageModel()
# 这里将模型参数移动到GPU上，如果cuda是可用的
model = m.to(device)

# 定义优化器
optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)

# 训练模型
for iter in range(max_iters):
    
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Iter {iter}, Train Loss: {losses['train']:.4f}, Test Loss: {losses['test']:.4f}")
    
    xb,yb = get_batch(split='train')
    
    logits,loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# 生成结果
context = torch.zeros((1,1), dtype=torch.long, device=device) # 将输入数据也放置在device上
model.eval()
print(decode(model.generate(idx=context,max_new_tokens=1000)[0].tolist()))