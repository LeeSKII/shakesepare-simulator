import math
import inspect
from dataclasses import dataclass
import os
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

# TODO: dropout,flash attention,init weights.

@dataclass
class GPTConfig:
    '''Config of the nano GPT model'''
    batch_size = 8 # 迭代一次使用多少批数据
    block_size = 32 # 上下文长度
    vocab_size = None # 词表大小
    n_layer = 3 # transformer中block的数量
    n_head = 8 # transformer中head的数量
    n_embd = 256 # embedding的维度
    dropout = 0.1 # dropout的概率
    max_iters = 5001 # 训练的最大步数
    eval_interval = 500 # 每隔多少步进行一次验证
    learning_rate = 3e-4 # 扩大规模后进一步降低学习率
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # 是否使用GPU
    bias = True # 是否使用bias


def prepare_data(config: GPTConfig):
    '''准备数据'''
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
    
    config.vocab_size = vocab_size
    
    return train_data,val_data,encode,decode

# 定义获取批数据函数
def get_batch(split,train_data,val_data,batch_size,block_size,device):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    # 这里主要是如果cuda是可用的，那么将数据放在GPU上会运行的更快
    x,y = x.to(device),y.to(device)
    return x, y
  
@torch.no_grad()
def estimate_loss(model,train_data,val_data,config: GPTConfig):
    out = {}
    model.eval()
    for split in ['train','test']:
        losses = torch.zeros(config.eval_interval)
        for k in range(config.eval_interval):
            X,Y = get_batch(split,train_data,val_data,config.batch_size,config.block_size,config.device)
            _,loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()    
    model.train()
    return out
  

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class MLP(nn.Module):
    def __init__(self,config:GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd,4*config.n_embd,bias=config.bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4*config.n_embd,config.n_embd,bias=config.bias)
        
    def forward(self,x):
        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out)
        return out
        
          
class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.dropout = config.dropout
        
        # k,q,v三个矩阵存储在一起
        # self.c_attn = nn.Linear(config.n_embd,config.n_embd * 3,bias=config.bias)
        
        self.q = nn.Linear(config.n_embd,config.n_embd,bias=False)
        self.k = nn.Linear(config.n_embd,config.n_embd,bias=False)
        self.v = nn.Linear(config.n_embd,config.n_embd,bias=False)
        self.register_buffer('mask_tril',torch.tril(torch.ones(config.block_size,config.block_size)))      
        
    def forward(self,x):
        b,t,c = x.shape
        
        query = self.q(x) # b,t,c
        key = self.k(x) # b,t,c
        value = self.v(x) # b,t,c
        
        # 使用多头注意力机制
        single_head_size = c//self.n_head
        query = query.view(b,t,self.n_head,single_head_size).transpose(1,2) # b,h,t,c/h
        key = key.view(b,t,self.n_head,single_head_size).transpose(1,2) # b,h,t,c/h
        value = value.view(b,t,self.n_head,single_head_size).transpose(1,2) # b,h,t,c/h
        
        wei = query @ key.transpose(-2,-1) * single_head_size ** -0.5 # b,t,c
        masked_wei = wei.masked_fill(self.mask_tril[:t,:t]==0,float('-inf')) # b,t,c
        attn = masked_wei.softmax(dim=-1) # b,t,c
        
        out = attn @ value # b,t,c        
        
        # 将转置后的不连续内存张量连续化再view成原始形状
        out = out.transpose(1,2).contiguous().view(b,t,c) # b,t,c 
       
        return out


class Block(nn.Module):
    def __init__(self,config:GPTConfig):
        super().__init__()
        self.config = config
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        
    def forward(self,x):
        # + 是残差连接操作
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
        

class GPT(nn.Module):
    def __init__(self,config:GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.n_embd is not None
        self.config = config
        self.n_embd = config.n_embd
        self.wte = nn.Embedding(config.vocab_size,config.n_embd) # B,T,C
        self.wpe = nn.Embedding(config.block_size,config.n_embd)
        
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln = nn.LayerNorm(config.n_embd)
        
        self.lm_head = nn.Linear(config.n_embd,config.vocab_size,bias=config.bias) # B,T,vocab_size
        
        
    def forward(self,idx,target = None):
        b,t = idx.shape
        token_embed = self.wte(idx) #b,t,n_embd
        pos = torch.arange(0,t,device=self.config.device,dtype=torch.long) # t
        position_embed = self.wpe(pos) #t,n_embd
        x = token_embed + position_embed #b,t,n_embd
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln(x)
        logits = self.lm_head(x)
        
        if target is None:
           loss = None
        else:
            # 因为cross_entropy的输入是(mini_batch,C)的，所以这里需要将logits和target转化为这种格式
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits,target)
        
        return logits,loss
    
    @torch.no_grad()
    def generate(self,idx,max_new_tokens,config:GPTConfig):
        # idx是shape为（B,T）的输入，每个T时间窗内为token的index
        for _ in range(max_new_tokens):
            # 4.使用了attention之后做的修改，只取最后block_size个token作为输入,因为限制了上下文长度
            idx_cond = idx[:,-config.block_size:] 
            logits,_ = self(idx_cond)
            # 取出最后一个时间步的输出作为输入的下一个token
            logits = logits[:,-1,:] # becomes shape (B,C)
            prob = F.softmax(logits,dim=-1) # becomes shape (B,C)
            # 随机采样一个token
            idx_next = torch.multinomial(prob,num_samples=1) # becomes shape (B,1)
            # 增加到序列中，作为下一次输入
            idx = torch.cat((idx,idx_next),dim=-1)
        return idx

if __name__ == '__main__':
    config = GPTConfig()
    print(f'Using device: {config.device}')
    
    torch.manual_seed(1337)
    #------------Preparing data------------------------  
    train_data,val_data,encode,decode = prepare_data(config)
   
    # 定义模型
    model = GPT(config)
    model = model.to(config.device)
    
    # 定义优化器
    optimizer = torch.optim.AdamW(model.parameters(),lr=config.learning_rate)

    start_time = time.time()
    # 训练模型
    for iter in range(config.max_iters):
        
        if iter % config.eval_interval == 0:
            immediate_time = time.time()
            
            losses = estimate_loss(model,train_data,val_data,config)
            print(f"Iter {iter}, Train Loss: {losses['train']:.4f}, Test Loss: {losses['test']:.4f},Time consume: {immediate_time-start_time}")
            start_time = immediate_time
        
        xb,yb = get_batch(split='train',train_data=train_data,val_data=val_data,batch_size=config.batch_size,block_size=config.block_size,device=config.device)
      
        logits,loss = model(xb,yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
    # 生成结果
    context = torch.zeros((1,1), dtype=torch.long, device=config.device) # 将输入数据也放置在device上
    model.eval()
    print(decode(model.generate(idx=context,max_new_tokens=1000,config=config)[0].tolist()))
