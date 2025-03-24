from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 384
    

class CausalSelfAttention(nn.Module):
    def __init__(self,config:GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.c_attn = nn.Linear(config.n_embd,config.n_embd*3)
        self.c_proj = nn.Linear(config.n_embd,config.n_embd)
        
        # 实际应该称呼为'mask'更合适，这里命名是为了和openai gpt-2代码保持一致
        self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size)).view(1,1,config.block_size,config.block_size))
        
    def forward(self,x):
        B,T,C = x.size()
        # 这里相当于用了一个全连接层将x的C维数扩展到了3*C，然后再切分成3个C维度的矩阵分别为q,k,v
        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd,dim=2)
        # B batch， T sequence length, nh number of head , hs head size
        # 拆分为多头注意力
        q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # B,nh,T,hs
        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # B,nh,T,hs
        v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # B,nh,T,hs
        # 计算token之间的注意力
        att = (q @ k.transpose(-1,-2)) * (1.0/math.sqrt(k.size(-1)))  # B,nh,T,T
        att = att.masked_fill(self.bias[:,:,:T,:T]==0,float('-inf')) # masking
        att = F.softmax(att,dim=-1) # attention, B,nh,T,T
        value = att @ v  # B,nh,T,T @ B,nh,T,hs -> B,nh,T,hs
        value = value.transpose(1,2).contiguous().view(B,T,C) # B,T,C
        # 在经过一次投影加到残差层上
        y = self.c_proj(value) # B,T,C
        return y

# 在论文中，这层是full connected feed-forward network，位于attention之后，
# 目的是独立的作用于每个时间步上的token，让所有的token在互相交流之后有一个类似独立思考的过程。
class MLP(nn.Module):
    def __init__(self,config:GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd,config.n_embd*4,bias=True)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.n_embd*4,config.n_embd,bias=True)
    
    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
        

class Block(nn.Module):
    def __init__(self,config:GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        # self attention layer
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        # feed forward layer
        self.mlp = MLP(config)
        
    def forward(self,x):
        # +号意味着使用残差连接，但是在计算self-attention之前计算layer norm，这是因为希望保持有一条梯度是直接从输入的x到输出的结果y的,保持梯度的纯粹性
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
    
    
class GPT(nn.Module):
    def __init__(self,config: GPTConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            # token embedding
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # position embedding
            wpe = nn.Embedding(config.block_size,config.n_embd),
            # transformer blocks
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # final layer norm according to GPT-2 paper
            ln_f = nn.LayerNorm(config.n_embd) 
        ))
        # final linear layer for language model head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    