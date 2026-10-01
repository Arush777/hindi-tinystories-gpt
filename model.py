import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def swiglu(x,W1,W2,W3):
  gate = W1(x)*torch.sigmoid(W1(x)) #silu W1(x)
  #W1,W3 are of size B,T,d_ff where d_ff ->8d/3 (thumb rule) d is d_mode
  value = W3(x) #value


  return W2(gate*value)

def rmsnorm(x,weights,eps=1e-5):

  mean_sq = (x * x).mean(dim=-1,keepdim = True)
  rms = torch.sqrt(mean_sq+eps)

  x_norm = x / rms
  y = x_norm * weights #RMS norm weights

  return y


import math
import torch
import torch.nn as nn


class MultiheadAttentionwithRoPE(nn.Module):
  def __init__(self,d_model,num_heads,theta=10000.0):
    super().__init__()

    assert d_model % num_heads == 0
    self.num_heads = num_heads
    self.d_model = d_model
    self.theta = theta

    self.d_in = d_model
    self.d_k = d_model
    self.d_v = d_model

    self.d_k_head = d_model // num_heads
    self.d_v_head = d_model // num_heads

    # weights

    self.W_q = nn.Linear(self.d_in,self.d_k,bias=False)
    self.W_k = nn.Linear(self.d_in,self.d_k,bias=False)
    self.W_v = nn.Linear(self.d_in,self.d_k,bias=False)
    self.W_o = nn.Linear(self.d_v,self.d_model,bias=False)

  def rope(self,x,token_positions):

    d_k = x.shape[-1];
    #split into pairs (even dims, odd dims)
    # (B, H, T, d_k/2)
    # TELLING THE last dimension start from 0 skip 2 and start from 1 skip 2
    x_even = x[...,0::2] # (B, H, T, d_k/2)
    x_odd  = x[...,1::2] # (B, H, T, d_k/2)

    # 2) frequency for each pair index: theta^(-2j/d_k)

    j = torch.arange(d_k//2 , device = x.device ,dtype = torch.float32)  # (d_k/2,) [0 1 2 3 ... d_k//2]
    theta_j = self.theta ** (-2.0 * j / d_k)  # (d_k/2,)

    pos = token_positions.to(torch.float32)[..., None]   # (T, 1) or (B, T, 1)
    angles = pos * theta_j      # (T, d_k/2) or (B, T, d_k/2) -> (T,1) * (theta * (-2j/d) {j -> [0 1 2  d_k//2]})

    cos = torch.cos(angles).to(x.dtype) # B,T,d_k//2
    sin = torch.sin(angles).to(x.dtype) # B,T,d_k//2

    while cos.ndim < x_even.ndim:  #(B,T,d/2) -> #B,1,T,d/2
      cos = cos.unsqueeze(-3)
      sin = sin.unsqueeze(-3)

    y_even = x_even * cos - x_odd * sin # B,H,T,d/2 * B,1,T,d/2
    y_odd  = x_even * sin + x_odd * cos

    y = torch.empty_like(x)   # (B, H, T, d_k) (create a empty array of size of x)
    y[..., 0::2] = y_even
    y[..., 1::2] = y_odd
    return y

  def forward(self,in_features,token_positions=None):
    B,T,d_in = in_features.shape; # shape is an attribute not a method so no brackets

    #projects

    #W_q is (d_model , d_in) -> W_q * x(B,T,d_in) > Q -> B,T,d_model
    Q = self.W_q(in_features) # W*x
    K = self.W_k(in_features)
    V = self.W_v(in_features)

    # we get B,T,d_model -> B,T,H,d_k -> tranpose->B,H,T,d_k
    # dividing Q,K,V projections for different heads
    Q = Q.view(B,T ,self.num_heads,self.d_k_head).transpose(1,2);
    K = K.view(B,T ,self.num_heads,self.d_k_head).transpose(1,2);
    V = V.view(B,T ,self.num_heads,self.d_v_head).transpose(1,2);

     # RoPE on Q and K only (not V)
     #(if None then u are giving 0 to T-1 already)
    if token_positions is None:
      token_positions = torch.arange(T, device=in_features.device)   # (T,)

    Q = self.rope(Q, token_positions)
    K = self.rope(K, token_positions)

    #SDPA

    scores = Q @ K.transpose(-2,-1) #B,H,T,D_K * B,H,d_k,T -> B,H,T,T
    normalized_scores = scores / math.sqrt(self.d_k_head)

    causal_mask = torch.tril(torch.ones(T,T,device = in_features.device ,dtype=torch.bool)) #T,T
    normalized_scores = normalized_scores.masked_fill(~causal_mask,-1e9) #B,H,T,T
    weights = normalized_scores.softmax(dim=-1)
    heads = weights @ V #B,H,T,T * B,H,T,d_k -> B,H,T,d_k

    mha = heads.transpose(1,2).reshape(B,T,self.num_heads*self.d_v_head) #b,h,t,d_k -> b,t,h,d_k -> b,t,d_v (go from left to right)
    out = self.W_o(mha) #B,T,d_model * B,T,d_v -> #B,T,d_model

    return out



class TransformerBlock(nn.Module):
  def __init__(self,d_model,num_heads,d_ff,theta = 10000.0):
    super().__init__()


    #attn
    self.attn = MultiheadAttentionwithRoPE(d_model,num_heads,theta)
    # pre norm attn
    self.norm1_w = nn.Parameter(torch.ones(d_model))
    self.theta = theta


   #  SwiGlu
    self.W1 = nn.Linear(d_model,d_ff,bias=False)
    self.W3 = nn.Linear(d_model,d_ff,bias = False)

    self.W2 = nn.Linear(d_ff,d_model,bias = False)

    self.norm2_w = nn.Parameter(torch.ones(d_model))

  def forward(self,x,token_positions=None):

    residual = x
    x_rms = rmsnorm(x,self.norm1_w)
    attn_x = self.attn(x_rms,token_positions)

    x = residual + attn_x

    residual = x

    x_rms_1 = rmsnorm(x,self.norm2_w)
    x_swiglu = swiglu(x_rms_1,self.W1,self.W2,self.W3)

    x = residual + x_swiglu

    return x




import torch.nn.functional as F

class GPT(nn.Module):
  def __init__(self,vocab_size,max_seq_len,num_layers,d_model,num_heads,d_ff,theta = 10000.0):
    super().__init__()

    self.max_seq_len = max_seq_len

    self.emb_tokens = nn.Embedding(vocab_size,d_model)
    self.blocks = nn.ModuleList(
        [TransformerBlock(d_model,num_heads,d_ff,theta) for _ in range(num_layers)]
    )


    self.norm_final_weights = nn.Parameter(torch.ones(d_model))
    self.lm_head = nn.Linear(d_model,vocab_size,bias=False)

  def forward(self,token_ids,targets = None,token_position = None):
    B,T  = token_ids.shape
    x = self.emb_tokens(token_ids)

    for block in self.blocks:
      x = block(x) # default token positions go 0,1,.. T-1 in the MHA code present

    x = rmsnorm(x,self.norm_final_weights)

    logits = self.lm_head(x)

    #Training code
    loss = None

    if targets is not None:
      pred = logits[:,:-1,:]  #B,T-1,V (vocab-size)
      true = targets[:,1:] #B,T-1 (next token pred)

      loss = F.cross_entropy(pred.reshape(-1,pred.shape[-1]),true.reshape(-1)) # loss between pred of the next token and the true has the next token

    return logits,loss

  @torch.no_grad()

  def generate(self,token_ids,n_new):
    for _ in range(n_new):
      context = token_ids[:,-self.max_seq_len:]
      logits,_ = self(context) #B,T,vocab_size

      last = logits[:,-1,:] #(B,vocab_size) last word only

      probs = last.softmax(dim=-1)
      next_id = torch.multinomial(probs,num_samples=1) #B,1 picsk index at random where each index chance is its probability
      #for greedy last.argmax(dim=-1,keepdim=True)
      token_ids = torch.cat([token_ids,next_id],dim=1) # B, T+1

    return token_ids
