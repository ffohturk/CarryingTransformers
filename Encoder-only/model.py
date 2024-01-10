import numpy as np
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F

class DecoderTot(nn.Module):

  def __init__(self, decoder, embed, generator):
    super().__init__()
    self.embed = embed
    self.gen = generator
    self.decoder = decoder
    self.generator = generator

  def forward(self, src, mask):
    return self.generator(self.decoder(self.embed(src), mask))


class Generator(nn.Module):

  def __init__(self, d_model, vocab_size):
    super().__init__()
    self.ln = nn.Linear(d_model, vocab_size)
    self.out = None

  def forward(self, x):
    self.out = self.ln(x)
    return F.log_softmax(self.ln(x), dim=-1)

class Decoder(nn.Module):
    
    def __init__(self, attn, ffn, d_model, dropout, ablation_data_dec):
        super().__init__()
        self.attn = attn
        self.ffn = ffn
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.out = None
        self.out_a = None
        self.n_ab_att = ablation_data_dec[0]
        self.n_ab_ffn = ablation_data_dec[1]
        
    def forward(self, x, mask, n):
        
        x1 = self.norm(x)

        if n in self.n_ab_att:
            x = x + 0*self.dropout(self.attn(x1, x1, x1, n, mask))
        else:
            x = x + self.dropout(self.attn(x1, x1, x1, n, mask))

        self.out_a = x
        
        if n in self.n_ab_ffn:
            self.out = x + 0*self.dropout(self.ffn(self.norm(x), n))
        else:
            self.out = x + self.dropout(self.ffn(self.norm(x), n))
        
        return self.out

class DecoderStack(nn.Module):
    
    def __init__(self, layer, N):
        super().__init__()
        self.N = N
        self.norm = nn.LayerNorm(layer.d_model)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        
    def forward(self, x, mask):
        n = 0
        for layer in self.layers:
            x = layer(x, mask, n)
            n += 1
        return self.norm(x)

def Attention(q, k, v, n, ablation_data_att, mask=None, dropout=None):
            
            M_apply, n_ab_head, ab_head, ab_head_row = ablation_data_att
            
            ### -- Softmax Attention -- ###
            
            # q, k, v are dims (batch_size, # heads, seq_len, d_{k,v}) 
            
            m = torch.arange(k.shape[-2]).view(k.shape[-2], 1)
            t = torch.arange(k.shape[-1]).view(1, k.shape[-1])
            t = torch.exp( - ( 2 * np.log(10**4) / k.shape[-1] ) * torch.floor(t/2) )
            r1 = torch.cos(m * t)
            r2 = torch.sin(m * t)
            
            K = torch.cat((q, k, v))
            
            Kp = torch.einsum('ijkl, kl -> ijkl', K, r1)
            
            L = torch.kron(torch.eye(k.shape[-1]//2), torch.Tensor([[0,-1],[1,0]]))
            K = torch.einsum('ijkl, ml -> ijkm', K, L)
            
            Kp += torch.einsum('ijkl, kl -> ijkl', K, r2)
            
            Kp = Kp.view(-1, k.shape[0], k.shape[1], k.shape[2], k.shape[-1])
            
            q, k, v = Kp[0], Kp[1], Kp[2]
            
            A = torch.matmul(q, k.transpose(-2,-1)) * k.size(-1)**(-0.5)
            
            if M_apply:

                range_in = np.arange(10)
                range_in_1 = np.delete(range_in, ab_head_row[n][0])
                range_in_2 = np.delete(range_in, ab_head_row[n][1])
                index_p1 = torch.tensor(range_in_1)
                index_p2 = torch.tensor(range_in_2)
                Ab_mask = torch.zeros_like(A)
                Ab_mask[:, 0, :, :].index_fill_(-2, index_p1, 1)
                Ab_mask[:, 1, :, :].index_fill_(-2, index_p2, 1)
            
            if mask is not None:
                # mask = mask.unsqueeze(1)
                A.masked_fill_(mask == 0, float('-inf'))

            O = F.softmax(A, dim=-1)

            if dropout is not None:
                O = dropout(O)
            
            if n in n_ab_head and M_apply:
                Ab = Ab_mask
                O = torch.einsum('ijkl, ijkl -> ijkl', O, Ab)
                
            if n in n_ab_head:
                Ab = ab_head[n]
                O = torch.einsum('ijkl, j -> ijkl', O, Ab)
            
            return torch.matmul(O, v), O, A


class MultiHeadedAttention(nn.Module):
    
    def __init__(self, h, d_model, ablation_data_attention, dropout=0.1):
        super().__init__()
        self.d_k = d_model // h
        self.h = h
        self.attn = None
        self.attnA = None
        self.out = None
        self.out_A = None
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.dropout =  nn.Dropout(p=dropout)
        self.ab = ablation_data_attention
        
    
    def forward(self, query, keys, values, n, mask=None):
        
        batch_size = query.shape[0]
        
        x = [l(z).view(batch_size, -1, self.h, self.d_k).transpose(1, 2) for l, z in zip(self.linears, (query, keys, values))]
        
        y, self.attn, self.attnA = Attention(x[0], x[1], x[2], n, ablation_data_att=self.ab, mask=mask, dropout=self.dropout)

        y = y.transpose(1,2).contiguous().view(batch_size, -1, self.h * self.d_k)
        
        self.out_A = y

        self.out = self.linears[-1](y)

        return self.linears[-1](y)
    
class FeedForward(nn.Module):
    
    def __init__(self, d_model, d_ff, dropout, ablation_data_ffn):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.out = None
        self.out_p = None
        self.FFN_needle = ablation_data_ffn
    
    def forward(self, x, n):
        A = self.w1(x)
        if n == 1:
            for i in range(A.shape[-1]):
                if i in self.FFN_needle:
                    A[:, :, i] *= 0
        self.out = self.relu(A)
        self.out_p = self.w2(self.dropout(self.out))
        return self.w2(self.dropout(self.out))

class Embeddings(nn.Module):
    
    def __init__(self, src_vocab, d_model):
        super().__init__()
        self.Emb = nn.Embedding(src_vocab, d_model)
        self.d_model = d_model
        self.out_e = None
    
    def forward(self, x):
        self.out_e = self.Emb(x) * np.sqrt(self.d_model)
        return self.Emb(x) * np.sqrt(self.d_model)

def make_model(vocab, N, d_model, d_ff, h, dropout, ablation_data):
    
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, ablation_data[0])
    ffn = FeedForward(d_model, d_ff, dropout, ablation_data[1])
    
    model = DecoderTot(DecoderStack(Decoder( c(attn), c(ffn), d_model, dropout, ablation_data[2]), N),
                           Embeddings(vocab, d_model),  Generator(d_model, vocab))
    
    for p in model.parameters():
        if p.dim() > 1: # This is there to not initialize the biases
            nn.init.xavier_uniform_(p)
    
    return model