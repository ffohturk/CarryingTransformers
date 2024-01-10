import sys
import numpy as np
import torch
import copy
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import time
from requests import get
import os
import random
import math
import pickle
import argparse
import wandb

parser = argparse.ArgumentParser()

parser.add_argument("--split", type=float)
parser.add_argument("--weight_decay", type=float)
parser.add_argument("--num_layers", type=int)
parser.add_argument("--epochs", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--learning_rate", type=float)
parser.add_argument("--n_digits", default=3, type=int)
parser.add_argument("--output_dir", default="", type=str)

args = parser.parse_args()

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
  
    """Unembedding and softmax layer of the transformer
    """

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.ln = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.ln(x), dim=-1)

class Decoder(nn.Module):

    def __init__(self, attn, ffn, d_model, dropout):
        super().__init__()
        self.attn = attn
        self.ffn = ffn
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.out = None

    def forward(self, x, mask):

        # LayerNorm
        x1 = self.norm(x)

        # Attention and skip connection
        x = x + self.dropout(self.attn(x1, x1, x1, mask))

        # LayerNorm, Feedforward and skip connection
        self.out = x + self.dropout(self.ffn(self.norm(x)))

        return self.out

class DecoderStack(nn.Module):

    def __init__(self, layer, N):
        super().__init__()
        self.N = N
        self.norm = nn.LayerNorm(layer.d_model)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

def Attention(q, k, v, mask=None, dropout=None):

        """Attention mechanism with RoFormer 

        Returns:
            tuple: (Attention output, Attention weights)
        """

        # q, k, v are dims (batch_size, # heads, seq_len, d_{k,v})

        # Rotary Embedding
        m = torch.arange(k.shape[-2]).view(k.shape[-2], 1).to(q.device)
        t = torch.arange(k.shape[-1]).view(1, k.shape[-1])
        t = torch.exp( - ( 2 * np.log(10**4) / k.shape[-1] ) * torch.floor(t/2.) ).to(q.device)
        r1 = torch.cos(m * t)
        r2 = torch.sin(m * t)

        K = torch.cat((q, k, v))

        Kp = torch.einsum('ijkl, kl -> ijkl', K, r1)

        L = torch.kron(torch.eye(k.shape[-1]//2), torch.Tensor([[0,-1],[1,0]])).to(q.device)
        K = torch.einsum('ijkl, ml -> ijkm', K, L)

        Kp += torch.einsum('ijkl, kl -> ijkl', K, r2)

        Kp = Kp.view(-1, k.shape[0], k.shape[1], k.shape[2], k.shape[-1])

        q, k, v = Kp[0], Kp[1], Kp[2]

        A = torch.matmul(q, k.transpose(-2,-1)) * k.size(-1)**(-0.5)

        if mask is not None:
            A.masked_fill_(mask == 0, float('-inf'))

        O = F.softmax(A, dim=-1)

        if dropout is not None:
            O = dropout(O)

        return torch.matmul(O, v), O


class MultiHeadedAttention(nn.Module):

    """Multi-head attention from 'All you need is attention' paper (1706.03762)
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        self.d_k = d_model // h
        self.h = h
        self.attn = None
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.dropout =  nn.Dropout(p=dropout)

    def forward(self, query, keys, values, mask=None):

        batch_size = query.shape[0]

        x = [l(z).view(batch_size, -1, self.h, self.d_k).transpose(1, 2) for l, z in zip(self.linears, (query, keys, values))]

        y, self.attn = Attention(x[0], x[1], x[2], mask=mask, dropout=self.dropout)

        y = y.transpose(1,2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.linears[-1](y)

class FeedForward(nn.Module):

    """Feedforward with ReLU activation from 'All you need is attention' paper (1706.03762)
    """

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.out = None
        self.out_p = None

    def forward(self, x):
        self.out = self.relu(self.w1(x))
        self.out_p = self.w2(self.dropout(self.out))
        return self.w2(self.dropout(self.out))

class Embeddings(nn.Module):

    def __init__(self, src_vocab, d_model):
        super().__init__()
        self.Emb = nn.Embedding(src_vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.Emb(x) * np.sqrt(self.d_model)

def make_model(vocab, N = 6, d_model = 512, d_ff = 2048, h = 8, dropout = 0.1):

    """Make model accoring to 'All you need is attention paper' (ArXiv:1706.03762)

    Returns:
        model
    """

    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ffn = FeedForward(d_model, d_ff, dropout)

    model = DecoderTot(DecoderStack(Decoder( c(attn), c(ffn), d_model, dropout), N),
                           Embeddings(vocab, d_model),  Generator(d_model, vocab))

    for p in model.parameters():
        if p.dim() > 1: # This is there to not initialize the biases
            nn.init.xavier_uniform_(p)
    # print('# of parameters =', sum(p.nelement() for p in model.parameters())) # number of parameters in total

    return model

def GenerateDataset(ndig):

    """Generates the datasets required for training and testing. 

    Arguments:
        ndig (int): An integer

    Returns:
        tuple of torch tensors: (vocabulary size, \
                                input ndig digit sums, \
                                target ndig digit sums)               

    """

    P = 10**ndig

    data_add = []
    target_add = []

    stoi = {'0': 0, '1': 1, '2': 2,'3': 3,'4': 4,'5': 5,'6': 6,'7': 7,'8': 8,'9': 9,'+': 10,'=': 11}

    for i in range(P):
        for j in range(P):
            li = list(f'{i}')
            lj = list(f'{j}')
            lij = list(f'{i+j}')
            if i + j < P:
                if len(li) < ndig:
                    li = ['0'] * (ndig - len(li)) + li
                if len(lj) < ndig:
                    lj = ['0'] * (ndig - len(lj)) + lj
                if len(lij) < ndig:
                    lij = ['0'] * (ndig - len(lij)) + lij

                lsum = li + ['+'] + lj + lij
                lt = lsum[1:] + ['=']
                data_add.append([stoi[lsum[i]] for i in range(len(lsum))])
                target_add.append([stoi[lt[i]] for i in range(len(lt))])

    vocab = len(stoi)

    data_f = torch.LongTensor(data_add)
    target_f = torch.LongTensor(target_add)

    return vocab, data_f, target_f

class Dataset(torch.utils.data.Dataset):

    def __init__(self, inputs, target):
        self.inputs = inputs
        self.target = target

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        src = self.inputs[index]
        tgt = self.target[index]

        return src, tgt

def prepare(rank, world_size, data, target, batch_size, pin_memory=True, num_workers=0):

    dataset = Dataset(data, target)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
    # dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False)
    return dataloader

def cleanup():
    dist.destroy_process_group()

def run_epoch(data, loader, model, optimizer, device, status='train'):

    """Performs one epoch of training or evaluation

    Returns:
        torch tensor: [loss, accuracy per position, weight norm]
    """

    for batch in loader:

        src, tgt = batch[0].to(device), batch[1].to(device)

        num_digits = args.n_digits
        seq_len = src.shape[-1]
        mask = torch.tril(torch.ones(seq_len, seq_len))
        logits = model.forward(src, mask)[:, -(num_digits + 1):]
        tgt = tgt[:, -(num_digits + 1):]

        kl_loss = nn.CrossEntropyLoss()

        loss = kl_loss(logits.transpose(-1, -2), tgt) # We want inputs to be (bs, vocab_size, seq len), so needed a transpose. Targets are (bs, seq len) with values in [0, vocab_size]

        a = (torch.argmax(logits.detach(), dim=-1) == tgt).float()
        a = a.mean(dim=0).tolist()
        a.append(loss.detach().item())

        if status == 'train':
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        if status == 'eval':
            print('evaluating', data.shape[0])
            w2 = sum((p.data**2).sum() for p in model.parameters()).clone().detach().to('cpu')
            a.append(w2.item())
        
        pre_data = torch.tensor(a)
        data = torch.cat((data, pre_data.unsqueeze(0)), 0)

        del loss, tgt, src, logits

    return data

def train():

    os.mkdir(args.output_dir)

    dist.init_process_group("nccl")
    dev = dist.get_rank()
    world_size = dist.get_world_size()

    rank = dev % torch.cuda.device_count()

    # Generate Dataset

    vocab, data, target = GenerateDataset(ndig=args.n_digits)

    # Shuffle dataset and create splits
    generator = torch.Generator()
    generator.manual_seed(42)
    indices = torch.randperm(len(data), generator=generator)

    split = args.split

    len_train = int(split * len(data))

    src_train, tgt_train = data[indices[:len_train]], target[indices[:len_train]]
    src_test, tgt_test = data[indices[len_train:int(2*len_train)]], target[indices[len_train:int(2*len_train)]]

    batch_size = args.batch_size

    # Create dataloaders
    train_loader = prepare(rank=rank, world_size=world_size, data=src_train, target=tgt_train, batch_size=batch_size)
    eval_loader = prepare(rank=rank, world_size=world_size, data=src_test, target=tgt_test, batch_size=batch_size)

    # Model parameters

    d_model = 128
    d_ff = 128
    n_heads = 2
    n_layers = args.num_layers

    # Make model
    model = make_model(vocab, N = n_layers, d_model = d_model, d_ff = d_ff, h = n_heads, dropout = 0.1)

    model = model.to(rank)

    ddp_model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    # Training parameters and optimizer

    lr = args.learning_rate
    weight_decay = args.weight_decay    
    num_epochs = args.epochs

    optimizer = torch.optim.AdamW(ddp_model.parameters(),
                                  lr = lr,
                                  betas = (0.9, 0.98),
                                  eps=1e-8,
                                  weight_decay=weight_decay)

    # Tracking

    wandb.init(project="Generative Addition",
                     config={"lr": lr,
                             "split":split,
                             "layers": n_layers,
                             "weight decay": weight_decay,
                             "d_ff": d_ff,
                             "d_model": d_model
                             }
                        )

    # Training    

    for epoch in range(num_epochs):

        data = torch.tensor([])
        data_t = torch.tensor([])

        ddp_model.train()
        
        # Run one epoch
        data = run_epoch(data, loader=train_loader, model=ddp_model, optimizer=optimizer, device=rank, status='train')
        
        data = data.mean(dim=0)
        s = {}
        s['training loss'] = data[-1]
        z = [f'training acc. pos {i}' for i in range(args.n_digits + 1)]
        for i in range(len(z)):
            s[z[i]] = data[i]
        s['training acc.'] = data[:-1].mean()
        wandb.log(s,
            step=epoch
            )

        ddp_model.eval()
        with torch.no_grad():
            data_t = run_epoch(data_t, loader=eval_loader, model=ddp_model, optimizer=optimizer, device=rank, status='eval')
        
        data = data_t.mean(dim=0)
        s = {}
        s['test loss'] = data[-2]
        z = [f'test acc. pos {i}' for i in range(args.n_digits + 1)]
        for i in range(len(z)):
            s[z[i]] = data[i]
        s['test acc.'] = data[:-2].mean()
        s['norm weights squared'] = data[-1]
        wandb.log(s,
            step=epoch
            )

        # Save model every 500 epochs.
        if epoch % 500 == 0:
            outputFile = args.output_dir + '/model_n{!s}_s{!s}_w{!s}_epoch{!s}'.format(n_layers, split, weight_decay, epoch)

            torch.save({
                    'model': ddp_model.module.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, outputFile)

    cleanup()
    wandb.finish()

    outputFile = args.output_dir + '/model_n{!s}_s{!s}_w{!s}_final'.format(n_layers, split, weight_decay)

    torch.save({
            'model': ddp_model.module.state_dict(),
            'optimizer': optimizer.state_dict()
            }, outputFile)

if __name__ == "__main__":
    train()
