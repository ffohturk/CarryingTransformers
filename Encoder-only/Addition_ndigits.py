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
parser.add_argument("--ID", default=0, type=int)
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
    
    data = []
    target = []

    stoi = {'0': 0, '1': 1, '2': 2,'3': 3,'4': 4,'5': 5,'6': 6,'7': 7,'8': 8,'9': 9,'+': 10,'=': 11}

    for i in range(P):
        for j in range(P):
            if i + j < P:
                li = list(f'{i}')
                lj = list(f'{j}')
                lij = list(f'{i+j}')
                if len(li) < ndig:
                    li = ['0'] * (ndig - len(li)) + li
                if len(lj) < ndig:
                    lj = ['0'] * (ndig - len(lj)) + lj
                if len(lij) < ndig:
                    lij = ['0'] * (ndig - len(lij)) + lij

                lsum = li + ['+'] + lj + ['='] * ndig
                data.append([stoi[lsum[i]] for i in range(len(lsum))])
                target.append([stoi[lij[i]] for i in range(len(lij))])

    vocab = len(stoi) 
    data = torch.LongTensor(data)
    target = torch.LongTensor(target)

    return vocab, data, target

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

    """Prepares dataset for Distributed Data Parallel
    """

    dataset = Dataset(data, target)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
    
    return dataloader

def cleanup():
    dist.destroy_process_group()

def run_epoch(data_iter, model, label_smoothing, optimizer, scheduler, device, status = "train"):

    """Performs one epoch of training or evaluation

    Returns:
        tuple: (loss, accuracy per position, accuracy, weight norm)
    """

    Loss_P = torch.tensor([]).to(device)
    Acc_P = torch.tensor([]).to(device)
    Acc_PP = torch.tensor([]).to(device)
    SQ_W = torch.tensor([]).to(device)

    for i, (src, tgt) in enumerate(data_iter):

        src = src.to(device)
        tgt = tgt.to(device)

        length = tgt.shape[-1]

        logits = model.forward(src, None)[:, -length:, :]

        target = label_smoothing(tgt) # Output is [number of examples, vocab_out size]

        out = logits

        logits = logits.contiguous().view(-1, logits.shape[-1])

        kl_loss = nn.KLDivLoss(reduction="batchmean") 
        loss = kl_loss(logits, target)
        
        pre_acc = torch.argmax(torch.exp(out[:, -length:]), -1)

        acc = sum((pre_acc[i] == tgt[i, -length:]).float() for i in range(len(tgt))) / len(tgt)
        
        acc_p = sum((pre_acc[i] == tgt[i, -length:]).float().min() for i in range(len(tgt))) / len(tgt)

        Acc_P = torch.cat((Acc_P, acc.unsqueeze(0)), 0)
        Acc_PP = torch.cat((Acc_PP, acc_p.unsqueeze(0)), 0)
        Loss_P = torch.cat((Loss_P, loss.unsqueeze(0)), 0)
        
        if status == "train":
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
    
        del loss, acc, acc_p, pre_acc, out, logits

        if status == "eval":
            squared_weight_sum = sum((p.data**2).sum() for p in model.parameters()).clone().detach()
            SQ_W = torch.cat((SQ_W, squared_weight_sum.unsqueeze(0)), 0)

            del squared_weight_sum

    return Loss_P.mean(0).view(1), Acc_P.mean(0), Acc_PP.mean(0), SQ_W.mean(0)

class LabelSmoothing(nn.Module):

    """Standard label smoothing
    """
    
    def __init__(self, size, smoothing=0.0):
        super().__init__()
        self.size = size
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.Starget = None
    
    def forward(self, target):
        
        batches = target.size(0)
        Starget = (F.one_hot(target, self.size) * ( self.confidence - self.smoothing / ( self.size - 2 ) ) + ( self.smoothing / ( self.size - 2 ) ))
        Starget = Starget.contiguous().view(-1, self.size)
        self.Starget = Starget.clone().detach()
        return self.Starget #.view(batches, -1, self.size).to(device)


def modelgeneration():
  
    dist.init_process_group("nccl")
    dev = dist.get_rank()
    world_size = dist.get_world_size()
    
    rank = dev % torch.cuda.device_count()

    # Generate and shuffle train and test datasets

    vocab, data, target = GenerateDataset(ndig=args.n_digits)

    random.seed()
    z = list(zip(data.tolist(), target.tolist()))
    random.shuffle(z)

    z1, z2 = zip(*z)
    src_array_sh, tgt_array_sh = torch.LongTensor(list(z1)), torch.LongTensor(list(z2))

    # Dataset parameters

    batch_size = args.batch_size

    vocab = 12

    split = args.split

    n1 = int(split*len(src_array_sh))

    # Create splits

    src_train, src_test = src_array_sh[:n1], src_array_sh[n1:2*n1]
    tgt_train, tgt_test = tgt_array_sh[:n1], tgt_array_sh[n1:2*n1]

    # Create dataloaders

    dataloader_train = prepare(rank, world_size, src_train, tgt_train, batch_size)

    dataloader_test = prepare(rank, world_size, src_test, tgt_test, batch_size)
    
    # Model parameters

    n_layer = args.num_layers # number of layers
    d_model = 128 # model dimension, residual stream
    d_ff = d_model # dim intermediate feed-forward layer
    h = 2 # number of heads (doesnt impact # of params)

    model = make_model(vocab, N = n_layer, d_model = d_model, d_ff = d_ff, h = h)

    model = model.to(rank)

    # We use Distributed Data Parallel

    ddp_model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    # Training parameters

    epochs = args.epochs

    weight_decay = args.weight_decay

    lr = args.learning_rate
   
    smoother = LabelSmoothing(vocab, smoothing = 0.0)

    smootherEval = LabelSmoothing(vocab, smoothing = 0.0)

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr = lr, betas = (0.9, 0.98), eps = 1e-8, weight_decay=weight_decay)

    lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda = lambda step: 1 )

    loss_train = [] # Records training loss
    loss_test = [] # Records test loss
    acc_train = [] # Records training accuracy per position
    acc_full_train = [] # Records training accuracy
    acc_test = [] # Records test accuracy per position
    acc_full_test = [] # Records test accuracy
    sq_weights = [] # Records weight norm squared
    
    run = wandb.init(project="3-digit addition", 
                     group='DDP'.format(world_size), 
                     config={"lr": lr, "split":split, "layers": n_layer, 
                             "weight decay": weight_decay, "d_ff": d_ff, "d_model": d_model})

    for j in range(epochs):
        ddp_model.train()

        dataloader_train.sampler.set_epoch(j)
        dataloader_test.sampler.set_epoch(j)

        l, a, af, _   = run_epoch( dataloader_train, ddp_model, smoother, optimizer, lr_scheduler, rank, status = "train" )
        
        # Gather results (loss, accuracy per position, accuracy) from different GPUs
        torch.cuda.set_device(rank)
        tensor_out_l = torch.zeros([world_size] + list(l.shape)).to(rank)
        dist.all_gather_into_tensor(tensor_out_l, l)

        tensor_out_a = torch.zeros([world_size] + list(a.shape)).to(rank)
        dist.all_gather_into_tensor(tensor_out_a, a)

        tensor_out_af = torch.zeros([world_size] + list(af.shape)).to(rank)
        dist.all_gather_into_tensor(tensor_out_af, af)

        loss_train.append(tensor_out_l.mean(0).to('cpu').item())
        acc_train.append(tensor_out_a.mean(0).to('cpu'))
        acc_full_train.append(tensor_out_af.mean(0).to('cpu'))

        ddp_model.eval()
        with torch.no_grad():
            lp, ap, apf, sq_w = run_epoch( dataloader_test, ddp_model, smootherEval, optimizer, lr_scheduler, rank, status = "eval" )
            
            # Gather results (loss, accuracy per position, accuracy, weight norm) from different GPUs
            torch.cuda.set_device(rank)
            tensor_out_lp = torch.zeros([world_size] + list(lp.shape)).to(rank)
            dist.all_gather_into_tensor(tensor_out_lp, lp)

            tensor_out_ap = torch.zeros([world_size] + list(ap.shape)).to(rank)
            dist.all_gather_into_tensor(tensor_out_ap, ap)

            tensor_out_apf = torch.zeros([world_size] + list(apf.shape)).to(rank)
            dist.all_gather_into_tensor(tensor_out_apf, apf)

            tensor_out_sq_w = torch.zeros([world_size] + list(sq_w.shape)).to(rank)
            dist.all_gather_into_tensor(tensor_out_sq_w, sq_w)

            sq_weights.append(tensor_out_sq_w.mean(0).data.to('cpu'))
            loss_test.append(tensor_out_lp.mean(0).to('cpu').item())
            acc_test.append(tensor_out_ap.mean(0).to('cpu'))
            acc_full_test.append(tensor_out_apf.mean(0).to('cpu'))

        wandb.log({"train loss": tensor_out_l.mean(0).item(),
                    "train acc": tensor_out_af.mean(0),
                    "test loss": tensor_out_lp.mean(0).item(),
                    "test acc": tensor_out_apf.mean(0),
                    "Norm-squared weights": tensor_out_sq_w.mean(0)
                    })

    cleanup()
    wandb.finish()
    dir = args.output_dir
    outputFile = dir + 'model_n{!s}_s{!s}_w{!s}_{!s}'.format(n_layer, split, weight_decay, args.ID)

    torch.save({
            'model': ddp_model.module.state_dict(),
            'optimizer': optimizer.state_dict()
            }, outputFile)
    
    DATA = [loss_train, loss_test, acc_train, acc_test, acc_full_train, acc_full_test, sq_weights]

    outputFile = dir + 'DATA_n{!s}_s{!s}_w{!s}_{!s}.data'.format(n_layer, split ,weight_decay, args.ID)
    fw = open(outputFile, 'wb')
    pickle.dump(DATA, fw)
    fw.close()

if __name__ == "__main__":
    modelgeneration()
