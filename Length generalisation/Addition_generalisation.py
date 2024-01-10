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
parser.add_argument("--n_digits_gen", default=6, type=int)
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

def make_model(vocab, N = 6, d_model = 512, d_ff = 2048, h_a = 8, dropout = 0.1):
    
    """Make model accoring to 'All you need is attention paper' (ArXiv:1706.03762)

    Returns:
        model
    """

    c = copy.deepcopy
    attn = MultiHeadedAttention(h_a, d_model)
    ffn = FeedForward(d_model, d_ff, dropout)
    
    model = DecoderTot(DecoderStack(Decoder( c(attn), c(ffn), d_model, dropout), N),
                           Embeddings(vocab, d_model),  Generator(d_model, vocab))
    
    for p in model.parameters():
        if p.dim() > 1: # This is there to not initialize the biases
            nn.init.xavier_uniform_(p)
    
    return model

def GenerateDataset(ndig: int, n_extra: int):

    """Generates the datasets required for training and testing. 

    Arguments:
        ndig (int): An integer
        n_extra (int): An integer

    Returns:
        tuple of torch tensors: (vocabulary size, \
                                input ndig digit sums, \
                                target ndig digit sums, \
                                input ndig+n_extra digit sums, \
                                target ndig+n_extra digit sums)               

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

                # We pad the integers with 0s till they have length ndig + n_extra

                if len(li) < ndig + n_extra:
                    li = ['0'] * (ndig + n_extra - len(li)) + li
                if len(lj) < ndig + n_extra:
                    lj = ['0'] * (ndig + n_extra - len(lj)) + lj
                if len(lij) < ndig + n_extra:
                    lij = ['0'] * (ndig + n_extra - len(lij)) + lij

                lsum = li + ['+'] + lj + ['='] * (ndig + n_extra)
                data.append([stoi[lsum[i]] for i in range(len(lsum))])
                target.append([stoi[lij[i]] for i in range(len(lij))])

    vocab = len(stoi) 
    data = torch.LongTensor(data)
    target = torch.LongTensor(target)

    data_f = []
    target_f = []

    # Generate 20000 examples of (ndig + n_extra) digit sums.

    P_f = 10**(ndig + n_extra) - 1

    k = 0
    while k < 20000:
        i = torch.randint(P_f, size=(1,)).item()
        j = torch.randint(P_f, size=(1,)).item()
        if i + j < P_f + 1: # We want the sums to such that the sum is less then ndig 
            li = list(f'{i}')
            lj = list(f'{j}')
            lij = list(f'{i+j}')

            # We pad the integers with 0s till they have length ndig + n_extra

            if len(li) < ndig + n_extra:
                li = ['0'] * (ndig + n_extra - len(li)) + li
            if len(lj) < ndig + n_extra:
                lj = ['0'] * (ndig + n_extra - len(lj)) + lj
            if len(lij) < ndig + n_extra:
                lij = ['0'] * (ndig + n_extra - len(lij)) + lij

            lsum = li + ['+'] + lj + ['='] * (ndig + n_extra)
            data_f.append([stoi[lsum[i]] for i in range(len(lsum))])
            target_f.append([stoi[lij[i]] for i in range(len(lij))])
            k += 1

    data_f = torch.LongTensor(data_f)
    target_f = torch.LongTensor(target_f)

    return vocab, data, target, data_f, target_f

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
    # sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False)
    
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

        logits = model.forward(src, None)[:, -length:]

        target = label_smoothing(tgt) # Output is [number of examples, vocab_out size]

        out = logits

        logits = logits.contiguous().view(-1, logits.shape[-1])

        kl_loss = nn.KLDivLoss(reduction="batchmean") 
        loss = kl_loss(logits, target)
        
        pre_acc = torch.argmax(torch.exp(out[:, -length:]), -1)

        # Accuracy per position
        acc = sum((pre_acc[i] == tgt[i, -length:]).float() for i in range(len(tgt))) / len(tgt)

        # Accuracy (correctness of output)
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

            squared_weight_sum = sum((torch.abs(p.data)**2).sum() for p in model.parameters()).clone().detach()
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

        return self.Starget

print('--- Generating data ---')

n_extra = args.n_digits_gen - args.n_digits
vocab, data, target, data_f, target_f = GenerateDataset(ndig=args.n_digits, n_extra=n_extra)

print('--- Finished generating data ---')

# Shuffling datasets

random.seed()
z = list(zip(data.tolist(), target.tolist()))
random.shuffle(z)

z1, z2 = zip(*z)
src_array_sh, tgt_array_sh = torch.LongTensor(list(z1)), torch.LongTensor(list(z2))

def modelgeneration():

    """In here we define the model, training and test datasets and everything 
    else needed for training the model.
    """
    
    rank = torch.device('cuda')
    world_size = 0

    # Dataset parameters

    batch_size = args.batch_size

    vocab = 12

    split = args.split

    n1 = int(split*len(src_array_sh))
    priming_examples = 100 # Number of priming examples

    # Construct train and test set for n_digit sums.
    src_train, src_test = src_array_sh[:n1], src_array_sh[n1:2*n1]
    tgt_train, tgt_test = tgt_array_sh[:n1], tgt_array_sh[n1:2*n1]

    # Construct train and test set for n_digit_gen sums.
    src_long, tgt_long = data_f[:priming_examples], target_f[:priming_examples]
    src_test_long, tgt_test_long = data_f[priming_examples:], target_f[priming_examples:]

    # Combine priming examples with n_digit sums in training set.
    src_train = torch.cat((src_train, src_long), 0)
    tgt_train = torch.cat((tgt_train, tgt_long), 0)

    # Shuffle priming examples with training set
    random.seed()
    z = list(zip(src_train.tolist(), tgt_train.tolist()))
    random.shuffle(z)

    z1, z2 = zip(*z)
    src_train, tgt_train = torch.LongTensor(list(z1)), torch.LongTensor(list(z2))

    # Construct data_loaders for train, test and test on longer digit sums

    dataloader_train = prepare(rank, world_size, src_train, tgt_train, batch_size)

    dataloader_test = prepare(rank, world_size, src_test, tgt_test, batch_size)

    dataloader_test_long = prepare(rank, world_size, src_test_long, tgt_test_long, batch_size)
    
    # Model parameters

    n_layer = args.num_layers # number of layers
    d_model = 128 # model dimension, residual stream
    d_ff = d_model # dim intermediate feed-forward layer
    h_a = 2 # number of heads in attention (doesnt impact # of params)

    model = make_model(vocab, N = n_layer, d_model = d_model, d_ff = d_ff, h_a = h_a)

    model = model.to(rank)

    print('--- model initialized ---')

    # Training parameters and optimizer

    epochs = args.epochs

    weight_decay = args.weight_decay

    lr = args.learning_rate
   
    smoother = LabelSmoothing(vocab, smoothing = 0.0)

    smootherEval = LabelSmoothing(vocab, smoothing = 0.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, betas = (0.9, 0.98), eps = 1e-8, weight_decay=weight_decay)

    lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda = lambda step: 1 )

    loss_train = [] # Records training loss
    loss_test = [] # Records test loss
    acc_train = [] # Records training accuracy per position
    acc_full_train = [] # Records training accuracy
    acc_test = [] # Records test accuracy per position
    acc_full_test = [] # Records test accuracy
    sq_weights = [] # Records weight norm squared
    loss_test_f = [] # Records test loss of longer addition sums
    acc_test_f = [] # Records test accuracy per position of longer addition sums
    acc_full_test_f = [] # Records test accuracy of longer addition sums

    project = f'{args.n_digit}-digit additon + {args.n_digit_gen}-digit test'

    run = wandb.init(project=project, 
                     config={"lr": lr, "split":split, "layers": n_layer, 
                             "weight decay": weight_decay, "d_ff": d_ff, "d_model": d_model,
                             "Attention heads": h_a, "Priming examples": priming_examples})

    for j in range(epochs):
        print(f'--- training epoch {j} started ---')

        model.train()
        
        # Training on training set (which includes priming examples)
        l, a, af, _   = run_epoch( dataloader_train, model, smoother, optimizer, lr_scheduler, rank, status = "train" )

        loss_train.append(l.mean(0).to('cpu').item())
        acc_train.append(a.to('cpu')) # No mean here if you have no DDP
        acc_full_train.append(af.mean(0).to('cpu'))

        model.eval()
        with torch.no_grad():
            # Evaluation on test set for n_digit sums
            lp, ap, apf, sq_w = run_epoch( dataloader_test, model, smootherEval, optimizer, lr_scheduler, rank, status = "eval" )
            
            sq_weights.append(sq_w.mean(0).data.to('cpu'))
            loss_test.append(lp.mean(0).to('cpu').item())
            acc_test.append(ap.to('cpu')) # No mean here if you have no DDP
            acc_full_test.append(apf.mean(0).to('cpu'))

            # Evaluation on test set for n_digit_gen sums
            lp_long, ap_long, apf_long, _ = run_epoch( dataloader_test_long, model, smootherEval, optimizer, lr_scheduler, rank, status = "eval" )

            loss_test_f.append(lp_long.mean(0).to('cpu').item())
            acc_test_f.append(ap_long.to('cpu')) # No mean here if you have no DDP
            acc_full_test_f.append(apf_long.mean(0).to('cpu'))

        wandb.log({"train loss": l.mean(0).item(),
                    "train acc": af.mean(0),
                    "test loss": lp.mean(0).item(),
                    "test acc": apf.mean(0),
                    "test acc long": apf_long.mean(0),
                    "test loss long": lp_long.mean(0).item(),
                    "Norm-squared weights": sq_w.mean(0)
                    })
        
        # Save model every 100 epochs
        if j % 100 == 0:

            dir = args.output_dir

            outputFile = dir + 'model_n{!s}_s{!s}_w{!s}_{!s}_{!s}'.format(n_layer, split, weight_decay, args.ID, j)

            torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, outputFile)
            
            DATA = [loss_train, loss_test, acc_train, acc_test, acc_full_train, acc_full_test, sq_weights, loss_test_f, acc_test_f, acc_full_test_f]

            outputFile = dir + 'DATA_n{!s}_s{!s}_w{!s}_{!s}_{!s}'.format(n_layer, split ,weight_decay, args.ID, j) + '.data'
            fw = open(outputFile, 'wb')
            pickle.dump(DATA, fw)

    wandb.finish()
    dir = args.output_dir

    outputFile = dir + 'model_n{!s}_s{!s}_w{!s}_{!s}'.format(n_layer, split, weight_decay, args.ID)

    torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
            }, outputFile)
    
    DATA = [loss_train, loss_test, acc_train, acc_test, acc_full_train, acc_full_test, sq_weights, loss_test_f, acc_test_f, acc_full_test_f]

    outputFile = dir + 'DATA_n{!s}_s{!s}_w{!s}_{!s}'.format(n_layer, split ,weight_decay, args.ID) + '.data'
    fw = open(outputFile, 'wb')
    pickle.dump(DATA, fw)
    fw.close()

if __name__ == "__main__":

    modelgeneration()

