import numpy as np
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F

def dataset_generator(P_f):

    data_f = []
    target_f = []

    for i in range(P_f):
        for j in range(P_f):
            if P_f > i + j:
                li = list(f'{i}')
                lj = list(f'{j}')
                lij = list(f'{i+j}')
                if len(li) < 3:
                    li = ['0'] * (3 - len(li)) + li
                if len(lj) < 3:
                    lj = ['0'] * (3 - len(lj)) + lj
                if len(lij) < 3:
                    lij = ['0'] * (3 - len(lij)) + lij

                lsum = li + ['+'] + lj + ['='] * 3
                data_f.append(lsum)
                target_f.append(lij)

    stoi = {'0': 0, '1': 1, '2': 2,'3': 3,'4': 4,'5': 5,'6': 6,'7': 7,'8': 8,'9': 9,'+': 10,'=': 11}

    data_f = [[stoi[data_f[i][j]] for j in range(len(data_f[i]))] for i in range(len(data_f))]
    target_f = [[stoi[target_f[i][j]] for j in range(len(target_f[i]))] for i in range(len(target_f))]

    data_f = torch.LongTensor(data_f)
    target_f = torch.LongTensor(target_f)

    return data_f, target_f, stoi