import numpy as np
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

def generate_head_layer_ablations(n: int, heads: int):

    """
    Generates all possible ablations for a given number of layers and heads

    Input is n: Int and heads: Int

    Output is a binary tensor of size (2**(n*heads), layers, heads)

    """

    l = n*heads
    Q = [[1]*l]

    for i in range(l):
        k = 0
        while Q[k][l-1-i] != 0:
            Q_new = Q[k].copy()
            Q_new[l-1-i] -= 1
            Q.append(Q_new)
            k += 1
    Q.sort(reverse=True,key=lambda i: sum(i))
    return torch.tensor(Q).view(-1,n,heads)

def FixedExamples(fixed_int: int, stoi: dict):

    """
    
    Generates a number of examples with a fixed outcome for three digit addition.

    Output is a tensor of size (num_examples, seq_len = 10)

    """
    
    ex_fixed = []

    for i in range(fixed_int+1):
        for j in range(fixed_int+1):
            if i + j == fixed_int and i >= j:
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
                ex_fixed.append(lsum)

    return torch.tensor([[stoi[ex_fixed[i][j]] for j in range(len(ex_fixed[i]))] for i in range(len(ex_fixed))])

def svd(Out_all: list, data_ff: list, target_ff: list):

    """
    
    Generates:

    1)  PCA of the hidden states after attention and MLP of each layer. 
        We use torch.pca_lowrank to calculate it quickly. 
        
        Output is svd_full. List of torch tensors. 

    2)  index (position) of examples in dataset data_ff that belong to one of the five classes relevant
        for three digit addition.

        Output is positions. List position of torch tensors.

    3)  Generates the indices of examples with a particular outcome at a particular positon. 
        We generate this with and without taking carrying over into accout. 

        Outputs are digit_ans_pos and digit_naive_ans_pos, respectively. List of List of torch tensors. 
    """

    r = range(0, 10)

    svd_full = []

    for i in range(len(Out_all)):
        svd_full.append([torch.pca_lowrank(Out_all[i][:, j, :], center=False, niter=20) for j in r])

    pos_nc = np.argwhere(sum((data_ff[:, j] + data_ff[:, j+4] >= 10) for j in range(3)) < 1)[0]
    pos_c1 = np.argwhere((data_ff[:, 1] + data_ff[:, 1+4] >= 10) & (sum((data_ff[:, j] + data_ff[:, j+4] >= 10).float() for j in np.delete(np.arange(3), 1)) < 1))[0]
    pos_c2 = np.argwhere((data_ff[:, 2] + data_ff[:, 2+4] >= 10) & (data_ff[:, 1] + data_ff[:, 5] < 9))[0]
    pos_2c = np.argwhere(sum((data_ff[:, j] + data_ff[:, j+4] >= 10) for j in range(3)) == 2)[0]
    pos_2cp = np.argwhere((data_ff[:, 1] + data_ff[:, 5] == 9) & (data_ff[:, 2] + data_ff[:, 6] >= 10))[0]
    
    positions = [pos_nc, pos_c1, pos_c2, pos_2c, pos_2cp]

    digit_ans_pos = []
    for k in range(3):
        ans_pp = []
        for i in range(10):
            ans_p = []
            for j in range(len(target_ff)):
                if target_ff[j, k] == i:
                    ans_p.append(j)
            ans_pp.append(torch.tensor(ans_p))
        digit_ans_pos.append(ans_pp)

    digit_naive_ans_pos = []
    for k in range(3):
        ans_pp = []
        for i in range(10):
            ans_p = []
            for j in range(len(data_ff)):
                if data_ff[j, k] + data_ff[j, k + 4] == i or data_ff[j, k] + data_ff[j, k + 4] == (i + 10):
                    ans_p.append(j)
            ans_pp.append(torch.tensor(ans_p))
        digit_naive_ans_pos.append(ans_pp)

    return svd_full, positions, digit_ans_pos, digit_naive_ans_pos

def PCA_plot(n: int, svd_full: list, positions: list, digit_ans_pos: list, digit_naive_ans_pos: list):

    """
    
    Generates plots of PCA data. For layer 0...n-2 w plot the SVD for the first seven positions (excluding '+' token) 
    and for the last layer we plot the last three. This is for three digit addition.
    
    """

    ##### Layer 0,...,n-2 ########

    for layer in range(n-1):

        for a in range(1):
            for b in range(a+1, a+2):

                fig, ax = plt.subplots(2, 6, figsize=(25, 8))
                fig.tight_layout(h_pad = 2)

                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                            '#bcbd22', '#17becf']

                for l in range(2):
                    emb_prin = svd_full[l + 2*layer]
                    
                    for k in np.delete(range(7), 3):
                        sk = k
                        x_nc = emb_prin[k][0][positions[0], a] * emb_prin[k][1][a]
                        y_nc = emb_prin[k][0][positions[0], b] * emb_prin[k][1][b]

                        x_c1 = emb_prin[k][0][positions[1], a] * emb_prin[k][1][a]
                        y_c1 = emb_prin[k][0][positions[1], b] * emb_prin[k][1][b]

                        x_c2 = emb_prin[k][0][positions[2], a] * emb_prin[k][1][a]
                        y_c2 = emb_prin[k][0][positions[2], b] * emb_prin[k][1][b]

                        x_2c = emb_prin[k][0][positions[3], a] * emb_prin[k][1][a]
                        y_2c = emb_prin[k][0][positions[3], b] * emb_prin[k][1][b]

                        x_2cp = emb_prin[k][0][positions[4], a] * emb_prin[k][1][a]
                        y_2cp = emb_prin[k][0][positions[4], b] * emb_prin[k][1][b]

                        if k > 3:
                            k -= 1

                        ax[l, k].plot(x_nc, y_nc, marker='o', markersize=3, linestyle='', color=colors[0], alpha = 0.3, rasterized=True)
                        ax[l, k].plot(x_c1, y_c1, marker='o', markersize=3, linestyle='', color=colors[1], alpha = 0.3, rasterized=True)
                        ax[l, k].plot(x_c2, y_c2, marker='o', markersize=3, linestyle='', color=colors[2], alpha = 0.3, rasterized=True)
                        ax[l, k].plot(x_2c, y_2c, marker='o', markersize=3, linestyle='', color=colors[3], alpha = 0.3, rasterized=True)
                        ax[l, k].plot(x_2cp, y_2cp, marker='o', markersize=3, linestyle='', color=colors[4], alpha = 0.3, rasterized=True)
                        if l == 0: 
                            ax[l, k].set_title('After Attention, Position {!s}'.format(sk))
                        else:
                            ax[l, k].set_title('After MLP, Position {!s}'.format(sk))
                lg = fig.legend(['$\\texttt{NC}$', '$\\texttt{C@1}$', '$\\texttt{C@2}$', '$\\texttt{C all}$', '$\\texttt{C all con.}$'], bbox_to_anchor=(1.17, 0.4), loc='lower right', borderaxespad=0.)
                for handle in lg.legend_handles:
                    handle.set_markersize(10)
                    handle.set_alpha(1)

                fig, ax = plt.subplots(2, 6, figsize=(25, 8))
                fig.tight_layout(h_pad = 2)

                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                            '#bcbd22', '#17becf']

                for l in range(2):
                    emb_prin = svd_full[l + 2*layer]
                    qq = False
                    for k in np.delete(range(7), 3):
                        
                        ss = k
                        if k < 3:
                            sk = k
                            ll = k
                            qq = True
                        elif k > 3:
                            sk = k - 4
                            ll = k - 1
                            qq = True
                        else:
                            qq = False

                        for m in range(10):
                            
                            if qq:
                                x = emb_prin[k][0][digit_naive_ans_pos[sk][m], a] * emb_prin[k][1][a]
                                y = emb_prin[k][0][digit_naive_ans_pos[sk][m], b] * emb_prin[k][1][b]
                                
                                ax[l, ll].plot(x, y, marker='o', markersize=3, linestyle='', color=colors[m], alpha = 0.3, rasterized=True)

                        if l == 0: 
                            ax[l, ll].set_title('After Attention, Position {!s}'.format(ss))
                        else:
                            ax[l, ll].set_title('After MLP, Position {!s}'.format(ss))

                lg = fig.legend(['$0$', '$1$', '$2$', '$3$', '$4$', '$5$', '$6$', '$7$', '$8$', '$9$'], bbox_to_anchor=(1.04, 0.30), loc='lower right', borderaxespad=0.)
                for handle in lg.legend_handles:
                    handle.set_markersize(10)
                    handle.set_alpha(1)

    ######## Final layer ##########

    for a in range(1):
        for b in range(a+1, a+2):

            fig, ax = plt.subplots(2, 3, figsize=(25, 8))
            fig.tight_layout(h_pad = 2)

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                        '#bcbd22', '#17becf']

            for l in range(2):
                emb_prin = svd_full[l+2*(n-1)]

                for k in range(3):
                    
                    k = k + 7
                    x_nc = emb_prin[k][0][positions[0], a] * emb_prin[k][1][a]
                    y_nc = emb_prin[k][0][positions[0], b] * emb_prin[k][1][b]

                    x_c1 = emb_prin[k][0][positions[1], a] * emb_prin[k][1][a]
                    y_c1 = emb_prin[k][0][positions[1], b] * emb_prin[k][1][b]

                    x_c2 = emb_prin[k][0][positions[2], a] * emb_prin[k][1][a]
                    y_c2 = emb_prin[k][0][positions[2], b] * emb_prin[k][1][b]

                    x_2c = emb_prin[k][0][positions[3], a] * emb_prin[k][1][a]
                    y_2c = emb_prin[k][0][positions[3], b] * emb_prin[k][1][b]

                    x_2cp = emb_prin[k][0][positions[4], a] * emb_prin[k][1][a]
                    y_2cp = emb_prin[k][0][positions[4], b] * emb_prin[k][1][b]
                    
                    k = k - 7
                    
                    ax[l, k].plot(x_nc, y_nc, marker='o', markersize=3, linestyle='', color=colors[0], alpha = 0.3, rasterized=True)
                    ax[l, k].plot(x_c1, y_c1, marker='o', markersize=3, linestyle='', color=colors[1], alpha = 0.3, rasterized=True)
                    ax[l, k].plot(x_c2, y_c2, marker='o', markersize=3, linestyle='', color=colors[2], alpha = 0.3, rasterized=True)
                    ax[l, k].plot(x_2c, y_2c, marker='o', markersize=3, linestyle='', color=colors[3], alpha = 0.3, rasterized=True)
                    ax[l, k].plot(x_2cp, y_2cp, marker='o', markersize=3, linestyle='', color=colors[4], alpha = 0.3, rasterized=True)
                    if l == 0: 
                        ax[l, k].set_title('After Attention, Position {!s}'.format(k))
                    else:
                        ax[l, k].set_title('After MLP, Position {!s}'.format(k))

            lg = fig.legend(['$\\texttt{NC}$', '$\\texttt{C@1}$', '$\\texttt{C@2}$', '$\\texttt{C all}$', '$\\texttt{C all con.}$'], bbox_to_anchor=(1.17, 0.4), loc='lower right', borderaxespad=0.)
            for handle in lg.legend_handles:
                handle.set_markersize(10)
                handle.set_alpha(1)

            fig, ax = plt.subplots(2, 3, figsize=(25, 8))
            fig.tight_layout(h_pad = 2)

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                        '#bcbd22', '#17becf']
            for l in range(2):
                emb_prin = svd_full[l+2*(n-1)]

                for k in range(3):

                    for m in range(10):
                        k = k + 7
                        x = emb_prin[k][0][digit_ans_pos[k-7][m], a] * emb_prin[k][1][a]
                        y = emb_prin[k][0][digit_ans_pos[k-7][m], b] * emb_prin[k][1][b]
                        k = k - 7
                        ax[l, k].plot(x, y, marker='o', markersize=3, linestyle='', color=colors[m], alpha = 0.3, rasterized=True)

                    if l == 0: 
                        ax[l, k].set_title('After Attention, Position {!s}'.format(k))
                    else:
                        ax[l, k].set_title('After MLP, Position {!s}'.format(k))

            lg = fig.legend(['$0$', '$1$', '$2$', '$3$', '$4$', '$5$', '$6$', '$7$', '$8$', '$9$'], bbox_to_anchor=(1.04, 0.30), loc='lower right', borderaxespad=0.)
            for handle in lg.legend_handles:
                handle.set_markersize(10)
                handle.set_alpha(1)

def plot_attention_patterns(n: int, model: nn.Module, positions: list):

    """
    
    Generates attention patterns for a given model for each layer and head and for each task in positions.
    
    """

    colors = ['Blues', 'Oranges', 'Greens', 'Reds', 'Purples']

    fig, ax = plt.subplots(2*n, 5, figsize=(12,7.5))
    fig.tight_layout(h_pad=-1, w_pad=1)

    rows = []
    for i in range(len(positions)):
        
        for l in range(n):

            att0 = model.decoder.layers[l].attn.attn[positions[i], :, :, :].clone().detach().mean(0)
            if l == n-1: 
                ax[0 + 2*l, i].imshow(att0[0, -3:, :], cmap=colors[i])
                ax[1 + 2*l, i].imshow(att0[1, -3:, :], cmap=colors[i])
            else:
                ax[0 + 2*l, i].imshow(att0[0, :, :], cmap=colors[i])
                ax[1 + 2*l, i].imshow(att0[1, :, :], cmap=colors[i])
            if l == 0:
                rows.extend(['$\\rm Head\;0\\hspace{-5pt}:\\hspace{-5pt}0$', '$\\rm Head\;0\\hspace{-5pt}:\\hspace{-5pt}1$'])
            elif l == 1:
                rows.extend(['$\\rm Head\;1\\hspace{-5pt}:\\hspace{-5pt}0$', '$\\rm Head\;1\\hspace{-5pt}:\\hspace{-5pt}1$'])

    ax[0, 0].set_title('$\\texttt{NC}$')
    ax[0, 1].set_title('$\\texttt{C@1}$')
    ax[0, 2].set_title('$\\texttt{C@2}$')
    ax[0, 3].set_title('$\\texttt{C all}$')    
    ax[0, 4].set_title('$\\texttt{C all con.}$')

    for i in range(len(positions)):
        for j in range(2*n):  
            ax[j, i].set_xticks([])
            ax[j, i].set_yticks([])
    for j in range(2*n):
        if j < 2*n-2:
            ax[j, 0].set_yticks(range(10), ['$*$']*3 + ['$+$'] + ['$*$']*3 + ['$=$'] * 3)
        else:
            ax[j, 0].set_yticks(range(3), ['$=$', '$=$', '$=$'])  
    for i in range(len(positions)):
        ax[-1, i].set_xticks(range(10), ['$*$']*3 + ['$+$'] + ['$*$']*3 + ['$=$'] * 3) 

    for ax, row in zip(ax[:,0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')

def Relative_Angle_AfterEmbedding_examples(model: nn.Module, examples: torch.tensor, All=True):

    """
    
    Generates and plots the absolute value of the cosine similarty between hiddens states after 
    embedding for a fixed set of examples. 

    """


    mask = None
    _ = model(examples, mask)

    Out_emb = model.embed.out_e[:, :, :].detach().clone()

    dot = torch.einsum('ilj, klj -> ikl', Out_emb, Out_emb)
    norm0 = torch.einsum('ilj, ilj -> il', Out_emb, Out_emb)**0.5
    norm = torch.einsum('il, kl -> ikl', norm0, norm0)

    angle_emb = dot / norm

    if All:
        fig, ax = plt.subplots(1, 7, figsize = (25, 3))
        for i in range(7):
            im = ax[i].imshow(np.abs(angle_emb[:, :, i]), cmap='Blues', vmin=0, vmax=1)
            ax[i].set_xlabel('$b$')
            ax[i].set_ylabel('$a$')
            ax[i].set_xticks([])
            ax[i].set_yticks([])
    else: 
        fig, ax = plt.subplots(1, 3, figsize = (12, 3))
        for i in range(3):
            im = ax[i].imshow(np.abs(angle_emb[:, :, i+7]), cmap='Blues', vmin=0, vmax=1)
            ax[i].set_xlabel('$b$')
            ax[i].set_ylabel('$a$')
            ax[i].set_xticks([])
            ax[i].set_yticks([])

    return angle_emb

def Relative_Angle_AfterAttention(model: nn.Module, examples: torch.tensor, layer: int, All=True):

    """
    
    Generates and plots the absolute value of the cosine similarty between hiddens states after 
    attention of a specified layer for a fixed set of example. 

    """

    mask = None
    out = model(examples, None)

    Out_attn = model.decoder.layers[layer].out_a[:, :, :].detach().clone()

    dot = torch.einsum('ilj, klj -> ikl', Out_attn, Out_attn)
    norm0 = torch.einsum('ilj, ilj -> il', Out_attn, Out_attn)**0.5
    norm = torch.einsum('il, kl -> ikl', norm0, norm0)

    angle_att = dot / (norm + 1e-9)

    if All:
        fig, ax = plt.subplots(1, 10, figsize = (25, 3))
        for i in range(10):
            im = ax[i].imshow(np.abs(angle_att[:, :, i]), cmap='Blues', vmin=0, vmax=1)
            ax[i].set_xlabel('$b$')
            ax[i].set_ylabel('$a$')
            ax[i].set_xticks([])
            ax[i].set_yticks([])
    else: 
        fig, ax = plt.subplots(1, 3, figsize = (12, 3))
        for i in range(3):
            im = ax[i].imshow(np.abs(angle_att[:, :, i+7]), cmap='Blues', vmin=0, vmax=1)
            
            ax[i].set_xlabel('$b$')
            ax[i].set_ylabel('$a$')
            ax[i].set_xticks([])
            ax[i].set_yticks([])
    fig.colorbar(im, ax=ax.ravel().tolist())
   
    return angle_att

def Relative_Angle_AfterFFN(model: nn.Module, examples: torch.tensor, layer: int, All=True):

    """
    
    Generates and plots the absolute value of the cosine similarty between hiddens states after 
    MLP of a specified layer for a fixed set of example. 

    """

    mask = None
    _ = model(examples, mask)

    Out_ffn = model.decoder.layers[layer].out[:, :, :].detach().clone()

    dot = torch.einsum('ilj, klj -> ikl', Out_ffn, Out_ffn)
    norm0 = torch.einsum('ilj, ilj -> il', Out_ffn, Out_ffn)**0.5
    norm = torch.einsum('il, kl -> ikl', norm0, norm0)

    angle_ffn = dot / norm

    if All:
        fig, ax = plt.subplots(1, 10, figsize = (25, 3))
        for i in range(10):
            im = ax[i].imshow(np.abs(angle_ffn[:, :, i]), cmap='Blues', vmin=0, vmax=1)
            ax[i].set_xlabel('$b$')
            ax[i].set_ylabel('$a$')
            ax[i].set_xticks([])
            ax[i].set_yticks([])
    else: 
        fig, ax = plt.subplots(1, 3, figsize = (12, 3))
        for i in range(3):
            im = ax[i].imshow(np.abs(angle_ffn[:, :, i+7]), cmap='Blues', vmin=0, vmax=1)
            ax[i].set_xlabel('$b$')
            ax[i].set_ylabel('$a$')
            ax[i].set_xticks([])
            ax[i].set_yticks([])

    fig.colorbar(im, ax=ax.ravel().tolist())

    return angle_ffn

def PCA_plot_different(n: int, svd_full: list, positions: list, digit_ans_pos: list, carry_or_non_carry_at_pos_8 = True):

    for a in range(2):
        for b in range(a+1, 3):

            fig, ax = plt.subplots(2, 3, figsize=(25, 8))
            fig.tight_layout(h_pad = 2)

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                        '#bcbd22', '#17becf']

            for l in range(2):
                emb_prin = svd_full[l+2*(n-1)]

                for k in range(3):
                    
                    k = k + 7
                    if not carry_or_non_carry_at_pos_8:
                        x_nc = emb_prin[k][0][positions[0], a] * emb_prin[k][1][a]
                        y_nc = emb_prin[k][0][positions[0], b] * emb_prin[k][1][b]

                        x_c1 = emb_prin[k][0][positions[1], a] * emb_prin[k][1][a]
                        y_c1 = emb_prin[k][0][positions[1], b] * emb_prin[k][1][b]

                        k = k - 7
                    
                        ax[l, k].plot(x_nc, y_nc, marker='o', markersize=3, linestyle='', color=colors[0], alpha = 0.3, rasterized=True)
                        ax[l, k].plot(x_c1, y_c1, marker='o', markersize=3, linestyle='', color=colors[1], alpha = 0.3, rasterized=True)
                    else:
                        x_c2 = emb_prin[k][0][positions[2], a] * emb_prin[k][1][a]
                        y_c2 = emb_prin[k][0][positions[2], b] * emb_prin[k][1][b]

                        x_2c = emb_prin[k][0][positions[3], a] * emb_prin[k][1][a]
                        y_2c = emb_prin[k][0][positions[3], b] * emb_prin[k][1][b]

                        x_2cp = emb_prin[k][0][positions[4], a] * emb_prin[k][1][a]
                        y_2cp = emb_prin[k][0][positions[4], b] * emb_prin[k][1][b]
                        
                        k = k - 7

                        ax[l, k].plot(x_c2, y_c2, marker='o', markersize=3, linestyle='', color=colors[2], alpha = 0.3, rasterized=True)
                        ax[l, k].plot(x_2c, y_2c, marker='o', markersize=3, linestyle='', color=colors[3], alpha = 0.3, rasterized=True)
                        ax[l, k].plot(x_2cp, y_2cp, marker='o', markersize=3, linestyle='', color=colors[4], alpha = 0.3, rasterized=True)
                    # if l == 0: 
                    #     ax[l, k].set_title(f'$\\rm After\,Attention,\,Position\,{{a}}$'.format(a=k+7))
                    # else:
                    #     ax[l, k].set_title(f'$\\rm After\,MLP,\,Position\,{{a}}$'.format(a=k+7))
            if not carry_or_non_carry_at_pos_8:
                lg = fig.legend(['$\\texttt{NC}$', '$\\texttt{C@1}$'], bbox_to_anchor=(1.07, 0.4), loc='lower right', borderaxespad=0.)
            else: 
                lg = fig.legend(['$\\texttt{C@2}$', '$\\texttt{C all}$', '$\\texttt{C all con.}$'], bbox_to_anchor=(1.07, 0.4), loc='lower right', borderaxespad=0.) 
            for handle in lg.legend_handles:
                handle.set_markersize(10)
                handle.set_alpha(1)

            # filename = 'PCA_layer{!s}/PCA_layer{!s}_n{!s}_s{!s}_w{!s}_model{!s}_{!s}_{!s}_abl_decision_head.pdf'.format(n-1, n-1, n, s, w, p, a, b)
            # filename = directory + filename
            # plt.savefig(filename, format='pdf', dpi=150, bbox_extra_artists=(lg,), bbox_inches='tight')

            fig, ax = plt.subplots(2, 3, figsize=(25, 8))
            fig.tight_layout(h_pad = 2)

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                        '#bcbd22', '#17becf']
            for l in range(2):
                emb_prin = svd_full[l+2*(n-1)]

                for k in range(3):

                    for m in range(10):
                        k = k + 7
                        if not carry_or_non_carry_at_pos_8:
                            q0 = torch.LongTensor(list(set(positions[2].tolist()) & set(digit_ans_pos[k-7][m].tolist())))
                            q1 = torch.LongTensor(list(set(positions[3].tolist()) & set(digit_ans_pos[k-7][m].tolist())))
                            q2 = torch.LongTensor(list(set(positions[4].tolist()) & set(digit_ans_pos[k-7][m].tolist())))
                            q2 = torch.cat((q0, q1, q2), 0)

                            x = emb_prin[k][0][q2, a] * emb_prin[k][1][a]
                            y = emb_prin[k][0][q2, b] * emb_prin[k][1][b]

                            k = k - 7
                            ax[l, k].plot(x, y, marker='o', markersize=3, linestyle='', color=colors[m], alpha = 0.3, rasterized=True)

                        else:
                            q0 = torch.LongTensor(list(set(positions[0].tolist()) & set(digit_ans_pos[k-7][m].tolist())))
                            q1 = torch.LongTensor(list(set(positions[1].tolist()) & set(digit_ans_pos[k-7][m].tolist())))
                            q2 = torch.cat((q0, q1), 0)

                            x = emb_prin[k][0][q2, a] * emb_prin[k][1][a]
                            y = emb_prin[k][0][q2, b] * emb_prin[k][1][b]

                            k = k - 7
                            ax[l, k].plot(x, y, marker='o', markersize=3, linestyle='', color=colors[m], alpha = 0.3, rasterized=True)
                                

                    # if l == 0: 
                    #         ax[l, k].set_title(f'$\\rm After\,Attention,\,Position\,{{a}}$'.format(a=k+7))
                    # else:
                    #     ax[l, k].set_title(f'$\\rm After\,MLP,\,Position\,{{a}}$'.format(a=k+7))

            lg = fig.legend(['$0$', '$1$', '$2$', '$3$', '$4$', '$5$', '$6$', '$7$', '$8$', '$9$'], bbox_to_anchor=(1.04, 0.30), loc='lower right', borderaxespad=0.)
            for handle in lg.legend_handles:
                handle.set_markersize(10)
                handle.set_alpha(1)