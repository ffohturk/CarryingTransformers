import numpy as np
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

# Generates all possible ablations for a given number of layers and heads.

def generate_head_layer_ablations(n, heads):
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

def svd(Out_all, data_ff, target_ff):

    r = range(0, 10)

    emb_svd_full = []

    for i in range(len(Out_all)):
        emb_svd_full.append([torch.pca_lowrank(Out_all[i][:, j, :], center=False, niter=20) for j in r])

    pos_nc = np.argwhere(sum((data_ff[:, j] + data_ff[:, j+4] >= 10) for j in range(3)) < 1)[0]
    pos_c1 = np.argwhere((data_ff[:, 1] + data_ff[:, 1+4] >= 10) & (sum((data_ff[:, j] + data_ff[:, j+4] >= 10).float() for j in np.delete(np.arange(3), 1)) < 1))[0]
    pos_c2 = np.argwhere((data_ff[:, 2] + data_ff[:, 2+4] >= 10) & (data_ff[:, 1] + data_ff[:, 5] < 9))[0]
    pos_2c = np.argwhere(sum((data_ff[:, j] + data_ff[:, j+4] >= 10) for j in range(3)) == 2)[0]
    pos_2cp = np.argwhere((data_ff[:, 1] + data_ff[:, 5] == 9) & (data_ff[:, 2] + data_ff[:, 6] >= 10))[0]
    # pos_2c = np.concatenate((pos_2c, pos_2cp), 1)
    
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

    return emb_svd_full, positions, digit_ans_pos, digit_naive_ans_pos

def PCA_plot(n, path, svd_full, positions, digit_ans_pos, digit_naive_ans_pos):

    for i in range(n):
        path = f'PCA_layer{i}'
        if not os.path.exists(path):
            os.mkdir(path)

    ##### Layer 0,...,n-2 ########

    for layer in range(n-1):

        for a in range(1):
            for b in range(a+1, a+2):

                fig, ax = plt.subplots(2, 3, figsize=(12, 8))
                fig.tight_layout(h_pad = 2)

                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                            '#bcbd22', '#17becf']

                for l in range(2):
                    emb_prin = svd_full[l + 2*layer]
                    
                    for k in range(4, 7):

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


                        ax[l, k-4].plot(x_nc, y_nc, marker='o', markersize=3, linestyle='', color=colors[0], alpha = 0.3, rasterized=True)
                        ax[l, k-4].plot(x_c1, y_c1, marker='o', markersize=3, linestyle='', color=colors[1], alpha = 0.3, rasterized=True)
                        ax[l, k-4].plot(x_c2, y_c2, marker='o', markersize=3, linestyle='', color=colors[2], alpha = 0.3, rasterized=True)
                        ax[l, k-4].plot(x_2c, y_2c, marker='o', markersize=3, linestyle='', color=colors[3], alpha = 0.3, rasterized=True)
                        ax[l, k-4].plot(x_2cp, y_2cp, marker='o', markersize=3, linestyle='', color=colors[4], alpha = 0.3, rasterized=True)
                        
                lg = fig.legend(['$\\texttt{NC}$', '$\\texttt{C@1}$', '$\\texttt{C@2}$', '$\\texttt{C all}$', '$\\texttt{C all con.}$'], bbox_to_anchor=(1.17, 0.4), loc='lower right', borderaxespad=0.)
                for handle in lg.legend_handles:
                    handle.set_markersize(10)
                    handle.set_alpha(1)

                # filename = 'PCA_layer{!s}/PCA_layer{!s}_n{!s}_s{!s}_w{!s}_{!s}_{!s}.pdf'.format(layer, layer, n, s, w, a, b)
                # filename = directory + filename
                # plt.savefig(filename, format='pdf', dpi=150, bbox_extra_artists=(lg,), bbox_inches='tight')

                fig, ax = plt.subplots(2, 3, figsize=(12, 8))
                fig.tight_layout(h_pad = 2)

                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                            '#bcbd22', '#17becf']

                for l in range(2):
                    emb_prin = svd_full[l + 2*layer]

                    for k in range(4, 7):
                        
                        sk = k - 4

                        for m in range(10):
                            
                            x = emb_prin[k][0][digit_naive_ans_pos[sk][m], a] * emb_prin[k][1][a]
                            y = emb_prin[k][0][digit_naive_ans_pos[sk][m], b] * emb_prin[k][1][b]
                            
                            ax[l, sk].plot(x, y, marker='o', markersize=3, linestyle='', color=colors[m], alpha = 0.3, rasterized=True)


                lg = fig.legend(['$0$', '$1$', '$2$', '$3$', '$4$', '$5$', '$6$', '$7$', '$8$', '$9$'], bbox_to_anchor=(1.04, 0.30), loc='lower right', borderaxespad=0.)
                for handle in lg.legend_handles:
                    handle.set_markersize(10)
                    handle.set_alpha(1)

                # filename = 'PCA_layer{!s}/PCA_layer{!s}_digits_n{!s}_s{!s}_w{!s}_{!s}_{!s}.pdf'.format(layer, layer, n, s, w, a, b)
                # filename = directory + filename
                # plt.savefig(filename, format='pdf', dpi=150, bbox_extra_artists=(lg,), bbox_inches='tight')

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
                    
                    k = k + 6
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
                    
                    k = k - 6
                    
                    ax[l, k].plot(x_nc, y_nc, marker='o', markersize=3, linestyle='', color=colors[0], alpha = 0.3, rasterized=True)
                    ax[l, k].plot(x_c1, y_c1, marker='o', markersize=3, linestyle='', color=colors[1], alpha = 0.3, rasterized=True)
                    ax[l, k].plot(x_c2, y_c2, marker='o', markersize=3, linestyle='', color=colors[2], alpha = 0.3, rasterized=True)
                    ax[l, k].plot(x_2c, y_2c, marker='o', markersize=3, linestyle='', color=colors[3], alpha = 0.3, rasterized=True)
                    ax[l, k].plot(x_2cp, y_2cp, marker='o', markersize=3, linestyle='', color=colors[4], alpha = 0.3, rasterized=True)
                    
            lg = fig.legend(['$\\texttt{NC}$', '$\\texttt{C@1}$', '$\\texttt{C@2}$', '$\\texttt{C all}$', '$\\texttt{C all con.}$'], bbox_to_anchor=(1.07, 0.4), loc='lower right', borderaxespad=0.)
            for handle in lg.legend_handles:
                handle.set_markersize(10)
                handle.set_alpha(1)

            # filename = 'PCA_layer{!s}/PCA_layer{!s}_n{!s}_s{!s}_w{!s}_{!s}_{!s}.pdf'.format(n-1, n-1, n, s, w, a, b)
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
                        k = k + 6
                        x = emb_prin[k][0][digit_ans_pos[k-6][m], a] * emb_prin[k][1][a]
                        y = emb_prin[k][0][digit_ans_pos[k-6][m], b] * emb_prin[k][1][b]
                        k = k - 6
                        ax[l, k].plot(x, y, marker='o', markersize=3, linestyle='', color=colors[m], alpha = 0.3, rasterized=True)

            lg = fig.legend(['$0$', '$1$', '$2$', '$3$', '$4$', '$5$', '$6$', '$7$', '$8$', '$9$'], bbox_to_anchor=(1.04, 0.30), loc='lower right', borderaxespad=0.)
            for handle in lg.legend_handles:
                handle.set_markersize(10)
                handle.set_alpha(1)

            # filename = 'PCA_layer{!s}/PCA_layer{!s}_digits_n{!s}_s{!s}_w{!s}_{!s}_{!s}.pdf'.format(n-1, n-1, n, s, w, a, b)
            # filename = directory + filename
            # plt.savefig(filename, format='pdf', dpi=150, bbox_extra_artists=(lg,), bbox_inches='tight')

def plot_attention_patterns(n, model, positions):

    colors = ['Blues', 'Oranges', 'Greens', 'Reds', 'Purples']

    fig, ax = plt.subplots(2*n, 5, figsize=(12,7.5))
    fig.tight_layout(h_pad=-1, w_pad=1)

    rows = []
    for i in range(len(positions)):
        
        for l in range(n):

            att0 = model.decoder.layers[l].attn.attn[positions[i], :, :, :].clone().detach().mean(0)
            if l == n-1: 
                ax[0 + 2*l, i].imshow(att0[0, :, :], cmap=colors[i])
                ax[1 + 2*l, i].imshow(att0[1, :, :], cmap=colors[i])
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
        if j < 2*n:
            ax[j, 0].set_yticks(range(10), ['$*$']*3 + ['$+$'] + ['$*$']*3 + ['$=$'] * 3)
        else:
            ax[j, 0].set_yticks(range(3), ['$=$', '$=$', '$=$'])  
    for i in range(len(positions)):
        ax[-1, i].set_xticks(range(10), ['$*$']*3 + ['$+$'] + ['$*$']*3 + ['$=$'] * 3) 

    for ax, row in zip(ax[:,0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')