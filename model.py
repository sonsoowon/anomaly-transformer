import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class SingleAnomalyAttention(nn.Module):
    def __init__(self, N, dim, use_cuda=True):
        super().__init__()

        self.device = torch.device('cpu')
        if torch.cuda.is_available() and use_cuda:
            self.device = torch.device('cuda')
        torch.set_default_device(self.device)

        self.dim = dim
        self.N = N

        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.Ws = nn.Linear(dim, 1, bias=False)

        self.Q = self.K = self.V = torch.zeros((N, dim))
        self.sigma = torch.zeros((N, 1))

        self.P = torch.zeros((N, N))
        self.S = torch.zeros((N, N))

    def forward(self, x):
        self._initialize(x)
        self.P = self.prior_association()
        self.S = self.series_association()
        Z = self.reconstruction()

        return Z

    def _initialize(self, x):
        self.Q = self.Wq(x)
        self.K = self.Wk(x)
        self.V = self.Wv(x)
        self.sigma = self.Ws(x)

    @staticmethod
    def gaussian_kernel(mean, sigma):
        normalize = 1 / (math.sqrt(2 * torch.pi) * sigma)
        return normalize * torch.exp(-0.5 * (mean / sigma).pow(2))

    def prior_association(self):
        distance = torch.from_numpy(np.abs(np.indices((self.N, self.N))[0] - np.indices((self.N, self.N))[1]))
        gaussian = self.gaussian_kernel(distance.float().to(self.device), self.sigma)
        div = torch.sum(gaussian, dim=-1).unsqueeze(-1)
        gaussian = gaussian / div

        return gaussian

    def series_association(self):
        return F.softmax((self.Q @ self.K.transpose(1, 2)) / math.sqrt(self.dim), dim=0)

    def reconstruction(self):
        return self.S @ self.V


class MultiHeadAnomalyAttention(nn.Module):
    def __init__(self, N, d_model, head_num, use_cuda=True):
        super().__init__()

        self.device = torch.device('cpu')
        if torch.cuda.is_available() and use_cuda:
            self.device = torch.device('cuda')
        torch.set_default_device(self.device)

        self.h = head_num
        self.single_dim = int(d_model / head_num)
        self.N = N

        self.P = torch.zeros((1, N, N))
        self.S = torch.zeros((1, N, N))

        self.attn_segments = nn.ModuleList(
            [SingleAnomalyAttention(self.N, self.single_dim, use_cuda) for _ in range(head_num)]
        )

    def forward(self, x):
        x_segments = torch.split(x, self.single_dim, dim=-1)
        reconstructions = []

        for x, single_attn in zip(x_segments, self.attn_segments):
            reconstructions.append(single_attn(x))

            self.P = single_attn.P / self.h + self.P
            self.S = single_attn.S / self.h + self.S

        return torch.cat(reconstructions, dim=-1)


class AnomalyTransformerBlock(nn.Module):
    def __init__(self, N, d_model, head_num, d_ff=None, dropout=0.1, activation="relu", use_cuda=True):
        super().__init__()
        self.N, self.d_model = N, d_model
        d_ff = d_ff or 4 * d_model

        self.attention = MultiHeadAnomalyAttention(self.N, self.d_model, head_num, use_cuda)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        new_x = self.attention(x)
        x_before_ff = self.norm1(x + self.dropout(new_x))

        x = self.ff1(x_before_ff)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.ff2(x)
        x_after_ff = self.dropout(x)

        return self.norm2(x_before_ff + x_after_ff)


class AnomalyTransformer(nn.Module):
    def __init__(self, N, d_model, head_num, layers, d_ff=None, dropout=0.0, activation='gelu', use_cuda=True):
        super().__init__()
        self.N, self.d_model = N, d_model

        self.blocks = nn.ModuleList(
            [AnomalyTransformerBlock(self.N, self.d_model, head_num, d_ff, dropout, activation, use_cuda) for _ in range(layers)]
        )

    def forward(self, x):
        prior_list, series_list = [], []
        for idx, block in enumerate(self.blocks):
            x = block(x)
            prior_list.append(block.attention.P)
            series_list.append(block.attention.S)

        return x, prior_list, series_list

