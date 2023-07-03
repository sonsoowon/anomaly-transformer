import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class SingleAnomalyAttention(nn.Module):
    def __init__(self, N, dim):
        super().__init__()

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        torch.set_default_device(device)

        self.dim = dim
        self.N = N

        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.Ws = nn.Linear(dim, 1, bias=False)

        self.Q = self.K = self.V = self.sigma = torch.zeros((N, dim))

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
        distance = torch.from_numpy(
            np.abs(np.indices((self.N, self.N))[0] - np.indices((self.N, self.N))[1])
        )
        gaussian = self.gaussian_kernel(distance.float(), self.sigma)
        gaussian /= gaussian.sum(dim=-1).view(-1, 1)

        return gaussian

    def series_association(self):
        return F.softmax((self.Q @ self.K.T) / math.sqrt(self.dim), dim=0)

    def reconstruction(self):
        return self.S @ self.V


class MultiHeadAnomalyAttention(nn.Module):
    def __init__(self, N, d_model, head_num):
        super().__init__()

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        torch.set_default_device(device)

        self.h = head_num
        self.single_dim = int(d_model / head_num)
        self.N = N

        self.P = torch.zeros((N, N))
        self.S = torch.zeros((N, N))

        self.attn_segments = nn.ModuleList(
            [SingleAnomalyAttention(self.N, self.single_dim) for _ in range(head_num)]
        )

    def forward(self, x):
        x_segments = torch.split(x, self.single_dim, dim=-1)
        reconstructions = []

        for x, single_attn in zip(x_segments, self.attn_segments):
            reconstructions.append(single_attn(x))
            self.P += single_attn.P / self.h
            self.S += single_attn.S / self.h

        return torch.cat(torch.tensor(reconstructions), dim=-1)


class AnomalyTransformerBlock(nn.Module):
    def __init__(self, N, d_model, head_num):
        super().__init__()
        self.N, self.d_model = N, d_model

        self.attention = MultiHeadAnomalyAttention(self.N, self.d_model, head_num)
        self.ln1 = nn.LayerNorm(self.d_model)
        self.ff = nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.ReLU())
        self.ln2 = nn.LayerNorm(self.d_model)

    def forward(self, x):
        x_identity = x
        x = self.attention(x)
        z = self.ln1(x + x_identity)

        z_identity = z
        z = self.ff(z)
        z = self.ln2(z + z_identity)

        return z

class AnomalyTransformer(nn.Module):
    def __init__(self, N, d_model, head_num, layers, lambda_):
        super().__init__()
        self.N = N
        self.d_model = d_model

        self.blocks = nn.ModuleList(
            [AnomalyTransformerBlock(self.N, self.d_model, head_num) for _ in range(layers)]
        )
        self.output = None
        self.lambda_ = lambda_

        self.P_layers = []
        self.S_layers = []

    def forward(self, x):
        for idx, block in enumerate(self.blocks):
            x = block(x)
            self.P_layers.append(block.attention.P)
            self.S_layers.append(block.attention.S)

        self.output = x
        return x

    def layer_association_discrepancy(self, Pl, Sl):
        rowwise_kl = lambda row: F.kl_div(Pl[row, :], Sl[row, :]) + F.kl_div(Sl[row, :], Pl[row, :])
        return torch.tensor([rowwise_kl(row) for row in range(Pl.shape[0])])

    def association_discrepancy(self, P_list, S_list):
        all_ass_dis = torch.tensor([self.layer_association_discrepancy(Pl, Sl)
                                    for Pl, Sl, in zip(P_list, S_list)])
        return torch.mean(all_ass_dis, dim=0).unsqueeze(1)

    def loss_function(self, x, x_reconstructed, P_list, S_list, lambda_):
        frob_norm = torch.linalg.matrix_norm(x_reconstructed - x, ord="fro")
        ass_dis = torch.linalg.norm(self.association_discrepancy(P_list, S_list), ord=1)
        return frob_norm - lambda_ * ass_dis

    def min_loss(self, x):
        P_list = self.P_layers
        S_list = [S.detach() for S in self.S_layers]
        lambda_ = -self.lambda_
        return self.loss_function(self.output, P_list, S_list, lambda_, x)

    def max_loss(self, x):
        P_list = [P.detach() for P in self.P_layers]
        S_list = self.S_layers
        lambda_ = self.lambda_
        return self.loss_function(self.output, P_list, S_list, lambda_, x)

    def anomaly_score(self, x):
        ass_dis_ratio = F.softmax(-self.association_discrepancy(self.P_layers, self.S_layers), dim=0)
        assert ass_dis_ratio.shape[0] == self.N

        reconstruction_error = torch.tensor([torch.linalg.norm(x[i, :] - self.output[i, :], ord=2) for i in range(self.N)])
        assert reconstruction_error.shape[0] == self.N

        score = torch.mul(ass_dis_ratio, reconstruction_error)
        return score

