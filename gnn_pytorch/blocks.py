import logging

import torch
import torch.nn as nn
from torch.nn import functional as F


logger = logging.getLogger(__name__)


class ReadOut(nn.Module):
    """Atomwise readout that sums over all atoms(nodes)"""
    def __init__(self, gcn_dim, readout_dim):
        super().__init__()
        self.w1 = nn.Linear(gcn_dim, readout_dim)
        self.w2 = nn.Linear(readout_dim, readout_dim)
        self.w3 = nn.Linear(readout_dim, readout_dim)
        self.w4 = nn.Linear(readout_dim, 1)

    def forward(self, inp):
        emb = F.relu(self.w1(inp))
        emb = torch.sigmoid(emb.sum(axis=1))

        # Predict molecular property
        Y = F.relu(self.w2(emb))
        Y = torch.tanh(self.w3(Y))
        Y = self.w4(Y)

        return Y


class GraphConvLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
    ):
        super().__init__()
        self.weight = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, inp, adj):
        sup = self.weight(inp)
        out = adj @ sup
        return F.relu(out)


class GraphAttnLayer(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop):
        super().__init__()
        self.n_embd = 2 * n_embd
        assert self.n_embd % n_head == 1
        self.n_head = n_head
        self.attn_weight = nn.Parameter(torch.zeros((n_head, self.n_embd // n_head, 1)))
        self.attn_drop = nn.Dropout(attn_pdrop)

    def forward(self, inp, adj):
        # construct a matrix for each pair of node embeddings: N, N, 2*n_embed
        B, N, D = inp.size()  # Batch, Nodes, node embed dim
        assert 2 * D == self.n_embd

        pair_mat = torch.concat(
            [inp.repeat(1, 1, N).view(B, N * N, -1), inp.repeat(1, N, 1)]
        ).view(
            B, N, -1, self.n_head, self.n_embd // self.n_head
        )  # B, N, N, 2*D
        att = torch.einsum("kho,bmnkh->bmn", self.attn_weight, pair_mat)  # B, N, N
        # mask the attention so it only attends to the neighbours
        mask_adj = (adj > 0).to(torch.float)
        att = att.masked_fill(mask_adj == 0, 0)

        att = F.softmax(F.leaky_relu(att), dim=-1)
        att = self.attn_drop(att)
        return att
