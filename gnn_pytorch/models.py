import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

from .blocks import GraphConvLayer, GraphAttnLayer, ReadOut


logger = logging.getLogger(__name__)


class GNN_Config:
    """(Most) GNN Config params according to https://arxiv.org/pdf/1805.10988.pdf"""

    attn_pdrop = 0.1
    gcn_pdrop = 0.1
    n_head = 4
    inp_dim = 58
    gcn_dim = 32  # in the paper every gcn layer is set to 32
    readout_dim = 512
    n_layers = 4

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            self.__setattr__(k, v)


class GCN_Stack(nn.ModuleList):
    """This block implements a forward function for nn.ModuleList
    The reasoning for this is that nn.Sequential doesn't allow <1 args 
    in their forward and we need to have both the node matrix """
    def forward(self, x, adj):
        for i, module in enumerate(self):
            if i == 0:
                x = module(x, adj)
            else:
                x = x + module(x, adj)
        return x    


class VanillaGCN(nn.Module):
    """Vanilla GCN with no fancy attention and gates"""

    def __init__(self, config):
        super().__init__()

        # setup the stack of gcn layers
        dims = [config.inp_dim]
        for _ in range((2 * config.n_layers) - 1):
            dims.append(config.gcn_dim)
        self.gcn_stack = GCN_Stack(
            [GraphConvLayer(dims[i], dims[i + 1]) for i in range(config.n_layers - 1)]
        )

        self.gcn_drop = nn.Dropout(config.gcn_pdrop)
        self.readout = ReadOut(config.gcn_dim, config.readout_dim)

        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        # initializing all weights to xavier as mentioned in the paper
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, inp, adj, targets=None):
        x = self.gcn_stack(inp, adj)
        x = self.gcn_drop(x)
        pred = self.readout(x)
        if targets is not None:
            # if training, return prediction as well asRMSE loss
            loss = torch.sqrt(F.mse_loss(pred, targets))
            return pred, loss
        return pred
