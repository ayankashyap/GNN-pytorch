import torch

def normalize(adj):
    """return the normalizing diagonal matrix"""
    assert adj.dtype == torch.float
    row_sum = adj.sum(axis=1)
    r_inv = torch.diag(row_sum**-1)
    r_inv[r_inv == float("inf")] = 0
    norm_adj = r_inv @ adj
    return norm_adj

