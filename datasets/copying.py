import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data.dataset import IterableDataset
import numpy as np


def np_copying_data(L, M, A, batch_shape=()):
    seq = np.random.randint(low=1, high=A-1, size=batch_shape+(M,))
    zeros_x = np.zeros(batch_shape+(L,))
    markers = (A-1) * np.ones(batch_shape+(M,))
    zeros_y = np.zeros(batch_shape+(M+L,))

    x_ = np.concatenate([seq, zeros_x, markers], axis=-1)
    y_ = np.concatenate([zeros_y, seq], axis=-1)
    x = F.one_hot(torch.tensor(x_, dtype=torch.int64), A).float()
    y = torch.tensor(y_, dtype=torch.int64)
    return x, y

def torch_copying_data(L, M, A, variable=False, batch_shape=()):
    tokens = torch.randint(low=1, high=A-1, size=batch_shape+(M,))
    if variable:
        total_batch = np.prod(batch_shape)
        inds = torch.stack([
            torch.randperm(L+M)[:M]
            for _ in range(total_batch)
            ], 0)
        inds = inds.reshape(batch_shape+(M,))
        inds, _ = inds.sort()
    else:
        inds = torch.arange(M).repeat(batch_shape+(1,))
    zeros_x = torch.zeros(batch_shape+(M+L,), dtype=torch.long)
    zeros_x.scatter_(-1, inds, tokens)
    markers = (A-1) * torch.ones(batch_shape+(M,), dtype=torch.long)

    x_ = torch.cat([zeros_x, markers], dim=-1)
    y_ = torch.cat([tokens], dim=-1)
    x = F.one_hot(x_, A).float()
    y = y_
    return x, y


def copying_static_dataset(L, M, A, variable, samples):
    all_x, all_y = torch_copying_data(L, M, A, variable, batch_shape=(samples,))
    print("Constructing Copying dataset of shape", all_x.shape)
    ds = torch.utils.data.TensorDataset(all_x, all_y)
    return ds
