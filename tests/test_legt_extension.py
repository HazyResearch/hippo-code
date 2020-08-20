import math
import unittest

import numpy as np
from scipy import linalg as la

import torch
import torch.nn.functional as F
import hippo

# from .op import transition

def transition(measure, N, **measure_args):
    """ A, B transition matrices for different measures """
    if measure == 'lagt':
        # A_l = (1 - dt / 4) * np.eye(N) + dt / 2 * np.tril(np.ones((N, N)))
        # A_r = (1 + dt / 4) * np.eye(N) - dt / 2 * np.tril(np.ones((N, N)))
        # alpha = dt / 2 / (1 - dt / 4)
        # col = -alpha / (1 + alpha) ** np.arange(1, N + 1)
        # col[0] += 1
        # A_l_inv = la.toeplitz(col / (1 - dt / 4), np.zeros(N))
        b = measure_args.get('beta', 1.0)
        A = np.eye(N) / 2 - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    if measure == 'tlagt':
        # beta = 1 corresponds to no tilt
        # b = measure_args['beta']
        b = measure_args.get('beta', 1.0)
        A = (1.-b)/2 * np.eye(N) - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    elif measure == 'legt':
        Q = np.arange(N, dtype=np.float64)
        R = (2*Q + 1)[:, None] # / theta
        j, i = np.meshgrid(Q, Q)
        A = np.where(i < j, -1, (-1.)**(i-j+1)) * R
        B = (-1.)**Q[:, None] * R

    elif measure == 'legt':
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]

    return A, B


class LegtTest(unittest.TestCase):

    def setUp(self):
        self.rtol = 10
        self.atol = 1e-3

    def test_legt_euler_forward_cpu(self):
        batch_size = 10
        memsize = 23
        memorder = 587
        dt = 0.27
        # batch_size = 1
        # memsize = 1
        # memorder = 5
        # dt = 0.5
        A, B = transition('legt', memorder)
        A = torch.Tensor(A)
        B = torch.Tensor(B).squeeze(-1)
        x = torch.randn(batch_size, memsize, memorder)
        input = torch.randn(batch_size, memsize)
        out = hippo.legt_euler_forward(x, input, dt)
        out_torch = x + dt * F.linear(x, A) + dt * input.unsqueeze(-1) * B
        out_double = x.double() + dt * F.linear(x.double(), A.double()) + dt * input.unsqueeze(-1).double() * B.double()
        err = (out - out_double).abs().max().item()
        err_torch = (out_torch - out_double).abs().max().item()
        # print(out_double)
        print((out - out_double).abs().max().item())
        print((out_torch - out_double).abs().max().item())
        self.assertTrue(err <= err_torch * (1 + self.rtol) + self.atol,
                        ((out - out_torch).abs().max().item()))

def timeit(fn, nsteps):
    import time
    fn()
    start = time.perf_counter()
    for _ in range(nsteps):
        fn()
    end = time.perf_counter()
    return (end - start) / nsteps


def benchmark():
    torch.set_num_threads(1)
    batch_size = 1
    memsize = 1
    memorder = 256
    dt = 0.27
    A, B = transition('legt', memorder)
    A = torch.Tensor(A)
    B = torch.Tensor(B).squeeze(-1)
    x = torch.randn(batch_size, memsize, memorder)
    input = torch.randn(batch_size, memsize)
    nsteps = 10000
    euler_forward_fn = lambda: hippo.legt_euler_forward(x, input, dt)
    euler_forward_torch_fn = lambda: x + dt * F.linear(x, A) + dt * input.unsqueeze(-1) * B
    print(f'Euler forward C++: {timeit(euler_forward_fn, nsteps)}s')
    print(f'Euler forward Pytorch: {timeit(euler_forward_torch_fn, nsteps)}s')

if __name__ == "__main__":
    benchmark()
