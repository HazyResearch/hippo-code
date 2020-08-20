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

    elif measure == 'legs':
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]

    return A, B


def slo(input, N, d_t=1.0, method='trapezoidal'):
    q = np.arange(N)
    col, row = np.meshgrid(q, q)
    r = 2 * q + 1
    M = -(np.where(row >= col, r, 0) - np.diag(q))
    T = np.sqrt(np.diag(2 * q + 1))
    A = T @ M @ np.linalg.inv(T)
    B = np.diag(T)
    # d, V = np.linalg.eig(A)
    # d, V = d[::-1], V[:, ::-1]
    c = np.zeros(N, dtype=np.float64)
    c[0] = input[0]
    for t in range(1, input.shape[0]):
        At = A / t
        Bt = B / t
        u = input[t]
        if method == 'euler' or method == 'forward_diff':
            c = (np.eye(N) + d_t * At) @ c + d_t * Bt * u
        elif method == 'backward_diff' or method == 'backward_euler':
            c = la.solve_triangular(np.eye(N) - d_t * At, c + d_t * Bt * u, lower=True)
        elif method == 'bilinear' or method == 'tustin' or method == 'trapezoidal':
            c = la.solve_triangular(np.eye(N) - d_t / 2 * At, (np.eye(N) + d_t / 2 * At) @ c + d_t * Bt * u, lower=True)
        elif method == 'zoh':
            # aa, bb, _, _, _ = signal.cont2discrete((A, B[:, None], np.ones((1, N)), np.zeros((1,))), dt=math.log(t + d_t) - math.log(t), method='zoh')
            # bb = bb.squeeze(-1)
            aa = la.expm(A * (math.log(t + d_t) - math.log(t)))
            bb = la.solve_triangular(A, aa @ B - B, lower=True)
            c = aa @ c + bb * f(t)
        else:
            assert False, f'method {method} not supported'
    # f_approx = (c @ (T @ ss.eval_legendre(np.arange(N)[:, None], 2 * t_vals / T_max - 1)))
    return c


class LegSTest(unittest.TestCase):

    def setUp(self):
        self.rtol = 10
        self.atol = 1e-3

    def test_legs_euler_forward_cpu(self):
        batch_size = 10
        memsize = 23
        memorder = 587
        dt = 0.27
        # batch_size = 1
        # memsize = 1
        # memorder = 5
        # dt = 0.5
        A, B = transition('legs', memorder)
        A = torch.Tensor(A)
        B = torch.Tensor(B).squeeze(-1)
        x = torch.randn(batch_size, memsize, memorder)
        input = torch.randn(batch_size, memsize)
        out = hippo.legs_euler_forward(x, input, dt)
        out_torch = x + dt * F.linear(x, A) + dt * input.unsqueeze(-1) * B
        out_double = x.double() + dt * F.linear(x.double(), A.double()) + dt * input.unsqueeze(-1).double() * B.double()
        err = (out - out_double).abs().max().item()
        err_torch = (out_torch - out_double).abs().max().item()
        # print(out_double)
        print((out - out_double).abs().max().item())
        print((out_torch - out_double).abs().max().item())
        self.assertTrue(err <= err_torch * (1 + self.rtol) + self.atol,
                        ((out - out_torch).abs().max().item()))

    def test_legs_euler_backward_cpu(self):
        batch_size = 10
        memsize = 23
        memorder = 587
        dt = 0.27
        # batch_size = 1
        # memsize = 1
        # memorder = 5
        # dt = 0.5
        A, B = transition('legs', memorder)
        A_inv = la.solve_triangular(np.eye(memorder) - dt * A, np.eye(memorder), lower=True)
        B_inv = la.solve_triangular(np.eye(memorder) - dt * A, B, lower=True)
        A_inv = torch.Tensor(A_inv)
        B_inv = torch.Tensor(B_inv).squeeze(-1)
        x = torch.randn(batch_size, memsize, memorder)
        input = torch.randn(batch_size, memsize)
        out = hippo.legs_euler_backward(x, input, dt)
        out_torch = F.linear(x, A_inv) + dt * input.unsqueeze(-1) * B_inv
        out_double = F.linear(x.double(), A_inv.double()) + dt * input.unsqueeze(-1).double() * B_inv.double()
        err = (out - out_double).abs().max().item()
        err_torch = (out_torch - out_double).abs().max().item()
        # print(out_double)
        print((out - out_double).abs().max().item())
        print((out_torch - out_double).abs().max().item())
        self.assertTrue(err <= err_torch * (1 + self.rtol) + self.atol,
                        ((out - out_torch).abs().max().item()))

    def test_legs_trapezoidal_cpu(self):
        batch_size = 10
        memsize = 23
        memorder = 587
        dt = 0.27
        # batch_size = 1
        # memsize = 1
        # memorder = 5
        # dt = 0.5
        A, B = transition('legs', memorder)
        trap_A_inv = la.solve_triangular(np.eye(memorder) - dt / 2 * A, np.eye(memorder) + dt / 2 * A, lower=True)
        trap_A_inv = torch.Tensor(trap_A_inv)
        trap_B_inv = la.solve_triangular(np.eye(memorder) - dt / 2 * A, B, lower=True)
        trap_B_inv = torch.Tensor(trap_B_inv).squeeze(-1)
        x = torch.randn(batch_size, memsize, memorder)
        input = torch.randn(batch_size, memsize)
        out = hippo.legs_trapezoidal(x, input, dt)
        out_torch = F.linear(x, trap_A_inv) + dt * input.unsqueeze(-1) * trap_B_inv
        out_double = F.linear(x.double(), trap_A_inv.double()) + dt * input.unsqueeze(-1).double() * trap_B_inv.double()
        err = (out - out_double).abs().max().item()
        err_torch = (out_torch - out_double).abs().max().item()
        # print(out_double)
        print((out - out_double).abs().max().item())
        print((out_torch - out_double).abs().max().item())
        self.assertTrue(err <= err_torch * (1 + self.rtol) + self.atol,
                        ((out - out_torch).abs().max().item()))

    def test_function_approx(self):
        length = int(1e3)
        memorder = 256
        input = torch.randn(length, dtype=torch.float64)
        mem = hippo.legs_function_approx_trapezoidal(input, memorder)
        mem_np = torch.Tensor(slo(input.cpu().numpy().astype(np.float64), memorder)).double()
        self.assertTrue(torch.allclose(mem, mem_np))


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
    A, B = transition('legs', memorder)
    A_inv = la.solve_triangular(np.eye(memorder) - dt * A, np.eye(memorder), lower=True)
    B_inv = la.solve_triangular(np.eye(memorder) - dt * A, B, lower=True)
    A_inv = torch.Tensor(A_inv)
    B_inv = torch.Tensor(B_inv).squeeze(-1)
    trap_A_inv = la.solve_triangular(np.eye(memorder) - dt / 2 * A, np.eye(memorder) + dt / 2 * A, lower=True)
    trap_A_inv = torch.Tensor(trap_A_inv)
    trap_B_inv = la.solve_triangular(np.eye(memorder) - dt / 2 * A, B, lower=True)
    trap_B_inv = torch.Tensor(trap_B_inv).squeeze(-1)
    A = torch.Tensor(A)
    B = torch.Tensor(B).squeeze(-1)
    x = torch.randn(batch_size, memsize, memorder)
    input = torch.randn(batch_size, memsize)
    nsteps = 10000
    euler_forward_fn = lambda: hippo.legs_euler_forward(x, input, dt)
    euler_forward_torch_fn = lambda: x + dt * F.linear(x, A) + dt * input.unsqueeze(-1) * B
    euler_backward_fn = lambda: hippo.legs_euler_backward(x, input, dt)
    euler_backward_torch_fn = lambda: F.linear(x, A_inv) + dt * input.unsqueeze(-1) * B_inv
    trapezoidal_fn = lambda: hippo.legs_trapezoidal(x, input, dt)
    trapezoidal_torch_fn = lambda: F.linear(x, trap_A_inv) + dt * input.unsqueeze(-1) * trap_B_inv
    print(f'Euler forward C++: {timeit(euler_forward_fn, nsteps)}s')
    print(f'Euler backward C++: {timeit(euler_backward_fn, nsteps)}s')
    print(f'Trapezoidal C++: {timeit(trapezoidal_fn, nsteps)}s')
    print(f'Euler forward Pytorch: {timeit(euler_forward_torch_fn, nsteps)}s')
    print(f'Euler backward Pytorch: {timeit(euler_backward_torch_fn, nsteps)}s')
    print(f'Trapezoidal Pytorch: {timeit(trapezoidal_torch_fn, nsteps)}s')

    length = int(1e6)
    input = torch.randn(length, dtype=torch.float64)
    trap_func_approx_fn = lambda: hippo.legs_function_approx_trapezoidal(input, memorder)
    nsteps = 1
    print(f'Function approx trapezoidal C++: {timeit(trap_func_approx_fn, nsteps)}s')


if __name__ == "__main__":
    benchmark()
