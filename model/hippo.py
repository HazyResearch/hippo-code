import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal
from scipy import linalg as la

from model import unroll
from model.op import transition

# forward_aliases   = ['euler', 'forward_euler', 'forward', 'forward_diff']
# backward_aliases  = ['backward', 'backward_diff', 'backward_euler']
# bilinear_aliases = ['bilinear', 'tustin', 'trapezoidal', 'trapezoid']
# zoh_aliases       = ['zoh']


class HiPPO(nn.Module):
    def __init__(self, N, dt=1.0, measure='legt', discretization='bilinear'):
        """
        N: the order of the HiPPO projection
        dt: discretization step size - should be roughly inverse to the length of the sequence
        """
        super().__init__()
        self.N = N
        A, B = transition(measure, N)
        C = np.ones((1, N))
        D = np.zeros((1,))
        # dt, discretization options
        A, B, _, _, _ = signal.cont2discrete((A, B, C, D), dt=dt, method=discretization)

        B = B.squeeze(-1)

        self.register_buffer('A', torch.Tensor(A)) # (N, N)
        self.register_buffer('B', torch.Tensor(B)) # (N,)

    def forward(self, inputs):
        """
        inputs : (length, ...)
        output : (length, ..., N) where N is the order of the HiPPO projection
        """

        inputs = inputs.unsqueeze(-1)
        u = inputs * self.B # (length, ..., N)

        c = torch.zeros(u.shape[1:])
        cs = []
        for f in inputs:
            c = F.linear(c, self.A) + self.B * f
            cs.append(c)
        return torch.stack(cs, dim=0)



class HiPPO_LegS(nn.Module):
    """ Vanilla HiPPO-LegS model (scale invariant instead of time invariant) """
    def __init__(self, N, max_length=1024, measure='legs', discretization='bilinear'):
        """
        max_length: maximum sequence length
        """
        super().__init__()
        self.N = N
        A, B = transition(measure, N)
        B = B.squeeze(-1)
        A_stacked = np.empty((max_length, N, N), dtype=A.dtype)
        B_stacked = np.empty((max_length, N), dtype=B.dtype)
        for t in range(1, max_length + 1):
            At = A / t
            Bt = B / t
            if discretization == 'forward':
                A_stacked[t - 1] = np.eye(N) + At
                B_stacked[t - 1] = Bt
            elif discretization == 'backward':
                A_stacked[t - 1] = la.solve_triangular(np.eye(N) - At, np.eye(N), lower=True)
                B_stacked[t - 1] = la.solve_triangular(np.eye(N) - At, Bt, lower=True)
            elif discretization == 'bilinear':
                A_stacked[t - 1] = la.solve_triangular(np.eye(N) - At / 2, np.eye(N) + At / 2, lower=True)
                B_stacked[t - 1] = la.solve_triangular(np.eye(N) - At / 2, Bt, lower=True)
            else: # ZOH
                A_stacked[t - 1] = la.expm(A * (math.log(t + 1) - math.log(t)))
                B_stacked[t - 1] = la.solve_triangular(A, A_stacked[t - 1] @ B - B, lower=True)
        self.A_stacked = torch.Tensor(A_stacked) # (max_length, N, N)
        self.B_stacked = torch.Tensor(B_stacked) # (max_length, N)
        # print("B_stacked shape", B_stacked.shape)

    def forward(self, inputs, fast=False):
        """
        inputs : (length, ...)
        output : (length, ..., N) where N is the order of the HiPPO projection
        """

        L = inputs.shape[0]

        inputs = inputs.unsqueeze(-1)
        u = torch.transpose(inputs, 0, -2)
        u = u * self.B_stacked[:L]
        u = torch.transpose(u, 0, -2) # (length, ..., N)

        if fast:
            result = unroll.variable_unroll_matrix(self.A_stacked[:L], u)
        else:
            result = unroll.variable_unroll_matrix_sequential(self.A_stacked[:L], u)
        return result

if __name__ == '__main__':
    N = 100
    L = 200
    hippo = HiPPO(N)
    hippo_legs = HiPPO_LegS(N)

    x = torch.randn(L, 1)

    y = hippo(x)
    # print(y.shape)
    y = hippo_legs(x)
    # print(y.shape)
    z = hippo_legs(x, fast=True)
    # print(z.shape)
    # print(y-z)
