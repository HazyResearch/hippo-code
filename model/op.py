import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal
from scipy import linalg as la
from scipy import special as ss


def transition(measure, N, **measure_args):
    """ A, B transition matrices for different measures.

    measure: the type of measure
      legt - Legendre (translated)
      legs - Legendre (scaled)
      glagt - generalized Laguerre (translated)
      lagt, tlagt - previous versions of (tilted) Laguerre with slightly different normalization
    """
    # Laguerre (translated)
    if measure == 'lagt':
        b = measure_args.get('beta', 1.0)
        A = np.eye(N) / 2 - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    if measure == 'tlagt':
        # beta = 1 corresponds to no tilt
        b = measure_args.get('beta', 1.0)
        A = (1.-b)/2 * np.eye(N) - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    # Generalized Laguerre
    # alpha 0, beta small is most stable (limits to the 'lagt' measure)
    # alpha 0, beta 1 has transition matrix A = [lower triangular 1]
    if measure == 'glagt':
        alpha = measure_args.get('alpha', 0.0)
        beta = measure_args.get('beta', 0.01)
        A = -np.eye(N) * (1 + beta) / 2 - np.tril(np.ones((N, N)), -1)
        B = ss.binom(alpha + np.arange(N), np.arange(N))[:, None]

        L = np.exp(.5 * (ss.gammaln(np.arange(N)+alpha+1) - ss.gammaln(np.arange(N)+1)))
        A = (1./L[:, None]) * A * L[None, :]
        B = (1./L[:, None]) * B * np.exp(-.5 * ss.gammaln(1-alpha)) * beta**((1-alpha)/2)
    # Legendre (translated)
    elif measure == 'legt':
        Q = np.arange(N, dtype=np.float64)
        R = (2*Q + 1) ** .5
        j, i = np.meshgrid(Q, Q)
        A = R[:, None] * np.where(i < j, (-1.)**(i-j), 1) * R[None, :]
        B = R[:, None]
        A = -A
    # LMU: equivalent to LegT up to normalization
    elif measure == 'lmu':
        Q = np.arange(N, dtype=np.float64)
        R = (2*Q + 1)[:, None] # / theta
        j, i = np.meshgrid(Q, Q)
        A = np.where(i < j, -1, (-1.)**(i-j+1)) * R
        B = (-1.)**Q[:, None] * R
    # Legendre (scaled)
    elif measure == 'legs':
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]

    return A, B



class AdaptiveTransition(nn.Module):
    def precompute_forward(self):
        raise NotImplementedError

    def precompute_backward(self):
        raise NotImplementedError

    def forward_mult(self, u, delta):
        """ Computes (I + delta A) u

        A: (n, n)
        u: (..., n)
        delta: (...) or scalar

        output: (..., n)
        """
        raise NotImplementedError

    def inverse_mult(self, u, delta): # TODO swap u, delta everywhere
        """ Computes (I - d A)^-1 u """
        raise NotImplementedError

    # @profile
    def forward_diff(self, d, u, v, **kwargs):
        """ Computes the 'forward diff' or Euler update rule: (I - d A)^-1 u + d B v
        d: (...)
        u: (..., n)
        v: (...)
        """
        # TODO F.linear should be replaced by broadcasting, self.B shouldl be shape (n) instead of (n, 1)
        # x = self.forward_mult(u, d) + dt * F.linear(v.unsqueeze(-1), self.B)
        v = d * v
        v = v.unsqueeze(-1) * self.B
        x = self.forward_mult(u, d, **kwargs)
        x = x + v
        return x

    # @profile
    def backward_diff(self, d, u, v, **kwargs):
        """ Computes the 'forward diff' or Euler update rule: (I - d A)^-1 u + d (I - d A)^-1 B v
        d: (...)
        u: (..., n)
        v: (...)
        """
        v = d * v
        v = v.unsqueeze(-1) * self.B
        x = u + v
        x = self.inverse_mult(x, d, **kwargs)
        return x

    # @profile
    def bilinear(self, dt, u, v, alpha=.5, **kwargs):
        """ Computes the bilinear (aka trapezoid or Tustin's) update rule.

        (I - d/2 A)^-1 (I + d/2 A) u + d B (I - d/2 A)^-1 B v
        """
        x = self.forward_mult(u, (1-alpha)*dt, **kwargs)
        v = dt * v
        v = v.unsqueeze(-1) * self.B
        x = x + v
        x = self.inverse_mult(x, (alpha)*dt, **kwargs)
        return x

    def zoh(self, dt, u, v):
        raise NotImplementedError

    def precompute(self, deltas):
        """ deltas: list of step sizes """
        for delta in deltas:
            # self.forward_cache[delta] = self.precompute_forward(delta)
            # self.backward_cache[delta] = self.precompute_backward(delta)
            # TODO being lazy here; should check whether bilinear rule is being used
            self.forward_cache[delta/2] = self.precompute_forward(delta/2)
            self.backward_cache[delta/2] = self.precompute_backward(delta/2)



class ManualAdaptiveTransition(AdaptiveTransition):
    def __init__(self, N, **kwargs):
        """ Slow (n^3, or n^2 if step sizes are cached) version via manual matrix mult/inv

        delta: optional list of step sizes to cache the transitions for
        """
        super().__init__()
        A, B = transition(type(self).measure, N, **kwargs)
        self.N = N
        self.register_buffer('A', torch.Tensor(A))
        self.register_buffer('B', torch.Tensor(B[:, 0]))
        self.register_buffer('I', torch.eye(self.N))

        # Precompute stacked A, B matrix for zoh computation
        AB = torch.cat((self.A, self.B.unsqueeze(-1)), dim=-1)
        AB = torch.cat((AB, torch.zeros((1, N+1))), dim=0)
        self.register_buffer('AB', AB)

        self.forward_cache = {}
        self.backward_cache = {}

        print(f"ManualAdaptiveTransition:\n  A {self.A}\nB {self.B}")

    def precompute_forward(self, delta):
        return self.I + delta*self.A

    def precompute_backward(self, delta):
        return torch.triangular_solve(self.I, self.I - delta*self.A, upper=False)[0]

    def precompute_exp(self, delta):
        # NOTE this does not work because torch has no matrix exponential yet, support ongoing:
        # https://github.com/pytorch/pytorch/issues/9983
        e = torch.expm(delta * self.AB)
        return e[:-1, :-1], e[:-1, -1]

    # @profile
    def forward_mult(self, u, delta, precompute=True):
        """ Computes (I + d A) u

        A: (n, n)
        u: (b1* d, n) d represents memory_size
        delta: (b2*, d) or scalar
          Assume len(b2) <= len(b1)

        output: (broadcast(b1, b2)*, d, n)
        """

        # For forward Euler, precompute materializes the matrix
        if precompute:
            if isinstance(delta, torch.Tensor):
                delta = delta.unsqueeze(-1).unsqueeze(-1)
            # print(delta, isinstance(delta, float), delta in self.forward_cache)
            if isinstance(delta, float) and delta in self.forward_cache:
                mat = self.forward_cache[delta]
            else:
                mat = self.precompute_forward(delta)
            if len(u.shape) >= len(mat.shape):
                # For memory efficiency, leverage extra batch dimensions
                s = len(u.shape)
                # TODO can make the permutation more efficient by just permuting the last 2 or 3 dim, but need to do more casework)
                u = u.permute(list(range(1, s)) + [0])
                x = mat @ u
                x = x.permute([s-1] + list(range(s-1)))
            else:
                x = (mat @ u.unsqueeze(-1))[..., 0]
            # x = F.linear(u, mat)
        else:
            if isinstance(delta, torch.Tensor):
                delta = delta.unsqueeze(-1)
            x = F.linear(u, self.A)
            x = u + delta * x

        return x


    # @profile
    def inverse_mult(self, u, delta, precompute=True):
        """ Computes (I - d A)^-1 u """

        if isinstance(delta, torch.Tensor):
            delta = delta.unsqueeze(-1).unsqueeze(-1)

        if precompute:
            if isinstance(delta, float) and delta in self.backward_cache:
                mat = self.backward_cache[delta]
            else:
                mat = self.precompute_backward(delta) # (n, n) or (..., n, n)

            if len(u.shape) >= len(mat.shape):
                # For memory efficiency, leverage extra batch dimensions
                s = len(u.shape)
                # TODO can make the permutation more efficient by just permuting the last 2 or 3 dim, but need to do more casework
                u = u.permute(list(range(1, s)) + [0])
                x = mat @ u
                x = x.permute([s-1] + list(range(s-1)))
            else:
                x = (mat @ u.unsqueeze(-1))[..., 0]

        else:
            _A = self.I - delta*self.A
            x = torch.triangular_solve(u.unsqueeze(-1), _A, upper=False)[0]
            x = x[..., 0]

        return x

    def zoh(self, dt, u, v):
        dA, dB = self.precompute_exp(dt)
        return F.linear(u, dA) + dB * v.unsqueeze(-1)

class LegSAdaptiveTransitionManual(ManualAdaptiveTransition):
    measure = 'legs'

class LegTAdaptiveTransitionManual(ManualAdaptiveTransition):
    measure = 'legt'

class LagTAdaptiveTransitionManual(ManualAdaptiveTransition):
    measure = 'lagt'

class TLagTAdaptiveTransitionManual(ManualAdaptiveTransition):
    measure = 'tlagt'
