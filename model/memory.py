import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from scipy import signal
from scipy import linalg as la
from functools import partial

from model.rnncell import RNNCell
from model.orthogonalcell import OrthogonalLinear
from model.components import Gate, Linear_, Modrelu, get_activation, get_initializer
from model.op import LegSAdaptiveTransitionManual, LegTAdaptiveTransitionManual, LagTAdaptiveTransitionManual, TLagTAdaptiveTransitionManual



forward_aliases   = ['euler', 'forward_euler', 'forward', 'forward_diff']
backward_aliases  = ['backward', 'backward_diff', 'backward_euler']
bilinear_aliases = ['bilinear', 'tustin', 'trapezoidal', 'trapezoid']
zoh_aliases       = ['zoh']


class MemoryCell(RNNCell):

    name = None
    valid_keys = ['uxh', 'ux', 'uh', 'um', 'hxm', 'hx', 'hm', 'hh', 'bias', ]

    def default_initializers(self):
        return {
            'uxh': 'uniform',
            'hxm': 'xavier',
            'hx': 'xavier',
            'hm': 'xavier',

            'um': 'zero',
            'hh': 'xavier',
        }


    def default_architecture(self):
        return {
            'ux': True,
            # 'uh': True,
            'um': False,
            'hx': True,
            'hm': True,
            'hh': False,
            'bias': True,
        }


    def __init__(self, input_size, hidden_size, memory_size, memory_order,
                 memory_activation='id',
                 gate='G', # 'N' | 'G' | UR'
                 memory_output=False,
                 **kwargs
                 ):
        self.memory_size       = memory_size
        self.memory_order      = memory_order

        self.memory_activation = memory_activation
        self.gate              = gate
        self.memory_output     = memory_output

        super(MemoryCell, self).__init__(input_size, hidden_size, **kwargs)


        self.input_to_hidden_size = self.input_size if self.architecture['hx'] else 0
        self.input_to_memory_size = self.input_size if self.architecture['ux'] else 0

        # Construct and initialize u
        self.W_uxh = nn.Linear(self.input_to_memory_size + self.hidden_size, self.memory_size,
                               bias=self.architecture['bias'])
        # nn.init.zeros_(self.W_uxh.bias)
        if 'uxh' in self.initializers:
            get_initializer(self.initializers['uxh'], self.memory_activation)(self.W_uxh.weight)
        if 'ux' in self.initializers:  # Re-init if passed in
            get_initializer(self.initializers['ux'], self.memory_activation)(self.W_uxh.weight[:, :self.input_size])
        if 'uh' in self.initializers:  # Re-init if passed in
            get_initializer(self.initializers['uh'], self.memory_activation)(self.W_uxh.weight[:, self.input_size:])


        # Construct and initialize h
        self.memory_to_hidden_size = self.memory_size * self.memory_order if self.architecture['hm'] else 0
        preact_ctor = Linear_
        preact_args = [self.input_to_hidden_size + self.memory_to_hidden_size, self.hidden_size,
                       self.architecture['bias']]

        self.W_hxm = preact_ctor(*preact_args)

        if self.initializers.get('hxm', None) is not None:  # Re-init if passed in
            get_initializer(self.initializers['hxm'], self.hidden_activation)(self.W_hxm.weight)
        if self.initializers.get('hx', None) is not None:  # Re-init if passed in
            get_initializer(self.initializers['hx'], self.hidden_activation)(self.W_hxm.weight[:, :self.input_size])
        if self.initializers.get('hm', None) is not None:  # Re-init if passed in
            get_initializer(self.initializers['hm'], self.hidden_activation)(self.W_hxm.weight[:, self.input_size:])

        if self.architecture['um']:
            # No bias here because the implementation is awkward otherwise, but probably doesn't matter
            self.W_um = nn.Parameter(torch.Tensor(self.memory_size, self.memory_order))
            get_initializer(self.initializers['um'], self.memory_activation)(self.W_um)

        if self.architecture['hh']:
            self.reset_hidden_to_hidden()
        else:
            self.W_hh = None

        if self.gate is not None:
            if self.architecture['hh']:
                print("input to hidden size, memory to hidden size, hidden size:", self.input_to_hidden_size, self.memory_to_hidden_size, self.hidden_size)
                preact_ctor = Linear_
                preact_args = [self.input_to_hidden_size + self.memory_to_hidden_size + self.hidden_size, self.hidden_size,
                               self.architecture['bias']]
            self.W_gxm = Gate(self.hidden_size, preact_ctor, preact_args, mechanism=self.gate)

    def reset_parameters(self):
        # super().reset_parameters()
        self.hidden_activation_fn = get_activation(self.hidden_activation, self.hidden_size) # TODO figure out how to remove this duplication
        self.memory_activation_fn = get_activation(self.memory_activation, self.memory_size)

    def forward(self, input, state):
        h, m, time_step = state

        input_to_hidden = input if self.architecture['hx'] else input.new_empty((0,))
        input_to_memory = input if self.architecture['ux'] else input.new_empty((0,))

        # Construct the update features
        memory_preact = self.W_uxh(torch.cat((input_to_memory, h), dim=-1))  # (batch, memory_size)
        if self.architecture['um']:
            memory_preact = memory_preact + (m * self.W_um).sum(dim=-1)
        u = self.memory_activation_fn(memory_preact) # (batch, memory_size)

        # Update the memory
        m = self.update_memory(m, u, time_step) # (batch, memory_size, memory_order)

        # Update hidden state from memory
        if self.architecture['hm']:
            memory_to_hidden = m.view(input.shape[0], self.memory_size*self.memory_order)
        else:
            memory_to_hidden = input.new_empty((0,))
        m_inputs = (torch.cat((input_to_hidden, memory_to_hidden), dim=-1),)
        hidden_preact = self.W_hxm(*m_inputs)

        if self.architecture['hh']:
            hidden_preact = hidden_preact + self.W_hh(h)
        hidden = self.hidden_activation_fn(hidden_preact)


        # Construct gate if necessary
        if self.gate is None:
            h = hidden
        else:
            if self.architecture['hh']:
                m_inputs = torch.cat((m_inputs[0], h), -1),
            g = self.W_gxm(*m_inputs)
            h = (1.-g) * h + g * hidden

        next_state = (h, m, time_step + 1)
        output = self.output(next_state)

        return output, next_state

    def update_memory(self, m, u, time_step):
        """
        m: (B, M, N) [batch size, memory size, memory order]
        u: (B, M)

        Output: (B, M, N)
        """
        raise NotImplementedError

    def default_state(self, input, batch_size=None):
        batch_size = input.size(0) if batch_size is None else batch_size
        return (input.new_zeros(batch_size, self.hidden_size, requires_grad=False),
                input.new_zeros(batch_size, self.memory_size, self.memory_order, requires_grad=False),
                0)

    def output(self, state):
        """ Converts a state into a single output (tensor) """
        h, m, time_step = state

        if self.memory_output:
            hm = torch.cat((h, m.view(m.shape[0], self.memory_size*self.memory_order)), dim=-1)
            return hm
        else:
            return h

    def state_size(self):
        return self.hidden_size + self.memory_size*self.memory_order

    def output_size(self):
        if self.memory_output:
            return self.hidden_size + self.memory_size*self.memory_order
        else:
            return self.hidden_size


class LTICell(MemoryCell):
    """ A cell implementing Linear Time Invariant dynamics: c' = Ac + Bf. """

    def __init__(self, input_size, hidden_size, memory_size, memory_order,
                 A, B,
                 trainable_scale=0., # how much to scale LR on A and B
                 dt=0.01,
                 discretization='zoh',
                 **kwargs
                 ):
        super().__init__(input_size, hidden_size, memory_size, memory_order, **kwargs)


        C = np.ones((1, memory_order))
        D = np.zeros((1,))
        dA, dB, _, _, _ = signal.cont2discrete((A, B, C, D), dt=dt, method=discretization)

        dA = dA - np.eye(memory_order)  # puts into form: x += Ax
        self.trainable_scale = np.sqrt(trainable_scale)
        if self.trainable_scale <= 0.:
            self.register_buffer('A', torch.Tensor(dA))
            self.register_buffer('B', torch.Tensor(dB))
        else:
            self.A = nn.Parameter(torch.Tensor(dA / self.trainable_scale), requires_grad=True)
            self.B = nn.Parameter(torch.Tensor(dB / self.trainable_scale), requires_grad=True)

    # TODO: proper way to implement LR scale is a preprocess() function that occurs once per unroll
    # also very useful for orthogonal params
    def update_memory(self, m, u, time_step):
        u = u.unsqueeze(-1) # (B, M, 1)
        if self.trainable_scale <= 0.:
            return m + F.linear(m, self.A) + F.linear(u, self.B)
        else:
            return m + F.linear(m, self.A * self.trainable_scale) + F.linear(u, self.B * self.trainable_scale)

class LSICell(MemoryCell):
    """ A cell implementing Linear 'Scale' Invariant dynamics: c' = 1/t (Ac + Bf). """

    def __init__(self, input_size, hidden_size, memory_size, memory_order,
                 A, B,
                 init_t = 0,  # 0 for special case at t=0 (new code), else old code without special case
                 max_length=1024,
                 discretization='bilinear',
                 **kwargs
                 ):
        """
        # TODO: make init_t start at arbitrary time (instead of 0 or 1)
        """

        # B should have shape (N, 1)
        assert len(B.shape) == 2 and B.shape[1] == 1

        super().__init__(input_size, hidden_size, memory_size, memory_order, **kwargs)

        assert isinstance(init_t, int)
        self.init_t = init_t
        self.max_length = max_length

        A_stacked = np.empty((max_length, memory_order, memory_order), dtype=A.dtype)
        B_stacked = np.empty((max_length, memory_order), dtype=B.dtype)
        B = B[:,0]
        N = memory_order
        for t in range(1, max_length + 1):
            At = A / t
            Bt = B / t
            if discretization in forward_aliases:
                A_stacked[t - 1] = np.eye(N) + At
                B_stacked[t - 1] = Bt
            elif discretization in backward_aliases:
                A_stacked[t - 1] = la.solve_triangular(np.eye(N) - At, np.eye(N), lower=True)
                B_stacked[t - 1] = la.solve_triangular(np.eye(N) - At, Bt, lower=True)
            elif discretization in bilinear_aliases:
                A_stacked[t - 1] = la.solve_triangular(np.eye(N) - At / 2, np.eye(N) + At / 2, lower=True)
                B_stacked[t - 1] = la.solve_triangular(np.eye(N) - At / 2, Bt, lower=True)
            elif discretization in zoh_aliases:
                A_stacked[t - 1] = la.expm(A * (math.log(t + 1) - math.log(t)))
                B_stacked[t - 1] = la.solve_triangular(A, A_stacked[t - 1] @ B - B, lower=True)
        B_stacked = B_stacked[:, :, None]

        A_stacked -= np.eye(memory_order)  # puts into form: x += Ax
        self.register_buffer('A', torch.Tensor(A_stacked))
        self.register_buffer('B', torch.Tensor(B_stacked))


    def update_memory(self, m, u, time_step):
        u = u.unsqueeze(-1) # (B, M, 1)
        t = time_step - 1 + self.init_t
        if t < 0:
            return F.pad(u, (0, self.memory_order - 1))
        else:
            if t >= self.max_length: t = self.max_length - 1
            return m + F.linear(m, self.A[t]) + F.linear(u, self.B[t])


class TimeMemoryCell(MemoryCell):
    """ MemoryCell with timestamped data """
    def __init__(self, input_size, hidden_size, memory_size, memory_order, **kwargs):
        super().__init__(input_size-1, hidden_size, memory_size, memory_order, **kwargs)
    def forward(self, input, state):
        h, m, time_step = state
        timestamp, input = input[:, 0], input[:, 1:]

        input_to_hidden = input if self.architecture['hx'] else input.new_empty((0,))
        input_to_memory = input if self.architecture['ux'] else input.new_empty((0,))

        # Construct the update features
        memory_preact = self.W_uxh(torch.cat((input_to_memory, h), dim=-1))  # (batch, memory_size)
        if self.architecture['um']:
            memory_preact = memory_preact + (m * self.W_um).sum(dim=-1)
        u = self.memory_activation_fn(memory_preact) # (batch, memory_size)

        # Update the memory
        m = self.update_memory(m, u, time_step, timestamp) # (batch, memory_size, memory_order)

        # Update hidden state from memory
        if self.architecture['hm']:
            memory_to_hidden = m.view(input.shape[0], self.memory_size*self.memory_order)
        else:
            memory_to_hidden = input.new_empty((0,))
        m_inputs = (torch.cat((input_to_hidden, memory_to_hidden), dim=-1),)
        hidden_preact = self.W_hxm(*m_inputs)

        if self.architecture['hh']:
            hidden_preact = hidden_preact + self.W_hh(h)
        hidden = self.hidden_activation_fn(hidden_preact)


        # Construct gate if necessary
        if self.gate is None:
            h = hidden
        else:
            if self.architecture['hh']:
                m_inputs = torch.cat((m_inputs[0], h), -1),
            g = self.W_gxm(*m_inputs)
            h = (1.-g) * h + g * hidden

        next_state = (h, m, timestamp)
        output = self.output(next_state)

        return output, next_state

class TimeLSICell(TimeMemoryCell):
    """ A cell implementing "Linear Scale Invariant" dynamics: c' = Ac + Bf with timestamped inputs. """

    name = 'tlsi'

    def __init__(self, input_size, hidden_size, memory_size=1, memory_order=-1,
                 measure='legs',
                 measure_args={},
                 method='manual',
                 discretization='bilinear',
                 **kwargs
                 ):
        if memory_order < 0:
            memory_order = hidden_size


        super().__init__(input_size, hidden_size, memory_size, memory_order, **kwargs)

        assert measure in ['legs', 'lagt', 'tlagt', 'legt']
        assert method in ['manual', 'linear', 'toeplitz']
        if measure == 'legs':
            if method == 'manual':
                self.transition = LegSAdaptiveTransitionManual(self.memory_order)
                kwargs = {'precompute': False}
        if measure == 'legt':
            if method == 'manual':
                self.transition = LegTAdaptiveTransitionManual(self.memory_order)
                kwargs = {'precompute': False}
        elif measure == 'lagt':
            if method == 'manual':
                self.transition = LagTAdaptiveTransitionManual(self.memory_order)
                kwargs = {'precompute': False}
        elif measure == 'tlagt':
            if method == 'manual':
                self.transition = TLagTAdaptiveTransitionManual(self.memory_order, **measure_args)
                kwargs = {'precompute': False}

        if discretization in forward_aliases:
            self.transition_fn = partial(self.transition.forward_diff, **kwargs)
        elif discretization in backward_aliases:
            self.transition_fn = partial(self.transition.backward_diff, **kwargs)
        elif discretization in bilinear_aliases:
            self.transition_fn = partial(self.transition.bilinear, **kwargs)
        else: assert False


    def update_memory(self, m, u, t0, t1):
        """
        m: (B, M, N) [batch, memory_size, memory_order]
        u: (B, M)
        t0: (B,) previous time
        t1: (B,) current time
        """

        if torch.eq(t1, 0.).any():
            return F.pad(u.unsqueeze(-1), (0, self.memory_order - 1))
        else:
            dt = ((t1-t0)/t1).unsqueeze(-1)
            m = self.transition_fn(dt, m, u)
        return m

class TimeLTICell(TimeLSICell):
    """ A cell implementing Linear Time Invariant dynamics: c' = Ac + Bf with timestamped inputs. """

    name = 'tlti'

    def __init__(self, input_size, hidden_size, memory_size=1, memory_order=-1,
                 dt=1.0,
                 **kwargs
                 ):
        if memory_order < 0:
            memory_order = hidden_size

        self.dt = dt

        super().__init__(input_size, hidden_size, memory_size, memory_order, **kwargs)

    def update_memory(self, m, u, t0, t1):
        """
        m: (B, M, N) [batch, memory_size, memory_order]
        u: (B, M)
        t0: (B,) previous time
        t1: (B,) current time
        """

        dt = self.dt*(t1-t0).unsqueeze(-1)
        m = self.transition_fn(dt, m, u)
        return m
