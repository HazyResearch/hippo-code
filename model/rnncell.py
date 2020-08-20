""" Baseline RNN cells such as the vanilla RNN and GRU. """

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.components import Gate, Linear_, Modrelu, get_activation, get_initializer
from model.orthogonalcell import OrthogonalLinear


class CellBase(nn.Module):
    """ Abstract class for our recurrent cell interface.

    Passes input through
    """
    registry = {}

    # https://www.python.org/dev/peps/pep-0487/#subclass-registration
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Only register classes with @name attribute
        if hasattr(cls, 'name') and cls.name is not None:
            cls.registry[cls.name] = cls

    name = 'id'
    valid_keys = []

    def default_initializers(self):
        return {}

    def default_architecture(self):
        return {}

    def __init__(self, input_size, hidden_size, initializers=None, architecture=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.architecture = self.default_architecture()
        self.initializers = self.default_initializers()
        if initializers is not None:
            self.initializers.update(initializers)
            print("Initializers:", initializers)
        if architecture is not None:
            self.architecture.update(architecture)

        assert set(self.initializers.keys()).issubset(self.valid_keys)
        assert set(self.architecture.keys()).issubset(self.valid_keys)

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, input, hidden):
        return input, input

    def default_state(self, input, batch_size=None):
        return input.new_zeros(input.size(0) if batch_size is None else batch_size,
                               self.hidden_size, requires_grad=False)

    def output(self, h):
        return h

    def state_size(self):
        return self.hidden_size

    def output_size(self):
        return self.hidden_size

    def initial_state(self, trainable=False):
        """ Return initial state of the RNN
        This should not need to see the input as it should be batch size agnostic and automatically broadcasted

        # TODO Currently not used
        """
        if trainable:
            self.initial_state = torch.zeros(self.hidden_size, requires_grad=True)
        else:
            return torch.zeros(self.hidden_size, requires_grad=True)



class RNNCell(CellBase):
    name = 'rnn'

    valid_keys = ['hx', 'hh', 'bias']

    def default_initializers(self):
        return {
            'hx': 'xavier',
            'hh': 'xavier',
            }

    def default_architecture(self):
        return {
            'bias': True,
            }


    def __init__(self, input_size, hidden_size,
                 hidden_activation='tanh',
                 orthogonal=False,
                 ortho_args=None,
                 zero_bias_init=False,
                 **kwargs
                 ):

        self.hidden_activation = hidden_activation
        self.orthogonal = orthogonal
        self.ortho_args = ortho_args
        self.zero_bias_init=zero_bias_init

        super().__init__(input_size, hidden_size,
                **kwargs,
                )

    def reset_parameters(self):
        self.W_hx = Linear_(self.input_size, self.hidden_size, bias=self.architecture['bias'], zero_bias_init=self.zero_bias_init)
        get_initializer(self.initializers['hx'], self.hidden_activation)(self.W_hx.weight)
        self.hidden_activation_fn = get_activation(self.hidden_activation, self.hidden_size)

        self.reset_hidden_to_hidden()

    def reset_hidden_to_hidden(self):
        if self.orthogonal:

            if self.ortho_args is None:
                self.ortho_args = {}
            self.ortho_args['input_size'] = self.hidden_size
            self.ortho_args['output_size'] = self.hidden_size

            self.W_hh = OrthogonalLinear(**self.ortho_args)
        else:
            self.W_hh = nn.Linear(self.hidden_size, self.hidden_size, bias=self.architecture['bias'])
            get_initializer(self.initializers['hh'], self.hidden_activation)(self.W_hh.weight)

    def forward(self, input, h):
        ### Update hidden state
        hidden_preact = self.W_hx(input) + self.W_hh(h)
        hidden = self.hidden_activation_fn(hidden_preact)

        return hidden, hidden

class GatedRNNCell(RNNCell):
    name = 'gru'

    def __init__(self, input_size, hidden_size,
                 gate='G', # 'N' | 'G'
                 reset='N',
                 **kwargs
                 ):
        self.gate  = gate
        self.reset = reset
        super().__init__(input_size, hidden_size, **kwargs)

    def reset_parameters(self):
        super().reset_parameters()

        preact_ctor = Linear_
        preact_args = [self.input_size + self.hidden_size, self.hidden_size, self.architecture['bias']]
        self.W_g     = Gate(self.hidden_size, preact_ctor, preact_args, mechanism=self.gate)
        self.W_reset = Gate(self.hidden_size, preact_ctor, preact_args, mechanism=self.reset)

    def forward(self, input, h):
        hx = torch.cat((input, h), dim=-1)
        reset = self.W_reset(hx)

        _, update = super().forward(input, reset*h)

        g = self.W_g(hx)
        h = (1.-g) * h + g * update

        return h, h

class MinimalRNNCell(CellBase):
    name = 'mrnn'

    valid_keys = ['hx', 'bias']

    def default_initializers(self):
        return {
            'hx': 'xavier',
            }

    def default_architecture(self):
        return {
            'bias': True,
            }


    def __init__(self, input_size, hidden_size,
                 hidden_activation='tanh',
                 orthogonal=False,
                 ortho_args=None,
                 zero_bias_init=False,
                 **kwargs
                 ):

        self.hidden_activation = hidden_activation
        self.zero_bias_init=zero_bias_init

        super().__init__(input_size, hidden_size,
                **kwargs,
                )

    def reset_parameters(self):
        self.W_hx = Linear_(self.input_size, self.hidden_size, bias=self.architecture['bias'], zero_bias_init=self.zero_bias_init)
        get_initializer(self.initializers['hx'], self.hidden_activation)(self.W_hx.weight)
        self.hidden_activation_fn = get_activation(self.hidden_activation, self.hidden_size)

        preact_ctor = Linear_
        preact_args = [self.input_size + self.hidden_size, self.hidden_size, self.architecture['bias']]
        self.W_g  = Gate(self.hidden_size, preact_ctor, preact_args, mechanism='G')


    def forward(self, input, h):
        ### Update hidden state
        hidden_preact = self.W_hx(input)
        hidden = self.hidden_activation_fn(hidden_preact)
        hx = torch.cat((input, h), dim=-1)
        g = self.W_g(hx)
        h = (1.-g) * h + g * hidden

        return h, h


class GatedSRNNCell(GatedRNNCell):
    name = 'grus'

    def __init__(self, input_size, hidden_size,
                 **kwargs
                 ):
        super().__init__(input_size, hidden_size, **kwargs)

    def reset_parameters(self):
        super().reset_parameters()

    def forward(self, input, hidden):
        hidden, t = hidden

        hx = torch.cat((input, hidden), dim=-1)
        reset = self.W_reset(hx)

        _, update = super().forward(input, reset*hidden)

        g = self.W_g(hx)
        g = g * 1. / (t+1)
        h = (1.-g) * hidden + g * update

        return h, (h, t+1)

    def default_state(self, input, batch_size=None):
        batch_size = input.size(0) if batch_size is None else batch_size
        return (input.new_zeros(batch_size, self.hidden_size, requires_grad=False),
                0)

    def output(self, state):
        """ Converts a state into a single output (tensor) """
        h, t = state

        return h

class ExpRNNCell(RNNCell):
    """ Note: there is a subtle distinction between this and the ExpRNN original cell (now implemented as orthogonalcell.OrthogonalCell) in the initialization of hx, but this shouldn't matter """
    name = 'exprnn'

    def __init__(self, input_size, hidden_size, **kwargs):
        super().__init__(input_size, hidden_size, orthogonal=True, hidden_activation='modrelu', **kwargs)
