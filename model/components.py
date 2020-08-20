from functools import partial
import torch
import torch.nn as nn

from model.exprnn.orthogonal import modrelu

def get_activation(activation, size):
    if activation == 'id':
        return nn.Identity()
    elif activation == 'tanh':
        return torch.tanh
    elif activation == 'relu':
        return torch.relu
    elif activation == 'sigmoid':
        return torch.sigmoid
    elif activation == 'modrelu':
        return Modrelu(size)
    else:
        raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))


def get_initializer(name, activation):
    if activation in ['id', 'identity', 'linear', 'modrelu']:
        nonlinearity = 'linear'
    elif activation in ['relu', 'tanh', 'sigmoid']:
        nonlinearity = activation
    else:
        assert False, f"get_initializer: activation {activation} not supported"
    if name == 'uniform':
        initializer = partial(torch.nn.init.kaiming_uniform_, nonlinearity=nonlinearity)
    elif name == 'normal':
        initializer = partial(torch.nn.init.kaiming_normal_, nonlinearity=nonlinearity)
    elif name == 'xavier':
        initializer = torch.nn.init.xavier_normal_
    elif name == 'zero':
        initializer = partial(torch.nn.init.constant_, val=0)
    elif name == 'one':
        initializer = partial(torch.nn.init.constant_, val=1)
    else:
        assert False, f"get_initializer: initializer type {name} not supported"

    return initializer



class Modrelu(modrelu):
    def reset_parameters(self):
        self.b.data.uniform_(-0.0, 0.0)


def Linear_(input_size, output_size, bias, init='normal', zero_bias_init=False, **kwargs):
    """ Returns a nn.Linear module with initialization options """
    l = nn.Linear(input_size, output_size, bias=bias, **kwargs)
    get_initializer(init, 'linear')(l.weight)
    if bias and zero_bias_init:
        nn.init.zeros_(l.bias)
    return l


class Gate(nn.Module):
    """ Implements gating mechanisms.

    Mechanisms:
    N  - No gate
    G  - Standard sigmoid gate
    """
    def __init__(self, size, preact_ctor, preact_args, mechanism='N'):
        super().__init__()
        self.size      = size
        self.mechanism = mechanism

        if self.mechanism == 'N':
            pass
        elif self.mechanism == 'G':
            self.W_g = preact_ctor(*preact_args)
        else:
            assert False, f'Gating type {self.mechanism} is not supported.'

    def forward(self, *inputs):
        if self.mechanism == 'N':
            return 1.0

        if self.mechanism == 'G':
            g_preact = self.W_g(*inputs)
            g = torch.sigmoid(g_preact)
        return g
