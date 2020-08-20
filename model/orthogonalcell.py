import torch
import torch.nn as nn

from model.exprnn.orthogonal import Orthogonal
from model.exprnn.trivializations import expm, cayley_map
from model.exprnn.initialization import henaff_init_, cayley_init_

from model.components import Modrelu

param_name_to_param = {'cayley': cayley_map, 'expm': expm}
init_name_to_init = {'henaff': henaff_init_, 'cayley': cayley_init_}


class OrthogonalLinear(Orthogonal):
    def __init__(self, input_size, output_size, method='exprnn', init='cayley', K=100):
        """ Wrapper around expRNN's Orthogonal class taking care of parameter names """
        if method == "exprnn":
            mode = "static"
            param = 'expm'
        elif method == "dtriv":
            # We use 100 as the default to project back to the manifold.
            # This parameter does not really affect the convergence of the algorithms, even for K=1
            mode = ("dynamic", ortho_args['K'], 100) # TODO maybe K=30? check exprnn codebase
            param = 'expm'
        elif method == "cayley":
            mode = "static"
            param = 'cayley'
        else:
            assert False, f"OrthogonalLinear: orthogonal method {method} not supported"
        
        param = param_name_to_param[param]
        init_A = init_name_to_init[init]
        super().__init__(input_size, output_size, init_A, mode, param)

class OrthogonalCell(nn.Module):
    """ Replacement for expRNN's OrthogonalRNN class

        initializer_skew (str): either 'henaff' or 'cayley'
        param (str): A parametrization of in terms of skew-symmetyric matrices, either 'cayley' or 'expm'
    """
    def __init__(self, input_size, hidden_size, **ortho_args):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_kernel = OrthogonalLinear(hidden_size, hidden_size, **ortho_args)
        self.input_kernel = nn.Linear(in_features=self.input_size, out_features=self.hidden_size, bias=False)
        self.nonlinearity = Modrelu(hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.input_kernel.weight.data, nonlinearity="relu")

    def forward(self, input, hidden):
        input = self.input_kernel(input)
        hidden = self.recurrent_kernel(hidden)
        out = input + hidden
        out = self.nonlinearity(out)

        return out, out

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
