# Adapted from https://github.com/Lezcano/expRNN

import torch
import torch.nn as nn

from .parametrization import Parametrization


class Orthogonal(Parametrization):
    """ Class that implements optimization restricted to the Stiefel manifold """
    def __init__(self, input_size, output_size, initializer_skew, mode, param):
        """
        mode: "static" or a tuple such that:
                mode[0] == "dynamic"
                mode[1]: int, K, the number of steps after which we should change the basis of the dyn triv
                mode[2]: int, M, the number of changes of basis after which we should project back onto the manifold the basis. This is particularly helpful for small values of K.

        param: A parametrization of in terms of skew-symmetyric matrices
        """
        max_size = max(input_size, output_size)
        A = torch.empty(max_size, max_size)
        base = torch.empty(input_size, output_size)
        super(Orthogonal, self).__init__(A, base, mode)
        self.input_size = input_size
        self.output_size = output_size
        self.param = param
        self.init_A = initializer_skew
        self.init_base = nn.init.eye_

        self.reset_parameters()

    def reset_parameters(self):
        self.init_A(self.A)
        self.init_base(self.base)

    def forward(self, input):
        return input.matmul(self.B)

    def retraction(self, A, base):
        # This could be any parametrization of a tangent space
        A = A.triu(diagonal=1)
        A = A - A.t()
        B = base.mm(self.param(A))
        if self.input_size != self.output_size:
            B = B[:self.input_size, :self.output_size]
        return B

    def project(self, base):
        try:
            # Compute the projection using the thin SVD decomposition
            U, _, V = torch.svd(base, some=True)
            return U.mm(V.t())
        except RuntimeError:
            # If the svd does not converge, fallback to the (thin) QR decomposition
            x = base
            if base.size(0) < base.size(1):
                x = base.t()
            ret = torch.qr(x, some=True).Q
            if base.size(0) < base.size(1):
                ret = ret.t()
            return ret


class modrelu(nn.Module):
    def __init__(self, features):
        # For now we just support square layers
        super(modrelu, self).__init__()
        self.features = features
        self.b = nn.Parameter(torch.Tensor(self.features))
        self.reset_parameters()

    def reset_parameters(self):
        self.b.data.uniform_(-0.01, 0.01)

    def forward(self, inputs):
        norm = torch.abs(inputs)
        biased_norm = norm + self.b
        magnitude = nn.functional.relu(biased_norm)
        phase = torch.sign(inputs)

        return phase * magnitude


class OrthogonalRNN(nn.Module):
    def __init__(self, input_size, hidden_size, initializer_skew, mode, param):
        super(OrthogonalRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_kernel = Orthogonal(hidden_size, hidden_size, initializer_skew, mode, param=param)
        self.input_kernel = nn.Linear(in_features=self.input_size, out_features=self.hidden_size, bias=False)
        self.nonlinearity = modrelu(hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.input_kernel.weight.data, nonlinearity="relu")

    def default_hidden(self, input):
        return input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)

    def forward(self, input, hidden):
        input = self.input_kernel(input)
        hidden = self.recurrent_kernel(hidden)
        out = input + hidden
        out = self.nonlinearity(out)

        return out, out
