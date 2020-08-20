import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_tuple(tup, fn):
    """Apply a function to a Tensor or a tuple of Tensor
    """
    if isinstance(tup, tuple):
        return tuple((fn(x) if isinstance(x, torch.Tensor) else x) for x in tup)
    else:
        return fn(tup)

def concat_tuple(tups, dim=0):
    """Concat a list of Tensors or a list of tuples of Tensor
    """
    if isinstance(tups[0], tuple):
        return tuple((torch.cat(xs, dim) if isinstance(xs[0], torch.Tensor) else xs[0]) for xs in zip(*tups))
    else:
        return torch.cat(tups, dim)


class RNN(nn.Module):

    def __init__(self, cell, dropout=0.0):
        super().__init__()
        self.cell = cell

        if dropout > 0.0:
            self.use_dropout = True
            self.drop_prob = dropout
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.use_dropout = False

    def forward(self, inputs, init_state=None, return_output=False):
        """
        cell.forward : (input, state) -> (output, state)
        inputs : [length, batch, dim]
        """
        # Similar implementation to https://github.com/pytorch/pytorch/blob/9e94e464535e768ad3444525aecd78893504811f/torch/nn/modules/rnn.py#L202
        is_packed = isinstance(inputs, nn.utils.rnn.PackedSequence)
        if is_packed:
            inputs, batch_sizes, sorted_indices, unsorted_indices = inputs
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = inputs.size(1)
            sorted_indices = None
            unsorted_indices = None
        # Construct initial state
        if init_state is None:
            state = self.cell.default_state(inputs[0], max_batch_size)
        else:
            state = apply_tuple(init_state, lambda x: x[sorted_indices] if sorted_indices is not None else x)
        # Construct recurrent dropout masks
        if self.use_dropout:
            input_dropout = self.dropout(torch.ones(max_batch_size, self.cell.input_size, device=inputs.device))
            recurrent_dropout = self.dropout(torch.ones(max_batch_size, self.cell.hidden_size, device=inputs.device))
            output_dropout = self.dropout(torch.ones(max_batch_size, self.output_size(), device=inputs.device))

        outputs = []
        if not is_packed:
            for input in torch.unbind(inputs, dim=0):
                if self.use_dropout:
                    ## Recurrent Dropout
                    input = input * input_dropout
                output, new_state = self.cell.forward(input, state)
                if self.use_dropout:
                    output = output * output_dropout
                    try:
                        state = (self.dropout(new_state[0]),) + new_state[1:] # TODO not general
                    except:
                        state = self.dropout(new_state)
                else:
                    state = new_state
                if return_output:
                    outputs.append(output)
            return torch.stack(outputs) if return_output else None, state
        else:
            # Following implementation at https://github.com/pytorch/pytorch/blob/9e94e464535e768ad3444525aecd78893504811f/aten/src/ATen/native/RNN.cpp#L621
            # Batch sizes is a sequence of decreasing lengths, which are offsets
            # into a 1D list of inputs. At every step we slice out batch_size elements,
            # and possibly account for the decrease in the batch size since the last step,
            # which requires us to slice the hidden state (since some sequences
            # are completed now). The sliced parts are also saved, because we will need
            # to return a tensor of final hidden state.
            batch_sizes_og = batch_sizes
            batch_sizes = batch_sizes.detach().cpu().numpy()
            input_offset = 0
            last_batch_size = batch_sizes[0]
            saved_states = []
            for batch_size in batch_sizes:
                step_input = inputs[input_offset:input_offset + batch_size]
                input_offset += batch_size
                dec = last_batch_size - batch_size
                if (dec > 0):
                    saved_state = apply_tuple(state, lambda x: x[batch_size:])
                    state = apply_tuple(state, lambda x: x[:batch_size])
                    saved_states.append(saved_state)
                last_batch_size = batch_size
                if self.use_dropout:
                    step_input = step_input * input_dropout[:batch_size]
                output, new_state = self.cell.forward(step_input, state)
                if self.use_dropout:
                    output = output * output_dropout[:batch_size]
                    try:
                        state = (self.dropout(new_state[0]),) + new_state[1:] # TODO not general
                    except:
                        state = self.dropout(new_state)
                else:
                    state = new_state
                if return_output:
                    outputs.append(output)
            saved_states.append(state)
            saved_states.reverse()
            state = concat_tuple(saved_states)
            state = apply_tuple(state, lambda x: x[unsorted_indices] if unsorted_indices is not None else x)
            if return_output:
                outputs = nn.utils.rnn.PackedSequence(torch.cat(outputs, dim=0), batch_sizes_og, sorted_indices, unsorted_indices)
            else:
                outputs = None
            return outputs, state

    def state_size(self):
        return self.cell.state_size()

    def output_size(self):
        return self.cell.output_size()

    def output(self, state):
        return self.cell.output(state)


class RNNWrapper(nn.RNN):

    def forward(self, inputs, h_0=None):
        output, h_n = super().forward(inputs, h_0)
        return output, h_n.squeeze(0)


class LSTMWrapper(nn.LSTM):

    # return_output is only here to absorb the argument, making the interface compatible with RNN
    def forward(self, inputs, return_output=None, init_state=None):
        # init_state is just to absorb the extra argument that can be passed into our custom RNNs. Replaces (h_0, c_0) argument of nn.LSTM
        output, (h_n, c_n) = super().forward(inputs, init_state)
        return output, (h_n.squeeze(0), c_n.squeeze(0))

    def state_size(self):
        return self.hidden_size

    def output_size(self):
        return self.hidden_size

    def output(self, state):
        return state[0]
