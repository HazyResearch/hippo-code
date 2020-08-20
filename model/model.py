import torch
import torch.nn as nn
from functools import partial

from model.rnn import RNN, RNNWrapper, LSTMWrapper
from model import rnncell, opcell # TODO: this is just to force cell_registry to update. There is probably a better programming pattern for this
from model.rnncell import CellBase
from model.orthogonalcell import OrthogonalCell

class Model(nn.Module):

    def __init__(
        self,
        input_size,
        output_size,
        output_len=0,
        cell='lstm',
        cell_args={},
        output_hiddens=[],
        embed_args=None,
        preprocess=None,
        ff=False,
        dropout=0.0,
        split=0,
    ):
        super(Model, self).__init__()

        # Save arguments needed for forward pass
        self.input_size = input_size
        self.output_size = output_size
        self.output_len = output_len
        assert output_len >= 0, f"output_len {output_len} should be 0 to return just the state or >0 to return the last output tokens"
        self.dropout = dropout
        self.split = split

        cell_args['input_size'] = input_size
        if embed_args is not None:
            self.embed_dim = embed_args['embed_dim']
            self.embedding = nn.Embedding(input_size, self.embed_dim)
            cell_args['input_size'] = self.embed_dim


        ### Handle optional Hippo preprocessing
        self.preprocess = preprocess
        if self.preprocess is not None:
            assert isinstance(self.preprocess, dict)
            assert 'order' in self.preprocess
            assert 'measure' in self.preprocess
            self.hippo = VariableMemoryProjection(**self.preprocess)
            cell_args['input_size'] *= (self.preprocess['order']+1) # will append this output to original channels

        ### Construct main RNN
        if ff: # feedforward model
            cell_args['input_size'] = input_size
            self.rnn = QRNN(**cell_args)
        else:
            # Initialize proper cell type
            if cell == 'lstm':
                self.rnn = LSTMWrapper(**cell_args, dropout=self.dropout)
            else:
                if cell in CellBase.registry:
                    cell_ctor = CellBase.registry[cell]
                elif cell == 'orthogonal':
                    cell_ctor = OrthogonalCell
                else:
                    assert False, f"cell {cell} not supported"

                self.rnn = RNN(cell_ctor(**cell_args), dropout=self.dropout)
                if self.split > 0:
                    self.initial_rnn = RNN(cell_ctor(**cell_args), dropout=self.dropout)


        ### Construct output head
        sizes = [self.rnn.output_size()] + output_hiddens + [output_size]
        self.output_mlp = nn.Sequential(*[nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)])


    # @profile
    def forward(self, inputs, len_batch=None):
        B, L, C = inputs.shape
        inputs = inputs.transpose(0, 1) # .unsqueeze(-1)  # (seq_length, batch, channels)

        # Apply Hippo preprocessing if necessary
        if self.preprocess is not None:
            p = self.hippo(inputs)
            p = p.reshape(L, B, self.input_size * self.preprocess['order'])
            inputs = torch.cat([inputs, p], dim=-1)

        # Handle embedding
        if hasattr(self, 'embedding'):
            inputs = self.embedding(inputs)
        if len_batch is not None:
            inputs = nn.utils.rnn.pack_padded_sequence(inputs, len_batch, enforce_sorted=False)

        # Option to have separate RNN for head of sequence, mostly for debugging gradients etc
        if self.split > 0:
            initial_inputs, inputs = inputs[:self.split], inputs[self.split:]
            _, initial_state = self.initial_rnn(initial_inputs, return_output=False)
        else:
            initial_state = None

        # Apply main RNN
        if self.output_len > 0:
            outputs, _ = self.rnn(inputs, init_state=initial_state, return_output=True)
            # get last output tokens
            outputs = outputs[-self.output_len:,:,:]
            outputs = outputs.transpose(0, 1)
            return self.output_mlp(outputs)
        else:
            _, state = self.rnn(inputs, init_state=initial_state, return_output=False)
            state = self.rnn.output(state)
            return self.output_mlp(state)

