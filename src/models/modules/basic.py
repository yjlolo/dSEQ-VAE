from abc import abstractmethod, ABC
from typing import Literal, List, Optional, Union

import torch
from torch import nn
import torch.nn.init as weight_init
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from src.utils.seq import pad_and_reverse


class BaseRNN(nn.Module, ABC):
    def __init__(
        self,
        rnn_type: str,
        input_dim: int,
        rnn_dim: int,
        bidirectional: bool,
        n_layer: int,
        drop_rate: float,
        orthogonal_init: bool
    ):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.input_dim = input_dim
        self.rnn_dim = rnn_dim
        self.bidirectional = bidirectional
        self.n_layer = n_layer
        self.drop_rate = drop_rate
        self.orthogonal_init = orthogonal_init

        if self.rnn_type == 'rnn':
            self.rnn = nn.RNN(
                input_size=input_dim,
                hidden_size=rnn_dim,
                batch_first=True,
                bidirectional=bidirectional,
                num_layers=n_layer,
                dropout=drop_rate
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=rnn_dim,
                batch_first=True,
                num_layers=n_layer,
                dropout=drop_rate
            )
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=rnn_dim,
                batch_first=True,
                bidirectional=bidirectional,
                num_layers=n_layer,
                dropout=drop_rate
            )
        else:
            raise ValueError(
                f"rnn_type {rnn_type} must instead be \
                  ['rnn', 'gru', 'lstm']."
            )

        if orthogonal_init:
            self.init_weights()

    def init_weights(self):
        for w in self.rnn.parameters():
            if w.dim() > 1:
                weight_init.orthogonal_(w)

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass


class RnnEncoder(BaseRNN):
    def __init__(
        self,
        input_dim: int,
        rnn_dim: int,
        rnn_type: str = 'lstm',
        bidirectional: bool = False,
        n_layer: int = 1,
        drop_rate: float = 0.0,
        orthogonal_init: bool = False,
        pool_strategy: Optional[str] = None,
        reverse_input: bool = False
    ):
        super().__init__(
            rnn_type, input_dim, rnn_dim, bidirectional, n_layer, drop_rate,
            orthogonal_init
        )
        if pool_strategy is not None:
            assert pool_strategy.lower() in ['avg', 'last'], \
                f"`pool_strategy` ({pool_strategy}) should be either \
                    'avg' or 'last'."
        self.pool_strategy = pool_strategy
        self.reverse_input = reverse_input

    def forward(self, x, seq_lengths, mask=None, h=None, c=None):
        batch_size = x.size(0)
        total_length = x.size(1)
        x = pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True)

        self.rnn.flatten_parameters()
        if self.rnn_type in ['rnn', 'gru']:
            if h is None:
                _out, h = self.rnn(x)
            else:
                _out, h = self.rnn(x, h)
        else:
            if h is None:
                _out, (h, c) = self.rnn(x)
            else:
                _out, (h, c) = self.rnn(x, (h, c))

        if self.reverse_input:
            out = pad_and_reverse(_out, seq_lengths, total_length)
        else:
            out, _ = pad_packed_sequence(
                _out, batch_first=True, total_length=total_length
            )

        if self.pool_strategy is not None:
            if mask is not None:
                effect_len = mask.sum(dim=1, keepdim=True)
                out = out * mask.unsqueeze(-1)
            else:
                effect_len = total_length * \
                    torch.ones([batch_size, 1], device=out.device)

            if self.pool_strategy == 'avg':
                out = out.sum(dim=1).div(effect_len)
            elif self.pool_strategy == 'last':
                effect_len = \
                    effect_len.type(torch.LongTensor).to(effect_len.device)
                effect_len = effect_len.squeeze(-1)
                out = out[torch.arange(batch_size), effect_len - 1]

            if self.bidirectional:
                out = 0.5 * (out[:, :self.rnn_dim] + out[:, self.rnn_dim:])
        else:
            if self.bidirectional:
                out = 0.5 * (
                    out[:, :, :self.rnn_dim] + out[:, :, self.rnn_dim:]
                )
        
        if self.rnn_type in ['rnn', 'gru']:
            return out, h, None
        
        return out, h, c


class LinearLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        layer_dims: Union[int, List[int]],
        nonlin: Literal['tanh', 'relu', 'sigmoid'] = 'tanh',
        avg_pool: bool = False,
    ):
        super().__init__()
        if isinstance(layer_dims, int):
            layer_dims = [layer_dims]

        nonlin = nonlin.lower()
        if nonlin == 'tanh':
            nonlin = nn.Tanh()       
        elif nonlin == 'relu':
            nonlin = nn.ReLU()
        elif nonlin == 'sigmoid':
            nonlin = nn.Sigmoid()

        self.input_dim = input_dim
        self.layer_dims = layer_dims
        self.nonlin = nonlin

        dims = [input_dim] + layer_dims
        layers = []
        for l in range(len(dims) - 1):
            layers.append(nn.Linear(dims[l], dims[l + 1]))
            layers.append(self.nonlin)
        self.layers = nn.Sequential(*layers)
        self.avg_pool = avg_pool

    def forward(self, x, mask=None):
        out = self.layers(x)
        if self.avg_pool:
            assert len(out.size()) == 3
            if mask is not None:
                effect_len = mask.sum(dim=1, keepdim=True)
                out = out * mask.unsqueeze(-1)
            else:
                effect_len = x.size(1) * \
                    torch.ones([x.size(0), 1], device=out.device)

            return out.sum(dim=1).div(effect_len)
        return out
