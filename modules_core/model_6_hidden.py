import torch
import numpy as np
import math
import torch
from torch import nn, sin, pow
from torch.nn import Parameter
from torch.distributions.exponential import Exponential


class Snake(nn.Module):
    def __init__(self, in_features, a=None, trainable=True):
        super(Snake,self).__init__()
        self.in_features = in_features if isinstance(in_features, list) else [in_features]

        # Initialize `a`
        if a is not None:
            self.a = Parameter(torch.ones(self.in_features) * a) # create a tensor out of alpha
        else:
            m = Exponential(torch.tensor([0.1]))
            self.a = Parameter((m.rsample(self.in_features)).squeeze()) # random init = mix of frequencies

        self.a.requiresGrad = trainable # set the training of `a` to true

    def forward(self, x):
        return  x + (1.0/self.a) * pow(sin(x * self.a), 2)

class Maxout(torch.nn.Module):
    """Class Maxout implements maxout unit introduced in paper by Goodfellow et al, 2013.

    :param in_feature: Size of each input sample.
    :param out_feature: Size of each output sample.
    :param n_channels: The number of linear pieces used to make each maxout unit.
    :param bias: If set to False, the layer will not learn an additive bias.
    """
    def __init__(self, in_features, out_features, n_channels, bias=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.n_channels = n_channels
        self.weight = torch.nn.Parameter(torch.Tensor(n_channels * out_features, in_features))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(n_channels * out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        irange = 0.005
        torch.nn.init.uniform_(self.weight, -irange, irange)
        if self.bias is not None:
            torch.nn.init.uniform_(self.bias, -irange, irange)

    def forward(self, input):
        a = torch.nn.functional.linear(input, self.weight, self.bias)
        b = torch.nn.functional.max_pool1d(a, kernel_size=self.n_channels)
        return b.squeeze()



class Model(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        stdv = 1.0 / math.sqrt(args.hidden_size)

        self.h_0 = torch.nn.Parameter(
            torch.ones(size=(args.lstm_layers, args.batch_size, args.hidden_size)).uniform_(-stdv, stdv)
        )
        self.c_0 = torch.nn.Parameter(
            torch.ones(size=(args.lstm_layers, args.batch_size, args.hidden_size)).uniform_(-stdv, stdv)
        )

        self.activation = torch.nn.Mish()
        if args.activation == 'snake':
            self.activation = Snake(in_features=args.hidden_size)

        self.linear_1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=2, out_features=args.hidden_size),
            torch.nn.LayerNorm(normalized_shape=args.hidden_size)
        )
        self.lstm_layer = torch.nn.LSTM(input_size=args.hidden_size, hidden_size=args.hidden_size, batch_first=True, num_layers=args.lstm_layers)

        if args.activation == 'maxout':
            self.linear_2 = Maxout(in_features=args.hidden_size, out_features=2, n_channels=args.maxout_layers)
        else:
            self.linear_2 = torch.nn.Sequential(
                torch.nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size),
                torch.nn.LayerNorm(normalized_shape=args.hidden_size),
                self.activation,
                torch.nn.Linear(in_features=args.hidden_size, out_features=2)
        )

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = (self.h_0, self.c_0)

        y_1 = self.linear_1.forward(x)
        lstm_out, hidden = self.lstm_layer.forward(y_1, hidden)
        y_prim = self.linear_2.forward(lstm_out)

        return y_prim, hidden
