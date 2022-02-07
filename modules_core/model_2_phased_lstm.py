import torch
import math
import torch
from torch import nn, sin, pow
from torch.nn import Parameter
from torch.distributions.exponential import Exponential

def fmod(a, b):
    return ( b / math.pi ) * torch.arctan( torch.tan( math.pi * ( a / b - 0.5 ) ) ) + b / 2

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

DEVICE = 'cuda'
class PhasedLSTM(torch.nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            alpha=1e-3,
            tau_max=3.0,
            r_on=5e-2
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.alpha = alpha
        self.r_on = r_on

        stdv = 1.0 / math.sqrt(self.hidden_size)

        self.W = torch.nn.Parameter(
            torch.FloatTensor(4 * input_size, 4 * hidden_size).uniform_(-stdv, stdv) # Test -1..1, default 0..1
        )

        self.U = torch.nn.Parameter(
            torch.FloatTensor(4 * input_size, 4 * hidden_size).uniform_(-stdv, stdv) # Test -1..1, default 0..1
        )

        self.b = torch.nn.Parameter(
            torch.FloatTensor(4 * hidden_size).zero_()
        )
        self.b.data[self.hidden_size:self.hidden_size*2].fill_(1.0)

        self.w_peep = torch.nn.Parameter(
            torch.FloatTensor(3 * hidden_size).uniform_(-stdv, stdv)
        )

        self.tau = torch.nn.Parameter(
            torch.FloatTensor(hidden_size).uniform_(0, tau_max).exp_()
        )

        self.shift = torch.nn.Parameter(
            torch.FloatTensor(hidden_size).uniform_(0, torch.mean(self.tau).item())
        )

        self.h_0 = torch.nn.Parameter(
            torch.FloatTensor(hidden_size).uniform_(-stdv, stdv)
        )
        self.c_0 = torch.nn.Parameter(
            torch.FloatTensor(hidden_size).uniform_(-stdv, stdv)
        )

        self.batch_norm = torch.nn.BatchNorm1d(num_features=hidden_size*3)

    def forward(self, x, h_c=None):
        if h_c is None:
            h = torch.zeros((x.size(0), self.hidden_size)).to(DEVICE)
            c = torch.zeros((x.size(0), self.hidden_size)).to(DEVICE)
            h = self.h_0.expand(x.size(0), self.hidden_size)
            c = self.c_0.expand(x.size(0), self.hidden_size)
        else:
            h, c = h_c
        h_out = []
        # cuda dnn
        # x => (B, Seq, F)
        x_seq = x.permute(1, 0, 2) # (Seq, B, F)

        seq_len = x_seq.size(0)
        times = torch.arange(seq_len).unsqueeze(dim=1) # (Seq, 1)
        times = times.expand((seq_len, self.hidden_size)).to(DEVICE)
        phi = fmod((times - self.shift), self.tau) / (self.tau + 1e-8)

        alpha = self.alpha
        if not self.training: # model = model.eval() Dropout
            alpha = 0

        k = torch.where(
            phi < 0.5 * self.r_on,
            2.0 * phi / self.r_on,
            torch.where(
                torch.logical_and(0.5 * self.r_on <= phi, phi < self.r_on),
                2.0 - (2.0 * phi / self.r_on),
                alpha * phi
            )
        )

        for t, x_t in enumerate(x_seq):

            gates = (torch.matmul(
                self.W[:self.hidden_size*3, :self.hidden_size*3], # (in, out),
                x_t.repeat(1, 3).unsqueeze(dim=-1)
            ) + torch.matmul(
                self.U[:self.hidden_size*3, :self.hidden_size*3],
                h.repeat(1, 3).unsqueeze(dim=-1)
            )).squeeze() + self.b[:self.hidden_size*3] + self.w_peep * c.repeat(1, 3) # ? should this be c_t or/and c_{t-1}


            i_t = torch.sigmoid(gates[:, 0:self.hidden_size])
            f_t = torch.sigmoid(gates[:, self.hidden_size:self.hidden_size*2])
            o_t = torch.sigmoid(gates[:, self.hidden_size*2:self.hidden_size*3])

            gate_c = (torch.matmul(
                self.W[self.hidden_size*3:self.hidden_size*4, self.hidden_size*3:self.hidden_size*4],
                x_t.unsqueeze(dim=-1)
            ) + torch.matmul(
                self.U[self.hidden_size*3:self.hidden_size*4, self.hidden_size*3:self.hidden_size*4],
                h.unsqueeze(dim=-1)
            )).squeeze() +\
            self.b[self.hidden_size*3:self.hidden_size*4]

            c_prim = f_t * c + i_t * torch.tanh(gate_c)
            c = k[t] * c_prim + (1 - k[t]) * c
            h_prim = torch.tanh(c_prim) * o_t
            h = k[t] * h_prim + (1 - k[t]) * h
            h_out.append(h)
        t_h_out = torch.stack(h_out)
        t_h_out = t_h_out.permute(1, 0, 2) #  (B, Seq, F)
        return t_h_out


class Model(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.activation = torch.nn.Mish()
        if args.activation == 'snake':
            self.activation = Snake(in_features=args.hidden_size)

        self.linear_1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=4, out_features=args.hidden_size),
            torch.nn.LayerNorm(normalized_shape=args.hidden_size)
        )

        layers = []
        for _ in range(2):
            layers.append(PhasedLSTM(
                input_size=args.hidden_size,
                hidden_size=args.hidden_size,
            ))
        self.lstm = torch.nn.Sequential(*layers)

        if args.activation == 'maxout':
            self.linear_2 = Maxout(in_features=args.hidden_size, out_features=4, n_channels=args.maxout_layers)
        else:
            self.linear_2 = torch.nn.Sequential(
                torch.nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size),
                torch.nn.LayerNorm(normalized_shape=args.hidden_size),
                self.activation,
                torch.nn.Linear(in_features=args.hidden_size, out_features=4)
        )

    def forward(self, x):
        y_1 = self.linear_1.forward(x)
        lstm_out, _ = self.lstm.forward(y_1)
        y_prim = self.linear_2.forward(lstm_out)

        return y_prim
