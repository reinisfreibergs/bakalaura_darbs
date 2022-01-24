import torch
import numpy as np
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(self.layer_norm_i.forward(gates[:, :HS])), # input
                torch.sigmoid(self.layer_norm_f.forward(gates[:, HS:HS*2])), # forget
                torch.tanh(self.layer_norm_g.forward(gates[:, HS*2:HS*3])),
                torch.sigmoid(self.layer_norm_o.forward(gates[:, HS*3:])), # output
            )
class Model(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.linear_1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=4, out_features=args.hidden_size),
            torch.nn.LayerNorm(normalized_shape=args.hidden_size)
        )
        self.lstm_layer = torch.nn.LSTM(input_size=args.hidden_size, hidden_size=args.hidden_size, batch_first=True, num_layers=args.lstm_layers)
        self.linear_2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size),
            torch.nn.LayerNorm(normalized_shape=args.hidden_size),
            torch.nn.Mish(),
            torch.nn.Linear(in_features=args.hidden_size, out_features=4)
        )
    def forward(self, x):
        y_1 = self.linear_1.forward(x)
        lstm_out, _ = self.lstm_layer.forward(y_1)
        y_prim = self.linear_2.forward(lstm_out)

        return y_prim
