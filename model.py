import torch
import torch_geometric.nn as PyG

from torch import nn
from nbeats import NBeatsEncoder
import torch_geometric_temporal as tgt


class LSTM(nn.Module):
    """
        Parameters:
            - input_size: feature size;
            - hidden_size: number of hidden units;
            - output_size: number of output;
            - num_layers: layers of LSTM to stack
    """
    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)    # 利用 torch.nn 构建 LSTM 模型
        self.linear = nn.Linear(hidden_size, output_size)   # 全连接层

    def forward(self, _x):
        x, _ = self.lstm(_x)    # _x is the input, size (seq_len, batch, input_size)
        s, b, h = x.shape   # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s*b, h)
        x = self.linear(x)
        x = x.view(s, b, -1)
        return x


class MPNNLSTM(tgt.nn.MPNNLSTM):
    def __init__(self, config):
        super().__init__()
        self.mpnn = tgt.nn.MPNNLSTM(in_channels=1, hidden_size=1, num_nodes=1, window=1, dropout=0.1)

    def forward(self):
        pass





