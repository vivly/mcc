import torch
import torch_geometric.nn as PyG
from torch import nn
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


class MPNNLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_channels = config.train_day_seq_len
        self.hidden_size = config.mpnnlstm_hidden_size
        self.num_nodes = int(config.num_nodes / config.mpnnlstm_num_group) + 1
        self.window = config.mpnnlstm_window
        self.dropout = config.mpnnlstm_dropout
        self.mpnnlstm = tgt.nn.MPNNLSTM(in_channels=self.in_channels, hidden_size=self.hidden_size,
                                         num_nodes=self.num_nodes, window=self.window, dropout=self.dropout)
        self.linear = nn.Linear(self.in_channels + 2 * self.hidden_size, 1)

    def forward(self, x, edge_idx, edge_wgt):
        H = self.mpnnlstm(X=x, edge_index=edge_idx, edge_weight=edge_wgt)
        h = self.linear(H).squeeze()
        return h





