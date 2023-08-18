import torch
import torch_geometric.nn as PyG

from torch import nn
from nbeats import NBeatsEncoder


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


class BaseGNNNet(nn.Module):
    def __init__(self):
        super().__init__()

    def dataflow_forward(self, X, g):
        raise NotImplementedError

    def subgraph_forward(self, X, g):
        raise NotImplementedError

    def forward(self, X, g, **kwargs):
        if g['type'] == 'dataflow':
            return self.dataflow_forward(X, g, **kwargs)
        elif g['type'] == 'subgraph':
            return self.subgraph_forward(X, g, **kwargs)
        else:
            raise Exception('Unsupported graph type {}'.format(g['type']))


class GCNConv(PyG.MessagePassing):
    def __init__(self, gcn_in_dim, config, **kwargs):
        super().__init__(aggr=config.gcn_aggr, node_dim=-2, **kwargs)

        self.gcn_in_dim = gcn_in_dim
        self.gcn_node_dim = config.gcn_node_dim
        self.gcn_dim = config.gcn_dim

        self.fea_map = nn.Linear(self.gcn_in_dim, self.gcn_dim)

    def forward(self, x, edge_index, edge_weight):
        batch_size, num_nodes, _ = x.shape
        if edge_weight.shape[0] == x.shape[0]:
            num_edges = edge_weight.shape[1]
            edge_weight = edge_weight.reshape(batch_size, num_edges, 1)
        else:
            num_edges = edge_weight.shape[0]
            edge_weight = edge_weight.reshape(1, num_edges, 1).expand(batch_size, -1, -1)

        # Calculate type-aware node info
        if isinstance(x, torch.Tensor):
            x = self.fea_map(x)
        else:
            x = (self.fea_map(x[0]), self.fea_map(x[1]))

        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return x_j * edge_weight

    def update(self, aggr_out, x):
        if not isinstance(x, torch.Tensor):
            x = x[1]

        return x + aggr_out


class NodeGNN(BaseGNNNet):
    def __init__(self, gcn_in_dim, config):
        super().__init__()
        self.layer_num = config.gcn_layer_num
        assert self.layer_num >= 1

        if config.gcn_type == 'gcn':
            GCNClass = GCNConv
        else:
            raise Exception(f'Unsupported gcn_type {config.gcn_type}')

        convs = [GCNClass(gcn_in_dim, config)]
        for _ in range(self.layer_num-1):
            convs.append(GCNClass(config.gcn_dim, config))
        self.convs = nn.ModuleList(convs)

    def subgraph_forward(self, X, g, edge_weight=None):
        edge_index = g['edge_index']
        if edge_weight is None:
            edge_weight = g['edge_attr']

        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)

        return x


class GNNModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.rnn_type = config.rnn_type
        if self.rnn_type == 'nbeats':
            self.rnn = NBeatsEncoder(config)
            self.rnn_hid_dim = config.hidden_dim
        else:
            raise Exception(f'Unsupported rnn type {self.rnn_type}')
        self.edge_gnn = None
        self.node_gnn = NodeGNN(self.rnn_hid_dim, config)
        self.gnn_fc = nn.Linear(config.gcn_dim, config.lookahead_days)

    def forward(self, input_days, g):
        nb_out, self.y_t = self.rnn(input_days, g)
        rnn_out, _ = nb_out.max(dim=-1)

        gcn_out = self.node_gnn(rnn_out, g)
        self.y_g = self.gnn_fc(gcn_out)

        y = self.y_t + self.y_g
        return y




