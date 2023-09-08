import load_data
import argparse
import random
import torch
import numpy as np
import pandas as pd

import models
from base import BaseConfig
from models import LSTM
from torch_scatter import scatter
from torch_geometric.loader import RandomNodeLoader
from torch_geometric.data import Data
from tqdm import tqdm


class Config(BaseConfig):
    def __init__(self):
        super().__init__()
        # Reset base variables
        self.max_epochs = 1000
        self.early_stop_epochs = 15
        self.infer = False
        self.cuda_device = torch.device('cuda:0')
        self.batch_size = 1

        self.start_date = '2020-03-01'
        self.min_peak_size = -1
        self.forecast_date = '2021-02-14'
        self.lookback_days = 14
        self.horizon = 21
        self.val_days = 1
        self.sample_ratio = 0.8
        self.lookahead_days = 1

        # For MPNNLSTM
        self.mpnnlstm_num_group = 10
        self.mpnnlstm_hidden_size = 16
        self.mpnnlstm_window = 1
        self.mpnnlstm_dropout = .01
        self.mpnnlstm_in_channels = 1

        # For LSTM Abl
        self.abl_lstm_dim = 16
        self.abl_lstm_layer_num = 3

        # For Abl
        self.abl_type = 'mpnn_lstm'


class Task:
    def __init__(self, conf):
        self.config = conf
        print('Initialize {}'.format(self.__class__))

        # read df and convert them into numpy array
        self.policy_df = node['policy']
        self.geo_df = node['geo']
        self.confirm_df = node['confirm']
        self.death_df = node['death']
        self.population_df = node['population']
        self.gdp_df = node['gdp']
        self.dates = self.policy_df.columns.tolist()

        self.policy_ary = np.array(self.policy_df).astype('float32')
        self.geo_ary = np.array(self.geo_df)
        self.confirm_ary = np.array(self.confirm_df).astype('float32')
        self.death_ary = np.array(self.death_df).astype('float32')
        self.population_ary = np.array(self.population_df).astype('float32')
        self.gdp_ary = np.array(self.gdp_df)
        assert self.policy_ary.shape == self.confirm_ary.shape == self.death_ary.shape

        # Make Node Representation
        node_ary_list = [self.policy_ary, self.confirm_ary, self.death_ary]
        self.day_inputs = np.stack(node_ary_list, axis=2)
        self.day_outputs = np.expand_dims(self.confirm_ary, axis=-1)

        print('The input vector shape is ' + str(self.day_inputs.shape))
        self.config.num_nodes = self.day_inputs.shape[0]
        self.config.day_seq_len = self.day_inputs.shape[1]
        self.config.in_fea_dim = self.day_inputs.shape[2]
        self.config.out_fea_dim = self.day_outputs.shape[2]

        self.init_data()
        self.init_graph()

        self.loss_func = torch.nn.L1Loss()

    def set_random_seed(self, seed=None):
        if seed is None:
            seed = self.config.random_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def init_data(self):
        use_dates = [
            pd.to_datetime(item) for item in self.dates
            if pd.to_datetime(item) <= pd.to_datetime(self.config.forecast_date)
        ]
        test_div = len(use_dates) - 1
        val_div = test_div - self.config.horizon
        train_div = val_div - self.config.val_days

        print('Training division: ' + str(self.dates[train_div]) + ', Validation division: ' + str(self.dates[val_div])
              + ', Testing division: ' + self.dates[test_div])

        self.train_day_inputs = self.day_inputs[:, train_div-self.config.lookback_days:train_div, :]
        self.train_day_outputs = self.day_outputs[:, train_div+self.config.horizon, :]
        self.config.train_day_seq_len = self.train_day_inputs.shape[1]

        if self.config.infer:
            self.val_day_inputs = self.day_inputs[:, :train_div+1, :]
            self.val_day_outputs = self.day_outputs[:, :train_div+1, :]
        else:
            self.val_day_inputs = self.day_inputs[:, val_div: val_div+1, :]
            self.val_day_outputs = self.day_outputs[:, val_div: val_div+1, :]

        self.test_day_inputs = self.day_inputs[:, test_div-self.config.lookback_days: test_div, :]
        self.test_day_outputs = self.day_outputs[:, test_div+self.config.horizon, :]

    def init_graph(self):
        self.edge_index = graph['edge_index']
        self.edge_weight = graph['edge_weight']
        self.node_name = graph['node_name']
        self.node_type = graph['node_type']
        self.config.num_edges = self.edge_weight.shape[0]
        self.config.num_node_types = int(graph['node_type'].max()) + 1

        base_ones = torch.ones_like(self.node_type, dtype=torch.float)
        node_type_count = scatter(base_ones, self.node_type, dim_size=self.config.num_node_types, reduce='sum')
        self.node_weight = 1.0 / node_type_count * node_type_count.max()

    def train_LSTM(self):
        # This method is used for LSTM comparison
        train_data_ratio = 0.8  # Choose 80% of the data for training
        train_data_len = int(self.config.day_seq_len * train_data_ratio)
        train_x = self.train_day_inputs[:, :, 1]
        train_y = self.train_day_outputs

        test_x = self.test_day_inputs[:, :, 1]
        test_y = self.test_day_outputs

        train_x_tensor = train_x.reshape(-1, self.config.batch_size, 1)
        train_y_tensor = train_y.reshape(-1, self.config.batch_size, self.config.out_fea_dim)

        train_x_tensor = torch.from_numpy(train_x_tensor)
        train_y_tensor = torch.from_numpy(train_y_tensor)

        lstm_model = LSTM(input_size=1, hidden_size=self.config.abl_lstm_dim,
                          output_size=1, num_layers=self.config.abl_lstm_layer_num)
        lstm_model = lstm_model.to(device=self.config.cuda_device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-2)

        prev_loss = 1000
        max_epochs = 1000

        train_x_tensor = train_x_tensor.to(self.config.cuda_device)
        train_y_tensor = train_y_tensor.to(self.config.cuda_device)

        for epoch in range(max_epochs):
            output = lstm_model(train_x_tensor).to(self.config.cuda_device)
            loss = criterion(output, train_y_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss < prev_loss:
                torch.save(lstm_model.state_dict(), 'lstm_model.pt')  # save model parameters to files
                prev_loss = loss

            if loss.item() < 1e-10:
                print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, max_epochs, loss.item()))
                print("The loss value is reached")
                break
            elif (epoch + 1) % 100 == 0:
                print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch + 1, max_epochs, loss.item()))

        # prediction on training dataset
        pred_y_for_train = lstm_model(train_x_tensor).to(self.config.cuda_device)
        pred_y_for_train = pred_y_for_train.view(-1, self.config.in_fea_dim).data.cpu().numpy()

        # ----------------- test -------------------
        lstm_model = lstm_model.eval()  # switch to testing model

        # prediction on test dataset
        test_x_tensor = test_x.reshape(-1, self.config.batch_size, 1)
        test_x_tensor = torch.from_numpy(test_x_tensor)
        test_x_tensor = test_x_tensor.to(self.config.cuda_device)

        pred_y_for_test = lstm_model(test_x_tensor).to(self.config.cuda_device)
        pred_y_for_test = pred_y_for_test.view(-1, 1).data.cpu().numpy()

        pred_y_for_test = pred_y_for_test.squeeze()
        test_y = test_y.squeeze()
        loss = criterion(torch.from_numpy(pred_y_for_test), torch.from_numpy(test_y))
        print("test loss：", loss.item())

        exit(10)

    def train_MPNN_LSTM(self):
        self.net = models.MPNNLSTM(self.config)
        edge_index = self.edge_index
        edge_weight = self.edge_weight
        train_x = torch.from_numpy(self.train_day_inputs[:, :, 0])
        test_x = torch.from_numpy(self.test_day_inputs[:, :, 0])
        train_y = torch.from_numpy(self.train_day_outputs.squeeze())
        test_y = torch.from_numpy(self.test_day_outputs.squeeze())
        g_train = Data(edge_index=edge_index, edge_attr=edge_weight, x=train_x, y=train_y)
        g_test = Data(edge_index=edge_index, edge_attr=edge_weight, x=test_x, y=test_y)
        train_loader = RandomNodeLoader(g_train, num_parts=self.config.mpnnlstm_num_group,
                                         shuffle=True, drop_last=True)
        test_loader = RandomNodeLoader(g_test, num_parts=self.config.mpnnlstm_num_group, shuffle=True, drop_last=True)
        model = self.net.to(self.config.cuda_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        def train():
            model.train()
            mae_list = []
            for data in train_loader:
                data = data.to(self.config.cuda_device)
                optimizer.zero_grad()
                out = model(x=data.x, edge_idx=data.edge_index, edge_wgt=data.edge_attr)
                criterion = self.loss_func
                y_loss = criterion(out, data.y)
                y_loss.backward()
                optimizer.step()

                mae = torch.abs(out - data.y).mean()
                mae_list.append(mae.cpu().detach().numpy())
            mae = np.mean(mae_list)
            return mae

        def test():
            model.eval()
            mae_list = []
            for data in test_loader:
                data = data.to(self.config.cuda_device)
                out = model(x=data.x, edge_idx=data.edge_index, edge_wgt=data.edge_attr)

                mae = torch.abs(out).mean()
                mae_list.append(mae.cpu().detach().numpy())
            mae = np.mean(mae_list)
            return mae

        for epoch in range(0, 1000):
            loss = train()
            accs = test()
            print(f'Epoch: {epoch:02d}, Train: {loss:.4f}, Test_Outputs: {accs:.4f}')


class Wrapper(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.config = conf

        if self.config.abl_type == 'lstm':
            task.train_LSTM()
        elif self.config.abl_type == 'mpnn_lstm':
            task.train_MPNN_LSTM()

        else:
            raise ValueError('Unexpected Model Type')

    def forward(self, input_days, g):
        if self.config.abl_type == 'mpnn_lstm':
            _x = input_days
            edge_idx = g['edge_index']
            edge_wgt = g['edge_attr']
            out = self.net(_x, edge_idx, edge_wgt)
            return out
        else:
            raise ValueError('Unexpected Model Type')

    def get_net_parameters(self):
        return self.net.parameters()

    def get_graph_parameters(self):
        return self.edge_weight


if __name__ == '__main__':
    config = Config()
    parser = argparse.ArgumentParser(description='Batch Neural Network Experiments')

    node, graph = load_data.load_data()
    task = Task(config)
    task.set_random_seed()
    # Set random seed before the initialization of network parameters
    net = Wrapper(task.config)

