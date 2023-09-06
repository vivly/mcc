import load_data
import argparse
import random
import torch
import numpy as np
import pandas as pd

import model
from base import BaseConfig
from model import LSTM
from torch_scatter import scatter


class Config(BaseConfig):
    def __init__(self):
        super().__init__()
        # Reset base variables
        self.max_epochs = 1000
        self.early_stop_epochs = 15
        self.infer = False
        self.cuda_device = torch.device('cuda:0')
        self.batch_size = 3

        self.start_date = '2020-03-01'
        self.min_peak_size = -1
        self.forecast_date = '2020-06-29'
        self.lookback_days = 14
        self.horizon = 3
        self.val_days = 1
        self.sample_ratio = 0.8
        self.lookahead_days = 1

        # For GCN
        self.gcn_dim = 64
        self.gcn_type = 'gcn'
        self.gcn_aggr = 'max'
        self.gcn_layer_num = 2
        self.gcn_node_dim = 4
        self.gcn_edge_dim = 4

        # For RNN
        self.rnn_dim = 16
        self.rnn_layer_num = 3
        self.date_emb_dim = 2
        self.block_size = 3
        self.hidden_dim = 32
        self.id_emb_dim = 8

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

        self.loss_func = torch.nn.MSELoss()

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

        self.train_day_inputs = self.day_inputs[:train_div+1]
        self.train_day_outputs = self.day_outputs[:train_div+1]
        self.train_dates = self.dates[:train_div+1]

        if self.config.infer:
            self.val_day_inputs = self.day_inputs[:train_div+1]
            self.val_day_outputs = self.day_outputs[:train_div+1]
            self.val_dates = self.dates[:train_div+1]
        else:
            self.val_day_inputs = self.day_inputs[val_div: val_div+1]
            self.val_day_outputs = self.day_outputs[val_div: val_div+1]
            self.val_dates = self.dates[val_div: val_div+1]

        self.test_day_inputs = self.day_inputs[test_div: test_div+1]
        self.test_day_outputs = self.day_outputs[test_div: test_div + 1]
        self.test_dates = self.dates[test_div: test_div+1]

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

        lstm_model = LSTM(input_size=1, hidden_size=self.config.rnn_dim,
                          output_size=1, num_layers=self.config.rnn_layer_num)
        lstm_model = lstm_model.to(device=self.config.cuda_device)
        criterion = torch.nn.L1Loss()
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


class Wrapper(torch.nn.Module):
    def __init__(self, conf, edge_weight):
        super().__init__()
        self.config = conf

        if conf.abl_type == 'lstm':
            task.train_LSTM()
        elif conf.abl_type == 'mpnn_lstm':
            self.net = model.MPNNLSTM(conf)
        else:
            raise ValueError('Not Expected Model Type')

    def forward(self, input_days, g):
        return self.net(input_days, g)

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
    net = Wrapper(task.config, task.edge_weight)

