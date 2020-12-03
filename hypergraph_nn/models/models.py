import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

class GNNStack(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args):
        super(GNNStack, self).__init__()
        conv_model = self.get_model(args.model_name)
        self.convs_layers = nn.ModuleList()
        self.convs_layers.append(conv_model(input_dim, hidden_dim))
        for layer in range(args.num_layers-1):
            self.convs_layers.append(conv_model(hidden_dim, hidden_dim))

        self.final_lin_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(args.dropout), 
            nn.Linear(hidden_dim, output_dim)
        )

        self.task = args.task
        self.dropout = args.dropout
        self.num_layers = args.num_layers

    def get_model(self, model_name):
        if model_name == 'GCN':
            return pyg_nn.GCNConv
        elif model_name == 'GraphSage':
            return pyg_nn.SAGEConv
        elif model_name == 'GAT':
            return pyg_nn.GATConv
        elif model_name == 'Hypergraph':
            pyg_nn.HypergraphConv
        else:
            raise RuntimeError('Unknown convolutional model.')

    def forward(self, data):
        if self.task == 'node':
            x, edge_index = data.x, data.edge_index
        else:
            x, edge_index = data.x, data.train_pos_edge_index

        for layer in range(self.num_layers):
            x = self.convs_layers[layer](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.final_lin_layers(x)

        if self.task == 'node':
            out = F.log_softmax(x, dim=1)
        elif self.task == 'link':
            out = x
        else:
            raise RuntimeError('Unknown task.')

        return out

    def loss(self, pred, label):

        if self.task == 'node':
            return F.nll_loss(pred, label)
        else:
            pass
