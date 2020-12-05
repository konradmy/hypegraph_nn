import argparse
from datetime import datetime
import time

import networkx as nx
import numpy as np
import torch
import torch.optim as optim

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GAE
from torch_geometric.utils import train_test_split_edges

import hypergraph_nn.models.models as models
import hypergraph_nn.datasets.biograkn.biograkn_dataset as biograkn_dataset
import os.path as osp

from tensorboardX import SummaryWriter
from sklearn.metrics import average_precision_score

def arg_parse():
    parser = argparse.ArgumentParser(description='GNN arguments.')
    parser.add_argument_group()
    parser.add_argument('--opt', dest='opt', type=str,
                        help='Type of optimizer', default='adam')
    parser.add_argument('--opt-scheduler', dest='opt_scheduler', type=str,
                        help='Type of optimizer scheduler. By default none', default='none')
    parser.add_argument('--opt-restart', dest='opt_restart', type=int,
                        help='Number of epochs before restart (by default set to 0 which means no restart)')
    parser.add_argument('--opt-decay-step', dest='opt_decay_step', type=int,
                        help='Number of epochs before decay')
    parser.add_argument('--opt-decay-rate', dest='opt_decay_rate', type=float,
                        help='Learning rate decay ratio')
    parser.add_argument('--lr', dest='lr', type=float,
                        help='Learning rate.', default=0.01)
    parser.add_argument('--clip', dest='clip', type=float,
                        help='Gradient clipping.')
    parser.add_argument('--weight_decay', type=float,
                        help='Optimizer weight decay.', default=0.0)
    parser.add_argument('--model_name', type=str,
                        help='Type of GNN model.', default='GCN')
    parser.add_argument('--batch_size', type=int,
                        help='Training batch size', default=32)
    parser.add_argument('--num_layers', type=int,
                        help='Number of graph conv layers', default=2)
    parser.add_argument('--hidden_dim', type=int,
                        help='Training hidden size', default=32)
    parser.add_argument('--dropout', type=float,
                        help='Dropout rate', default=0.0)
    parser.add_argument('--epochs', type=int,
                        help='Number of training epochs', default=200)
    parser.add_argument('--dataset', type=str,
                        help='Dataset', default='biograkn')
    parser.add_argument('--task', type=str,
                        help='Type of GNN task', default='node')
    parser.add_argument('--graph_type', type=str,
                        help='Type of data representation', default='casual_graph')
    parser.add_argument('--metrics_for_labels', type=str,
                        help='Print metrics for each label separately', default='False')
    parser.add_argument('--features', type=str,
                        help='Node features which should be used', default='ones')

    return parser.parse_args()

def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    return scheduler, optimizer

def train(dataset, args, writer = None):
    task = args.task
    test_loader = loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    if task == 'link':
        model = GAE(models.GNNStack(dataset.num_node_features, args.hidden_dim, int(dataset.num_classes), 
                            args))
    elif task == 'node':
        model = models.GNNStack(dataset.num_node_features, args.hidden_dim, int(dataset.num_classes), 
                            args)
    else:
        raise RuntimeError("Unknown task.")
    metrics_for_labels = True if args.metrics_for_labels == 'True' else False
    scheduler, opt = build_optimizer(args, model.parameters())
    print("Training \nModel: {}, Data representation: {}. Dataset: {}, Task type: {}". format(args.model_name, args.graph_type, args.dataset, args.task))
    metric_text = 'test accuracy' if task == 'node' else 'test precision'
    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        for batch in loader:
            opt.zero_grad()
            if task == 'node':
                pred = model(batch)
                label = batch.y
                pred = pred[batch.train_mask]
                label = label[batch.train_mask]
                loss = model.loss(pred, label)
            else:
                train_pos_edge_index = batch.train_pos_edge_index
                z = model.encode(batch)
                loss = model.recon_loss(z, train_pos_edge_index)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)
        if writer == None:
            print(total_loss)
        else:
            writer.add_scalar("loss", total_loss, epoch)

        if epoch % 10 == 0:
            test_acc, _ = test(loader, model, task = task)
            if writer == None:
                print(test_acc, metric_text)
            else:
                writer.add_scalar(metric_text, test_acc, epoch)
        if metrics_for_labels == True and task == 'link' and epoch == args.epochs -1:
            _, labels_metrics = test(loader, model, task = task, metrics_for_labels=metrics_for_labels)
            print('{} for labels:\n {}'.format(metric_text, labels_metrics))

def test(loader, model, task, is_validation=False, metrics_for_labels = False):
    model.eval()
    if task == 'node':
        correct = 0
        for data in loader:
            with torch.no_grad():
                pred = model(data).max(dim=1)[1]
                label = data.y
            
            mask = data.val_mask if is_validation else data.test_mask
            pred = pred[mask]
            label = data.y[mask]
            correct += pred.eq(label).sum().item()
        
        total = 0
        for data in loader.dataset:
            total += torch.sum(data.test_mask).item()
        return correct / total, None
    else:
        test_prec = []
        y = []
        pred = []
        labels = []
        for data in loader:
            with torch.no_grad():
                z = model.encode(data)
                test_prec.append(model.test(z, data.test_pos_edge_index, data.test_neg_edge_index)[1].item())
                pos_y = z.new_ones(data.test_pos_edge_index.size(1))
                neg_y = z.new_zeros(data.test_neg_edge_index.size(1))
                y.append(pos_y)
                y.append(neg_y)
                pos_pred = model.decoder(z, data.test_pos_edge_index, sigmoid=True)
                neg_pred = model.decoder(z, data.test_neg_edge_index, sigmoid=True)
                pred.append(pos_pred)
                pred.append(neg_pred)
                if metrics_for_labels:
                    labels_temp = get_labels_for_nodes(data.y, data.test_pos_edge_index[1], data.test_neg_edge_index[1])
                    labels.append(labels_temp[0])
                    labels.append(labels_temp[1])
        y = torch.cat(y, dim=0)
        pred = torch.cat(pred, dim=0)
        labels = torch.cat(labels, dim=0) if metrics_for_labels else []

        if metrics_for_labels:
            return average_precision_score(y, pred), metric_for_label(y, pred, labels, average_precision_score)
        else:
            return average_precision_score(y, pred), None

def metric_for_label(y_true, y_pred, labels, metric):
    '''
    Returns a dictionary with nodes labels as keys and metrics for a specific label as value
    Args:
    y_true - list of true target indices
    y_pred - list of predicted target indices
    labels = labels for which a given metrics should be applied
    metric - metric function that which should be applied
    '''
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    unique_lbls = np.unique(np.array(labels))
    labels = labels.numpy()
    metrics = {}
    # print(labels.shape, y_true.shape, y_pred.shape)
    for label in unique_lbls:
        lbls_index_list = labels == label
        # print(lbls_index_list)
        # print(y_true[lbls_index_list].shape, y_pred[lbls_index_list].shape)
        metrics[label] = metric(y_true[lbls_index_list], y_pred[lbls_index_list])

    return metrics

def get_labels_for_nodes(y, test_pos_edge_index, test_neg_edge_index):
    '''
    Returns list of labels for nodes_ids provided in test_pos_edge_index list and test_neg_edge_index list
    Args:
    y = original list of labels
    test_pos_edge_index - list of indices from test_pos_edge_index that labels should be retrieved
    test_neg_edge_index - list of indices from test_neg_edge_index that labels should be retrieved
    '''
    labels = []
    temp_labels = y
    temp_pos_labels = [temp_labels[i] for i in test_pos_edge_index]
    temp_neg_labels = [temp_labels[i] for i in test_neg_edge_index]
    labels.append(torch.tensor(temp_pos_labels))
    labels.append(torch.tensor(temp_neg_labels))
    return labels

def main():
    args = arg_parse()
    task = args.task
    graph_type = args.graph_type
    model_name = args.model_name
    features = args.features
    writer = SummaryWriter("./log/" + task + '/' + graph_type + '/' + model_name + datetime.now().strftime("%Y%m%d-%H%M%S"))
    task_path = 'nodes_label' if task == 'node' else 'link_pred'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'datasets', 'biograkn', task_path, graph_type)
    if args.dataset == 'biograkn':
        dataset = biograkn_dataset.BiograknDataset(path, task = task, graph_type = graph_type, features=features)
    train(dataset, args, writer = writer) 

if __name__ == '__main__':
    main()
