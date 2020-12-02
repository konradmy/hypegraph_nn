import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import random
from torch_geometric.data import DataLoader
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import train_test_split_edges

def get_biograkn_data(edge_csv_file_path, nodes_labs_file_path, task = 'node', val_ratio=0.3, test_ratio=0.3):
    '''
    Returns pytorch 'Data' object from provided filepaths.
    Args:
    edge_csv_file_path: filepath to a csv file where nodes and hyperrelations are columns and traversed graph paths are rows
    nodes_labs_file_path: filepath to a two columns csv files with nodes ids and their labels
    val_ratio: percentage ratio for validation data mask
    test_ratio: percentage ratio for test data mask
    '''

    edge_index_df = pd.read_csv(edge_csv_file_path)
    nodes_labels = pd.read_csv(nodes_labs_file_path)
    edge_index = torch.tensor([edge_index_df["source"],
                           edge_index_df["target"]], dtype=torch.long)
    y = torch.tensor(nodes_labels["label"], dtype=torch.long)
    num_nodes = len(y)
    x = torch.ones(num_nodes, 10)

    if task == 'node':
        dataset_masks = create_masks(nodes_labels["node"], val_ratio, test_ratio)
        num_classes = torch.unique(y).size(0)
        data = Data(y=y, x=x, edge_index=edge_index, num_nodes = num_nodes, test_mask=dataset_masks["test"], train_mask=dataset_masks["train"], val_mask=dataset_masks["validation"], num_classes=num_classes)
    elif task == 'link':
        data = Data(y=y, x=x, edge_index=edge_index, num_nodes = num_nodes)
        data = train_test_split_edges(data)
    else:
        raise RuntimeError('Unknown task.')
    return data
    
class BiograknDataset(InMemoryDataset):
    '''
    Pytorch Dataset object for Biograkn data
    '''
    def __init__(self, root, task = 'node', graph_type = 'casual_graph', transform=None, pre_transform=None):
        self.task = task
        self.graph_type = graph_type
        super(BiograknDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return []
        
    @property
    def processed_file_names(self):
        
        return ['dataset.dataset']

    def download(self):
        pass
        
    def process(self):

        data_list = []
        file_path = get_process_file_path(self.graph_type)
        edge_csv_file_path = file_path + 'all_edges_index.csv'
        nodes_labs_file_path = file_path + 'nodes_labels.csv'

        data_list.append(get_biograkn_data(edge_csv_file_path, nodes_labs_file_path, task = self.task))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def get_process_file_path(graph_type):
    '''
        return a file path to get edge and nodes csv files for a given task and graph type
        Args:
        task: Graph NN task ('node', 'link')
        graph_type: Type of the data representation ('casual_graph', 'hypergraph', 'grakn_hypergraph')
    '''
    file_path = '../../data/biograkn/'
    
    if graph_type == 'casual_graph':
        file_path = file_path + 'casual_graph/encoded/'
    elif graph_type == 'hypergraph':
        file_path = file_path + 'hypergraph/encoded/'
    elif graph_type == 'grakn_hypergraph':
        file_path = file_path + 'grakn_hypergraph/encoded/'
    else:
        raise RuntimeError('Unknown graph type.')

    return file_path
    

def create_masks(nodes_list, val_ratio, test_ratio):
    val_len = int(len(nodes_list) * val_ratio)
    test_len = int(len(nodes_list) * test_ratio)
    train_len = len(nodes_list) - val_len - test_len
    len_dictionary = {
        "validation": val_len,
        "test": test_len,
        "train": train_len
    }

    datasets = {}
    labels_set = set(nodes_list)

    for dataset_type in ["train", "validation", "test"]:
        temp_list = list(labels_set)
        sampling = random.sample(temp_list, k=len_dictionary[dataset_type])
        datasets[dataset_type] = torch.tensor(np.isin(nodes_list, sampling), dtype=torch.bool)
        labels_set.difference_update(set(sampling))
    
    return datasets



if __name__ == "__main__":
    dataset_lg = BiograknDataset('./link_pred/casual_graph/', task = 'link', graph_type = 'casual_graph')
    dataset_lgh = BiograknDataset('./link_pred/grakn_hypergraph/', task = 'link', graph_type = 'grakn_hypergraph')
    dataset_ng = BiograknDataset('./nodes_label/casual_graph/', task = 'node', graph_type = 'casual_graph')
    dataset_ngh = BiograknDataset('./nodes_label/grakn_hypergraph/', task = 'node', graph_type = 'grakn_hypergraph')
    print(dataset_lg[0], '\n\n')
    print(dataset_lgh[0], '\n\n')
    print(dataset_ng[0], '\n\n')
    print(dataset_ngh[0], '\n\n')
