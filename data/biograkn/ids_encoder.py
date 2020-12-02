import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_ids(edge_csv_file_path, nodes_labs_file_path, output_path):
    '''
    Encodes nodes or hyperrelations ids so that they are within range 1...N where N is number of nodes in nodes_labels. Saves data to csv in provided directory.
    Args:
    edge_csv_file_path: filepath to a csv file where nodes and hyperrelations are columns and traversed graph paths are rows
    nodes_labs_file_path: filepath to a two columns csv files with nodes ids and their labels
    output_path: file directory to save encoded dataframes
    '''
    edge_index_df = pd.read_csv(edge_csv_file_path)
    nodes_labels = pd.read_csv(nodes_labs_file_path)

    lbl_encoder = LabelEncoder()
    lbl_encoder.fit(nodes_labels["node"])
    nodes_labels["node"] = lbl_encoder.transform(nodes_labels["node"])
    edge_index_df["source"] = lbl_encoder.transform(edge_index_df["source"])
    edge_index_df["target"] = lbl_encoder.transform(edge_index_df["target"])
    edge_index_df.to_csv("{}all_edges_index.csv".format(output_path), index=False)
    nodes_labels.to_csv("{}nodes_labels.csv".format(output_path), index=False)

def encode_graph():
    edge_csv_file_path = 'casual_graph/all_edges_index.csv'
    nodes_labs_file_path = 'casual_graph/nodes_labels.csv'
    output_path = 'casual_graph/encoded/'
    encode_ids(edge_csv_file_path, nodes_labs_file_path, output_path)

def encode_grakn_hypergraph():
    edge_csv_file_path = 'grakn_hypergraph/all_edges_index.csv'
    nodes_labs_file_path = 'grakn_hypergraph/nodes_labels.csv'
    output_path = 'grakn_hypergraph/encoded/'
    encode_ids(edge_csv_file_path, nodes_labs_file_path, output_path)

if __name__ == "__main__":
    encode_graph()
    encode_grakn_hypergraph()

