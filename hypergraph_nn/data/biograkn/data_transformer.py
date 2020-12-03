import pandas as pd

def transform_data(edges_file, edges_tuples, output_dir="", several_out_files=False, undirected=False):
    '''
    Loads the data from a provided file and transforms the data to a two column dataframe with source and target nodes (edge_index/hyperedge_index). Finally saves data to csv file/files.
    Args:
    edges_file: filepath to a csv file where nodes and hyperrelations are columns and traversed graph paths are rows
    edge_tuples: list of tuples: (node, containing hyperrelation) or (hyperrelation, containing hyperrelation)
    output_dir: directory where the output dataframe should be saved
    several_out_files: if True saves transformed dataframes separately for each tuple from 'edge_tuples' (default: False)
    undirected: if True creates an edge_index file/files for undirected graph
    '''

    edges_df = pd.read_csv(edges_file)
    for column in edges_df.columns:
        edges_df[column] = edges_df[column].str.replace("V","")

    temp_cols = ["source", "target"]
    if several_out_files == True:
        for edge_tuple in edges_tuples:
            temp_df = pd.DataFrame(columns=temp_cols)
            file_out_name = "{}{}_{}_edges.csv".format(output_dir, edge_tuple[0], edge_tuple[1])
            temp_df[temp_cols] = edges_df[edge_tuple]
            if undirected:
                temp_df = make_undirected(temp_df)
            temp_df.to_csv(file_out_name, index=False)
    else:
        temp_df = pd.DataFrame(columns = temp_cols)
        for edge_tuple in edges_tuples:
            temp_df = pd.concat([temp_df, edges_df[edge_tuple].rename(columns={edge_tuple[0]:temp_cols[0], edge_tuple[1]:temp_cols[1]})], ignore_index = True)
        temp_df = temp_df.drop_duplicates()
        if undirected:
            temp_df = make_undirected(temp_df)
        temp_df.to_csv(output_dir + "all_edges_index.csv", index=False)

    
def make_undirected(edges_df):
    '''
    Returns undirected graph from given edge_index/hyperedge_index dataframe.
    Args:
    edges_df: edge_index/hyperedge_index dataframe with the following columns: "source" and "target
    '''
    temp_df = edges_df.rename(columns = {"source": "target", "target": "source"})
    temp_df = pd.concat([edges_df, temp_df], ignore_index=True, sort=False)
    return temp_df


def transform_graph():
    edges_tuples = [["pub", "g1"],
                    ["pub", "g2"],
                    ["g1", "g2"],
                    ["g1", "v1"],
                    ["g2", "v2"],
                    ["pub", "per"],
                    ["pub", "j"],
                    ]
    file_name = "raw_relationship_data.csv"
    transform_data(file_name, edges_tuples, output_dir = 'casual_graph/', undirected=True)

def transform_hypergraph():
    ## To be implemented...
    #Lack of multiple relationships in a current graph
    return None

def transform_grakn_hypergraph():
    edges_tuples = [["pub", "m"],
                    ["mgr", "m"],
                    ["g1", "mgr"],
                    ["g2", "mgr"],
                    ["g1", "gva1"],
                    ["v1", "gva1"],
                    ["g2", "gva2"],
                    ["v2", "gva2"],
                    ["pub", "a"],
                    ["per", "a"],
                    ["pub", "pj"],
                    ["j", "pj"]
                    ]
    file_name = "raw_relationship_data.csv"
    transform_data(file_name, edges_tuples, output_dir = 'grakn_hypergraph/', undirected=False)

if __name__ == "__main__":
    transform_graph()
    transform_grakn_hypergraph()
