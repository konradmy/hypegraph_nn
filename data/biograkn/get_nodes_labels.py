import pandas as pd
import json
import re

def get_labels(edges_file, output_dir = "", read_cols = None, cols_to_rename=None):
    '''
    Loads the data from a provided file and transforms the data to a dataframe with two columns: "node" and "label" ("node" columns contains sorted distinct node ids). Finally saves data to 'nodes_labels.csv' csv file.
    edges_file: filepath to a csv file where nodes and hyperrelations are columns and traversed graph paths are rows
    output_dir: directory where the output dataframe should be saved
    read_cols: list of columns names to be transofmed from provided files
    cols_to_rename: dictionary used to rename columns (which are used as labels); useful if several columns contains the same type of nodes e.g. 'g1' and 'g2' both have label 'gene' but appear separately because of specific graph traversion path
    '''
    usecols = None
    if read_cols != None:
        usecols = read_cols
    raw_df = pd.read_csv(edges_file, usecols=usecols)
    raw_df = rename_cols(raw_df, cols_to_rename)
    labels = raw_df.columns

    enc_dict = encode_columns(labels, output_dir)
    raw_df = raw_df.rename(columns = enc_dict)

    raw_df = pd.DataFrame(raw_df.stack()).droplevel(0)
    raw_df.reset_index(inplace=True)
    raw_df.columns = ["label", "node"]
    raw_df = raw_df[["node", "label"]]
    raw_df["node"] = raw_df["node"].str.replace("V","")
    raw_df["node"] = pd.to_numeric(raw_df["node"])
    raw_df = raw_df.sort_values(by=["node"])
    raw_df = raw_df.drop_duplicates()
    raw_df.to_csv("{}nodes_labels.csv".format(output_dir), index=False)
    
def rename_cols(df, cols_to_rename):
    cols_to_rename = cols_to_rename
    if cols_to_rename == None:
        #take only names that contain digits and map them to dict {col_name: new_col_name}, where "new_col_name" is a "col_name" up to the first digit in a string
        cols_to_rename = {col: col[:col.find(re.search(r'\d', col)[0])] for col in df.columns if bool(re.search(r'\d', col))}
    return df.rename(columns = cols_to_rename)
    


def encode_columns(columns, output_dir):
    dec_dict = {i: val for i, val in enumerate(set(columns))}
    save_dict_as_json(dec_dict, output_dir)
    enc_dict = {val: i for i, val in enumerate(set(columns))}
    return enc_dict

def save_dict_as_json(enc_dict, output_dir):
    enc_json = json.dumps(enc_dict)
    f = open("{}encoded_labels.json".format(output_dir),"w")
    f.write(enc_json)
    f.close()

def graph_labels():
    raw_df = "raw_relationship_data.csv"
    get_labels(raw_df, output_dir='casual_graph/')

def grakn_hypergraph_labels():
    raw_df = "raw_relationship_data.csv"
    get_labels(raw_df, output_dir='grakn_hypergraph/')

if __name__ == "__main__":
    graph_labels()
    grakn_hypergraph_labels()