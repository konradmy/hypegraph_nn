from grakn.client import GraknClient
import pandas as pd

def load_data(uri, keyspace, query, output_dir="", credentials = None):
    '''
    Given graql query, grakn server and keyspace loads the data and saves them into csv file "raw_relationship_data.csv", where columns are nodes and relationships, and rows are traversed graph paths for the specific query.
    Args:
    uri: uri to grakn server
    keyspace: name of the keyspace to load data from
    query: query to be used to fetch the data
    output_dir: directory where the output dataframe should be saved
    credentials (optional): credentials to get access to grakn server
    '''

    with GraknClient(uri=uri, credentials=credentials) as client:
        with client.session(keyspace=keyspace) as session:
            transaction = session.transaction().read()
            data_iterator = transaction.query(query).get()

            print("Fetching...")
            data = [{con: ans.get(con).id for con in ans.map().keys()} for ans in data_iterator]
            print("Fetched")
            print("Transofrming to Pandas DF...")
            data_df = pd.DataFrame(data)
            print("Transformd to Pandas DF")
            print("Saving dataframe to file...")
            data_df.to_csv("{}raw_relationship_data.csv".format(output_dir), index=False)
            print("Saved")



if __name__ == "__main__":
    uri = "localhost:48555"
    keyspace = "biograkn_covid"
    query = "match $g1 isa gene; $g2 isa gene; $v1 isa virus; $v2 isa virus; $pub isa publication; $per isa person; $j isa journal; $mgr ($g1, $g2) isa relation; $gva1 ($g1, $v1) isa gene-virus-association; $gva2 ($g2, $v2) isa gene-virus-association; $m ($pub, $mgr) isa mention; $a ($pub, $per) isa authorship; $pj ($pub, $j) isa publishing; get $m, $mgr, $gva1, $gva2, $a, $pj, $pub, $mgr, $g1, $g2, $v1, $v2, $per, $j;"
    load_data(uri, keyspace, query)