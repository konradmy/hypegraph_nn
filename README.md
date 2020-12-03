# hypegraph_nn
This repository was crated for initial analysis of Graph Neural Networks on hypergraphs. The main goal is a comparison results between hypergraphs and graphs using different data representation and different models.

## Data storage
Data are stored in Grakn [(grakn.ai)](https://grakn.ai) - open-source knowledge graph. It is designed to store data represented as a [hypergraph](https://en.wikipedia.org/wiki/Hypergraph). It enables to create hyperedges which are non-pair wise relationships. Moreover, hyperedges can link other hyperedges. Considering these features from Data Representation point of view, Grakn provides an opportunity to build a rich data structure, which can be further 'flatten' to less expresive forms.


## Data

### Datasets
* **biograkn covid** - initial dataset for testing. Contains data from *Biograkn COVID*, which is a knowledge graph with COVID-19 related data [link](https://towardsdatascience.com/weve-released-a-covid-19-knowledge-graph-96a15d112fac).

So far no features has been added and GNN Stack learns relationships based on graph structure

## Models

* **GCNConv** - described in [“Semi-supervised Classification with Graph Convolutional Networks”](https://arxiv.org/abs/1609.02907) paper. [PyTorch GCNConv model] (https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv)
* **GraphSAGE** - described in ["Inductive Representation Learning on Large Graphs”] (https://arxiv.org/abs/1706.02216) paper. [PyTorch SAGEConv model] (https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SAGEConv)
* **GAT*** - desribed in [“Graph Attention Networks”](https://arxiv.org/abs/1710.10903) paper. [PyTorch SAGEConv model] (https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATConv)
* **HypergraphConv*** - described in [“Hypergraph Convolution and Hypergraph Attention”](https://arxiv.org/abs/1901.08150) paper. [PyTorch HypergraphConv model](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.HypergraphConv)
* **HyperGraphConv extension

### **HyperGraphConv extension

## Data Representation
In order to assess the influence of hypergraphs data representation on graph algorithms, there is a need to define different data representation types which can be applied to the same datasets, which allows for comparison. The following types has been defined to meet these criteria.

### Normal graph

Data are stored in a casual pair-wise graph. Hyperrelations are removed and replaced by pair-wise relations in a following manner:
* hyperrelation which links *N* nodes is replaced by pair-wise relations between all of these *N* nodes - number of created edges: *N(N – 1)/2*

This representation is reffered as *casual_graph* in a code.

### (Grakn) Hypergraph mapped to a Normal graph

Data from a given hypergraph (can be Grakn type of hypergraph) are stored in a pair-wise graph where:
* hyperrelations become nodes in a pair-wise graph and all vertices related by a given hyperrelation have now pair-wise edges with a hyperrelation represented as a node

This representation is referred as *hypergraph_to_graph* in a code.

### Hypergraph

Data are represented as a classic definition of hypergraph. It's hyperedges are non pair-wise and can link more than 2 edges. Yet, it doesn't allow a hyperedge to link other hyperedges. Data from Grakn hypergraph are mapped in a following way:
* all related hyperedges are merged to one and nodes that they link becomes link by newly created merged hyperedge

This representation is referred as *hypergraph* in a code.

### Grakn hypergraph

This is the starting method of data representation. Hyperedges can can link many nodes (other hyperedges). It can also link hyperedges with nodes.

This representation is referred as *grakn_hypergraph* in a code.
