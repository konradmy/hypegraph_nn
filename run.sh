###Graph representation: Casual Graph
##Task node
python3 -m hypergraph_nn.models.train --dataset=biograkn --dropout=0.5 --weight_decay=5e-3 --epochs=500 --task=node --graph_type=casual_graph --model_name=GCN
python3 -m hypergraph_nn.models.train --dataset=biograkn --dropout=0.5 --weight_decay=5e-3 --epochs=500 --task=node --graph_type=casual_graph --model_name=GraphSage
python3 -m hypergraph_nn.models.train --dataset=biograkn --dropout=0.5 --weight_decay=5e-3 --epochs=500 --task=node --graph_type=casual_graph --model_name=GAT
##Task link
python3 -m hypergraph_nn.models.train --dataset=biograkn --dropout=0.5 --weight_decay=5e-3 --epochs=500 --task=link --graph_type=casual_graph --model_name=GCN
python3 -m hypergraph_nn.models.train --dataset=biograkn --dropout=0.5 --weight_decay=5e-3 --epochs=500 --task=link --graph_type=casual_graph --model_name=GraphSage
python3 -m hypergraph_nn.models.train --dataset=biograkn --dropout=0.5 --weight_decay=5e-3 --epochs=500 --task=link --graph_type=casual_graph --model_name=GAT

#####Graph representation: Hypergraph mapped to casual graph
##Task node
python3 -m hypergraph_nn.models.train --dataset=biograkn --dropout=0.5 --weight_decay=5e-3 --epochs=500 --task=node --graph_type=hypergraph_to_graph --model_name=GCN
python3 -m hypergraph_nn.models.train --dataset=biograkn --dropout=0.5 --weight_decay=5e-3 --epochs=500 --task=node --graph_type=hypergraph_to_graph --model_name=GraphSage
python3 -m hypergraph_nn.models.train --dataset=biograkn --dropout=0.5 --weight_decay=5e-3 --epochs=500 --task=node --graph_type=hypergraph_to_graph --model_name=GAT
##Task link
python3 -m hypergraph_nn.models.train --dataset=biograkn --dropout=0.5 --weight_decay=5e-3 --epochs=500 --task=link --graph_type=hypergraph_to_graph --model_name=GCN
python3 -m hypergraph_nn.models.train --dataset=biograkn --dropout=0.5 --weight_decay=5e-3 --epochs=500 --task=link --graph_type=hypergraph_to_graph --model_name=GraphSage
python3 -m hypergraph_nn.models.train --dataset=biograkn --dropout=0.5 --weight_decay=5e-3 --epochs=500 --task=link --graph_type=hypergraph_to_graph --model_name=GAT

###Graph representation: Grakn Hypergraph
##Task node grakn_hypergraph
python3 -m hypergraph_nn.models.train --dataset=biograkn --dropout=0.5 --weight_decay=5e-3 --epochs=500 --task=node --graph_type=grakn_hypergraph --model_name=HypergraphConv
##Task link
python3 -m hypergraph_nn.models.train --dataset=biograkn --dropout=0.5 --weight_decay=5e-3 --epochs=500 --task=link --graph_type=grakn_hypergraph --model_name=HypergraphConv
