import networkx as nx
import matplotlib.pyplot as plt
import torch

def draw_graph(G):
    elist = [(u, v) for (u, v, d) in G.edges(data=True)]
    elist1 = []
    elist2 = []
    for pair in elist:
        if ((pair[1], pair[0]) not in elist1):
            elist1.append(pair)
        else:
            elist2.append(pair)
    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos)

    pos1 = {k: (v[0], v[1]+0.02) for k, v in pos.items()}
    nx.draw_networkx_edges(G, pos1, edgelist=elist1)

    pos2 = {k: (v[0], v[1]-0.02) for k, v in pos.items()}
    nx.draw_networkx_edges(G, pos2, edgelist=elist2)

    edge_labels1 = {k:f'{v:.2f}' for (k,v) in nx.get_edge_attributes(G, "weight").items() if k in elist1} 
    pos1 = {k: (v[0]+0.1, v[1]) for k, v in pos1.items()}
    nx.draw_networkx_edge_labels(G, pos1, edge_labels1)

    edge_labels2 = {k:f'{v:.2f}' for (k,v) in nx.get_edge_attributes(G, "weight").items() if k in elist2} 
    pos2 = {k: (v[0]-0.1, v[1]) for k, v in pos2.items()}
    nx.draw_networkx_edge_labels(G, pos2, edge_labels2)

    plt.show()


def get_nx_representation(x, edge_index):
    G = nx.DiGraph()
    for node in x:
        node1 = torch.nonzero(node[:30]).squeeze().item()
        delays_to_neighbours = node[30:60]
        neighbours_indices = torch.nonzero(delays_to_neighbours).reshape(-1)
        for node2 in neighbours_indices:
            delay = delays_to_neighbours[node2].item()
            G.add_edge(node1, node2.item(), weight=delay)
    return G

def draw(graph):
    draw_graph(get_nx_representation(graph))
