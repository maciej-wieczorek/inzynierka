import sys
import networkx as nx
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
    G = nx.DiGraph()
    for line in sys.stdin:
        node1, node2, delay = line.split(',')
        G.add_edge(node1, node2, weight=float(delay.strip()))

    draw_graph(G)
