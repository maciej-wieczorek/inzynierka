import sys
import networkx as nx
import matplotlib.pyplot as plt

if __name__ == '__main__':
    G = nx.DiGraph()
    nodes = []
    for line in sys.stdin:
        node1, node2, delay = line.split(',')
        G.add_edge(node1, node2, weight=float(delay.strip()))

    pos = nx.circular_layout(G)
    elist = [(u, v) for (u, v, d) in G.edges(data=True)]
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos, edgelist=elist)
    nx.draw_networkx_labels(G, pos)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels)

    plt.show()