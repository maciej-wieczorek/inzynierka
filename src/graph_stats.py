import networkx as nx
import matplotlib.pyplot as plt

def compute_graph_statistics(G):
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G)
    average_degree = sum(dict(G.degree()).values()) / num_nodes
    degree_distribution = nx.degree_histogram(G)
    average_clustering_coefficient = nx.average_clustering(G)
    shortest_path_length = nx.average_shortest_path_length(G)
    degree_centrality = nx.degree_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    eccentricity = nx.eccentricity(G)
    diameter = nx.diameter(G)
    radius = nx.radius(G)

    return {
        'NumNodes': num_nodes,
        'NumEdges': num_edges,
        'Density': density,
        'AverageDegree': average_degree,
        'DegreeDistribution': degree_distribution,
        'AverageClusteringCoefficient': average_clustering_coefficient,
        'AverageShortestPathLength': shortest_path_length,
        'DegreeCentrality': degree_centrality,
        'ClosenessCentrality': closeness_centrality,
        'BetweennessCentrality': betweenness_centrality,
        'Eccentricity': eccentricity,
        'Diameter': diameter,
        'Radius': radius
    }

def draw_degree_dist(degree_distribution, title):
    plt.bar(range(len(degree_distribution)), degree_distribution)
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.show()
