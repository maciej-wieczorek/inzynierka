from torch_geometric.data import Data
import torch
import numpy as np
import pandas as pd

NUM_GROUPS = 30

def z_scale(input_list):
    if input_list and len(input_list) > 0:
        mean_value = sum(input_list) / len(input_list)
        std_deviation = (sum((x - mean_value) ** 2 for x in input_list) / len(input_list)) ** 0.5
        if std_deviation != 0:
            z_scaled_list = [(x - mean_value) / std_deviation for x in input_list]
            return z_scaled_list
        else:
            return input_list
    else:
        return input_list

def build_graph_tensor_representation(label_index, graph):
    node_data = []
    edges = []
    edges.append([])
    edges.append([])
    edge_data = []

    for line in graph.split(' '):
        split = line.split(',')
        node1_index = int(split[0])
        if node1_index not in node_data:
            node_data.append(node1_index)
        node2_index = int(split[1])
        if node2_index not in node_data:
            node_data.append(node2_index)
        edges[0].append(node_data.index(node1_index))
        edges[1].append(node_data.index(node2_index))
        delay = float(split[2])
        edge_data.append(delay)

    x_data = torch.from_numpy(np.eye(NUM_GROUPS, dtype=np.float32)[node_data])
    edge_index_data = torch.tensor(edges, dtype=torch.int64)
    edge_attr_data = torch.tensor(z_scale(edge_data), dtype=torch.float32)
    y_data = torch.tensor([label_index], dtype=torch.int64)

    return Data(x=x_data, edge_index=edge_index_data, edge_attr=edge_attr_data,y=y_data)

def build_data():
    count = 0
    print(f'Building data: {count}  ', end='\r')

    df = pd.read_csv('graphs.csv')
    df = df[df['datasource'] == 'VPN/NONVPN NETWORK APPLICATION TRAFFIC DATASET (VNAT)']
    # df = df[~df['label'].str.contains('scp')]
    # df = df[~df['label'].str.contains('sftp')]
    # df = df[df['label'].str.contains('nonvpn')]

    labels = list(df['label'].unique())

    graph_data_list = []

    for _, row in df.iterrows():
        graph_data_list.append(build_graph_tensor_representation(labels.index(row['label']), row['graph']))

        print(f'Building data: {count}  ', end='\r')
        count += 1

    return graph_data_list, labels
