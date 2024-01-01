from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
import pandas as pd
from scapy.all import PcapReader
from scapy.compat import raw
import os

NUM_GROUPS = 30

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

    x_data_1 = torch.from_numpy(np.eye(NUM_GROUPS, dtype=np.float32)[node_data])

    edge_data = StandardScaler().fit_transform(np.array(edge_data).reshape(-1, 1)).reshape(-1)
    x_data_2 = np.zeros([len(node_data), NUM_GROUPS], dtype=np.float32)
    x_data_3 = np.zeros([len(node_data), NUM_GROUPS], dtype=np.float32)
    for i in range(len(edges[0])):
        x_data_2[edges[0][i]][edges[1][i]] = edge_data[i]
        x_data_3[edges[1][i]][edges[0][i]] = edge_data[i]
    x_data_2 = torch.from_numpy(x_data_2)
    x_data_3 = torch.from_numpy(x_data_3)

    x_data = torch.cat((x_data_1, x_data_2, x_data_3), dim=1)
    edge_index_data = torch.tensor(edges, dtype=torch.int64)
    edge_attr_data = torch.tensor(edge_data, dtype=torch.float32)
    y_data = torch.tensor([label_index], dtype=torch.int64)

    return Data(x=x_data, edge_index=edge_index_data, edge_attr=edge_attr_data,y=y_data)

def build_data():
    count = 0
    print(f'Building data: {count}  ', end='\r')

    df = pd.read_csv('graphs.csv')
    df = df[df['datasource'] == 'VPN/NONVPN NETWORK APPLICATION TRAFFIC DATASET (VNAT)']
    df = df[~df['label'].str.contains('nonvpn-voip')]
    df = df[~df['label'].str.contains('nonvpn-ssh')]
    # df = df[~df['label'].str.contains('sftp')]
    # df = df[df['label'].str.contains('nonvpn')]

    labels = list(df['label'].unique())

    graph_data_list = []

    for _, row in df.iterrows():
        graph_data_list.append(build_graph_tensor_representation(labels.index(row['label']), row['graph']))

        print(f'Building data: {count}  ', end='\r')
        count += 1

    return graph_data_list, labels

def build_data2(captures_path):
    CONNECTION_SIZE = 100
    MAX_PACKET_SIZE = 1500

    graph_data_list = []
    labels = []

    for filename in os.listdir(captures_path):
        label = "-".join(filename.split('.')[0].split('_')[:2])
        if label not in labels:
            labels.append(label)

    for filename in os.listdir(captures_path):
        label = "-".join(filename.split('.')[0].split('_')[:2])
        filepath = os.path.join(captures_path, filename)
        if os.path.isfile(filepath):
            x_data_list = []
            with PcapReader(filepath) as pcap:
                print(f'Reading: {filepath}')
                count = 0
                for packet in pcap:
                    x = np.frombuffer(raw(packet), dtype=np.uint8)[0:MAX_PACKET_SIZE] / 255
                    if len(x) < MAX_PACKET_SIZE:
                        x = np.pad(x, pad_width=(0, MAX_PACKET_SIZE - len(x)), constant_values=0)
                    x_data_list.append(x)

                    if len(x_data_list) == CONNECTION_SIZE:
                        x_data = torch.from_numpy(np.array(x_data_list, dtype=np.float32))
                        edge_index_data = torch.tensor([[i, i+1] for i in range(len(x_data)-1)]).t().contiguous()
                        y_data = torch.tensor([labels.index(label)], dtype=torch.int64)
                        data = Data(x=x_data, edge_index=edge_index_data, y=y_data)
                        graph_data_list.append(data)
                        x_data_list = []

                    count += 1
                    print(f'{count} packets', end='\r')

    return graph_data_list, labels
