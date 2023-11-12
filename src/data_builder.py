from torch_geometric.data import Data
import torch
import numpy as np
import os
from grouper import group

CLASSES = ['video stream', 'file share', 'vpn traffic'] # TODO

def get_graph_class(label):
    # idx = CLASSES.index(label) # TODO
    idx = np.random.randint(0, len(CLASSES))
    return idx

def build_data():
    connections_root_dir = os.path.join('src', 'connections')
    graph_data_list = []

    # go through every folder in connections folder
    # labal/class of a graph is a folders name
    # Example 1
    dirs = [d for d in os.listdir(connections_root_dir) if os.path.isdir(os.path.join(connections_root_dir, d))]
    for dir in dirs:
        class_root_dir = os.path.join(connections_root_dir, dir)
        class_dirs = [d for d in os.listdir(class_root_dir) if os.path.isdir(os.path.join(class_root_dir, d))]
        for class_label in class_dirs:
            class_packet_dump_root = os.path.join(class_root_dir, class_label)
            class_packet_dump_files = [f for f in os.listdir(class_packet_dump_root) if os.path.isfile(os.path.join(class_packet_dump_root, f))]
            for packets_dump_file in class_packet_dump_files:
                packets_dump_file_path = os.path.join(class_packet_dump_root, packets_dump_file)
                packets_dump_file_path = 'src\\connections\\connections_1699798282.9167988\\443\\44.241.24.115_443-192.168.0.100_59702.txt'
                with open(packets_dump_file_path) as f:
                    node_data = []
                    edges = []
                    edges.append([])
                    edges.append([])
                    edge_data = []
                    for line in group(f):
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
                    x_data = torch.from_numpy(np.eye(15, dtype=np.float32)[node_data])
                    edge_index_data = torch.tensor(edges, dtype=torch.int64)
                    edge_attr_data = torch.tensor(edge_data, dtype=torch.float32)
                    y_data = torch.tensor([get_graph_class(class_label)], dtype=torch.int64)
                    graph_data_list.append(Data(x=x_data, edge_index=edge_index_data, edge_attr=edge_attr_data,y=y_data))

    return graph_data_list

