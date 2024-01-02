from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
import pandas as pd
from scapy.all import PcapReader
from scapy.compat import raw
import os
import random
import pickle
from sklearn.model_selection import train_test_split

NUM_GROUPS = 30

def get_label(filename, combined_labels=True):
    if combined_labels:
        categories = {
            'video-stream': ['youtube', 'netflix', 'vimeo'],
            'file-transfer': ['scp', 'sftp', 'rsync'],
            'chat' : ['chat'],
            'voip' : ['voip'],
            'remote-desktop': ['rdp'],
            'ssh' : ['ssh']
        }

        for category in categories:
            for option in categories[category]:
                if option in filename.lower():
                    return category
        return 'other'
    else:
        return "-".join(filename.split('.')[0].split('_')[:2]) 

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
    df = df[~df['label'].str.contains('voip')]
    # df = df[~df['label'].str.contains('nonvpn-ssh')]
    # df = df[df['label'].str.contains('nonvpn')]

    labels = list(set(map(lambda x: get_label(x), list(df['label'].unique()))))

    graph_data_list = []

    for _, row in df.iterrows():
        graph_data_list.append(build_graph_tensor_representation(labels.index(get_label(row['label'])), row['graph']))

        print(f'Building data: {count}  ', end='\r')
        count += 1

    return graph_data_list, labels

def build_data2(captures_path):
    CONNECTION_SIZE = 10
    MAX_PACKET_SIZE = 1500
    PACKET_LIMIT_PER_LABEL = 40_000

    graph_data_list = []
    labels = []

    labels_count = {}

    for filename in os.listdir(captures_path):
        label = get_label(filename, combined_labels=False)
        if label not in labels:
            labels.append(label)

    labels_combined = list({get_label(x) for x in labels})

    for filename in os.listdir(captures_path):
        label = get_label(filename, combined_labels=False)
        if label not in labels_count:
            labels_count[label] = 0
        filepath = os.path.join(captures_path, filename)
        if os.path.isfile(filepath):
            x_data_list = []
            with PcapReader(filepath) as pcap:
                print(f'Reading: {filepath}')
                count = 0
                for packet in pcap:
                    if labels_count[label] >= PACKET_LIMIT_PER_LABEL:
                        break

                    x = np.frombuffer(raw(packet), dtype=np.uint8)[0:MAX_PACKET_SIZE] / 255
                    if len(x) < MAX_PACKET_SIZE:
                        x = np.pad(x, pad_width=(0, MAX_PACKET_SIZE - len(x)), constant_values=0)
                    x_data_list.append(x)

                    if len(x_data_list) == CONNECTION_SIZE:
                        x_data = torch.from_numpy(np.array(x_data_list, dtype=np.float32))
                        edge_index_data = torch.tensor([[i, i+1] for i in range(len(x_data)-1)]).t().contiguous()
                        y_data = torch.tensor([labels_combined.index(get_label(label))], dtype=torch.int64)
                        data = Data(x=x_data, edge_index=edge_index_data, y=y_data)
                        graph_data_list.append(data)
                        x_data_list = []

                    labels_count[label] += 1
                    count += 1
                    print(f'{count} packets', end='\r')

    return graph_data_list, labels_combined

def print_class_distribution(dataset, labels):
    class_distribution = np.array([data.y[0].tolist() for data in dataset])
    unique_classes, class_counts = np.unique(class_distribution, return_counts=True)
    for class_index, count in zip(unique_classes, class_counts):
        print(f"Class {class_index} ({labels[class_index]}): {count} {round(100*count/class_distribution.size, 2)}%")

def print_dataset_info(dataset, labels, name):
    print(f'{name} = {len(dataset)} graphs')
    print_class_distribution(dataset, labels)
    print()

def balance_dataset(dataset):
    # Count the number of examples for each class
    num_examples_per_class = {}
    for data in dataset:
        label = data.y.item()
        if label not in num_examples_per_class:
            num_examples_per_class[label] = 0
        num_examples_per_class[label] += 1

    # Determine the target number of examples per class (minimum count)
    target_num_examples = min(num_examples_per_class.values())

    # Create a new list to store the balanced dataset
    balanced_dataset = []

    # Iterate through the dataset, keeping only the target number of examples for each class
    for label, count in num_examples_per_class.items():
        indices = [i for i, data in enumerate(dataset) if data.y.item() == label]
        random.shuffle(indices)
        selected_indices = indices[:target_num_examples]
        balanced_dataset.extend([dataset[i] for i in selected_indices])

    return balanced_dataset

def split_dataset(dataset, test_size=0.2, validation_size=0.1, random_state=None):
    indices = list(range(len(dataset)))

    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_state)
    train_indices, val_indices = train_test_split(train_indices, test_size=validation_size, random_state=random_state)

    train_dataset = [dataset[i] for i in train_indices]
    test_dataset = [dataset[i] for i in test_indices]
    val_dataset = [dataset[i] for i in val_indices]

    return train_dataset, test_dataset, val_dataset

def write_dataset(dataset, labels, dataset_name="dataset.pickle", labels_name="labels.pickle"):
    with open(dataset_name, 'wb') as file:
        pickle.dump(dataset, file)
    with open(labels_name, 'wb') as file:
        pickle.dump(labels, file)

def read_dataset(dataset_name="dataset.pickle", labels_name="labels.pickle"):
    with open(dataset_name, 'rb') as file:
        dataset = pickle.load(file)
    with open(labels_name, 'rb') as file:
        labels = pickle.load(file)
    
    return dataset, labels
