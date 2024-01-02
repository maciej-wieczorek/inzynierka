import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GCNConv, GINConv, GraphConv
from torch_geometric.nn import global_mean_pool, global_add_pool, TopKPooling, global_max_pool as gmp, global_mean_pool as gap
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from copy import deepcopy
import matplotlib.pyplot as plt
import pickle

from data_builder import build_data, build_data2

# Define our GCN class as a pytorch Module
class GCN2(torch.nn.Module):
    def __init__(self, dim_i, dim_o):
        super(GCN2, self).__init__()
        self.conv1 = GraphConv(dim_i, 128)
        self.pool1 = TopKPooling(128, ratio=0.5)
        self.bn1 = BatchNorm1d(128)
        
        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.5)
        self.bn2 = BatchNorm1d(128)
        
        self.conv3 = GraphConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.5)
        self.bn3 = BatchNorm1d(128)
        
        self.conv4 = GraphConv(128, 128)
        self.pool4 = TopKPooling(128, ratio=0.5)
        self.bn4 = BatchNorm1d(128)
        
        self.conv5 = GraphConv(128, 128)
        self.pool5 = TopKPooling(128, ratio=0.5)
        self.bn5 = BatchNorm1d(128)
        
        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64 )
        self.lin3 = torch.nn.Linear(64, dim_o)
        
    def forward(self, data, batch):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x, edge_index, _, batch, _, _ = self.pool4(x, edge_index, None, batch)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)         
        
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x, edge_index, _, batch, _, _ = self.pool5(x, edge_index, None, batch)
        x5 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1) 
       
        x = x1+x2+x3+x4+x5
        x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.lin2(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.log_softmax(self.lin3(x), dim=-1)
        
        return x

class GCN(torch.nn.Module):
    """GCN"""
    def __init__(self, dim_i, dim_h, dim_o):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dim_i, dim_h)
        self.conv2 = GCNConv(dim_h, dim_h)
        self.conv3 = GCNConv(dim_h, dim_h)
        self.lin = Linear(dim_h, dim_o)

    def forward(self, data, batch):
        x = data.x
        edge_index = data.edge_index

        # Node embeddings 
        h = self.conv1(x, edge_index)
        h = h.relu()
        h = self.conv2(h, edge_index)
        h = h.relu()
        h = self.conv3(h, edge_index)

        # Graph-level readout
        hG = global_mean_pool(h, batch)

        # Classifier
        h = F.dropout(hG, p=0.5, training=self.training)
        h = self.lin(h)
        
        return F.softmax(h, dim=1)

def train(model, loader, val_loader, epochs=100, check_point_every=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    best_validation_score = 0.0
    model.train()
    for epoch in range(epochs+1):
        total_loss = 0
        acc = 0

        # Train on batches
        for data in loader:
            data = data.to(device, non_blocking=True)
            optimizer.zero_grad()
            out = model(data, data.batch)
            loss = criterion(out, data.y)
            total_loss += loss / len(loader)
            acc += accuracy(out.argmax(dim=1), data.y) / len(loader)
            loss.backward()
            optimizer.step()

        if(epoch % check_point_every == 0):
            val_loss, val_acc = test(model, val_loader)
            if val_acc > best_validation_score:
                best_model_state = deepcopy(model.state_dict())
                best_validation_score = val_acc
            print(f'Epoch {epoch:>3} | Train Loss: {total_loss:.2f} | Train Acc: {acc*100:>5.2f}% | Val Loss: {val_loss:.2f} | Val Acc: {val_acc*100:.2f}%')
            
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    torch.save(model, 'model.pth')
    return model

@torch.no_grad()
def test(model, loader, conf_matrix=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    loss = 0
    acc = 0

    model_pred = []
    correct_pred = []

    for data in loader:
        data = data.to(device)
        out = model(data, data.batch)
        loss += criterion(out, data.y) / len(loader)
        acc += accuracy(out.argmax(dim=1), data.y) / len(loader)

        if conf_matrix:
            model_pred.extend(list(out.argmax(dim=1).cpu().numpy()))
            correct_pred.extend(list(data.y.cpu().numpy()))

    if conf_matrix:
        ConfusionMatrixDisplay.from_predictions(model_pred, correct_pred)
        plt.show()

    return loss, acc

def print_class_distribution(dataset, labels):
    class_distribution = np.array([data.y[0].tolist() for data in dataset])
    unique_classes, class_counts = np.unique(class_distribution, return_counts=True)
    for class_index, count in zip(unique_classes, class_counts):
        print(f"Class {class_index} ({labels[class_index]}): {count} {round(100*count/class_distribution.size, 2)}%")

def print_dataset_info(dataset, labels, name):
    print(f'{name} = {len(dataset)} graphs')
    print_class_distribution(dataset, labels)
    print()

def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()

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

def write_dataset(dataset, labels):
    with open('dataset.pickle', 'wb') as file:
        pickle.dump(dataset, file)
    with open('labels.pickle', 'wb') as file:
        pickle.dump(labels, file)

def read_dataset():
    with open('dataset.pickle', 'rb') as file:
        dataset = pickle.load(file)
    with open('labels.pickle', 'rb') as file:
        labels = pickle.load(file)
    
    return dataset, labels

def train_model():
    # dataset, labels = build_data2(r'D:\captures\VPN\VNAT_release_1')
    # dataset, labels = build_data()
    # write_dataset(dataset, labels)
    dataset, labels = read_dataset()

    # dataset = balance_dataset(dataset)
    # write_dataset(dataset, labels)

    print_dataset_info(dataset, labels, 'Full dataset')

    train_dataset, test_dataset, val_dataset = split_dataset(dataset)

    print_dataset_info(train_dataset, labels, 'Training set')
    print_dataset_info(val_dataset, labels, 'Validation set')
    print_dataset_info(test_dataset, labels, 'Test set')

    # Create mini-batches
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gcn = GCN2(dim_i=dataset[0].num_features, dim_o=len(labels)).to(device)
    gcn = train(gcn, train_loader, val_loader, epochs=8)
    test_loss, test_acc = test(gcn, test_loader, conf_matrix=True)
    print(f'Test Loss: {test_loss:.2f} | Test Acc: {test_acc*100:.2f}%')

def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('model.pth').to(device)
    dataset, labels = read_dataset()
    dataset_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    test_loss, test_acc = test(model, dataset_loader, conf_matrix=True)
    print(f'Test Loss: {test_loss:.2f} | Test Acc: {test_acc*100:.2f}%')

if __name__ == '__main__':
    train_model()
    # load_model()