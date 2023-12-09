import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from data_builder import NUM_GROUPS, build_data

class GCN(torch.nn.Module):
    """GCN"""
    def __init__(self, dim_h, dim_o):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(NUM_GROUPS, dim_h)
        self.conv2 = GCNConv(dim_h, dim_h)
        self.conv3 = GCNConv(dim_h, dim_h)
        self.lin = Linear(dim_h, dim_o)

    def forward(self, x, edge_index, batch):
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

def train(model, loader, print_every=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    epochs = 100

    model.train()
    for epoch in range(epochs+1):
        total_loss = 0
        acc = 0

        # Train on batches
        for data in loader:
            data = data.to(device, non_blocking=True)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            total_loss += loss / len(loader)
            acc += accuracy(out.argmax(dim=1), data.y) / len(loader)
            loss.backward()
            optimizer.step()

        if(epoch % print_every == 0):
            print(f'Epoch {epoch:>3} | Train Loss: {total_loss:.2f} | Train Acc: {acc*100:>5.2f}%')
            
    return model

@torch.no_grad()
def test(model, loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    loss = 0
    acc = 0

    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss += criterion(out, data.y) / len(loader)
        acc += accuracy(out.argmax(dim=1), data.y) / len(loader)

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

def split_dataset(dataset, test_size=0.2, validation_size=0.1, random_state=42):
    # Extract labels and indices
    labels = [data.y[0] for data in dataset]
    indices = list(range(len(dataset)))

    # Split the dataset into training and temp (test + validation)
    train_indices, temp_indices, train_labels, temp_labels = train_test_split(
        indices, labels, test_size=(test_size + validation_size), stratify=labels, random_state=random_state)

    # Split the temp dataset into test and validation
    test_indices, validation_indices, _, _ = train_test_split(
        temp_indices, temp_labels, test_size=(validation_size / (test_size + validation_size)),
        stratify=temp_labels, random_state=random_state)

    # Create DataLoader for each split
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    train_loader = DataLoader(dataset, batch_size=64, sampler=train_sampler)

    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
    test_loader = DataLoader(dataset, batch_size=64, sampler=test_sampler)

    validation_sampler = torch.utils.data.SubsetRandomSampler(validation_indices)
    validation_loader = DataLoader(dataset, batch_size=64, sampler=validation_sampler)

    return train_loader, test_loader, validation_loader

def train_model():
    dataset, labels = build_data()
    random.shuffle(dataset)

    print_dataset_info(dataset, labels, 'Full dataset')

    # Create training, validation, and test sets
    train_dataset = dataset[:int(len(dataset)*0.8)]
    val_dataset   = dataset[int(len(dataset)*0.8):int(len(dataset)*0.9)]
    test_dataset  = dataset[int(len(dataset)*0.9):]

    #train_dataset, test_dataset, val_dataset = split_dataset(dataset)

    print_dataset_info(train_dataset, labels, 'Training set')
    print_dataset_info(val_dataset, labels, 'Validation set')
    print_dataset_info(test_dataset, labels, 'Test set')

    # Create mini-batches
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gcn = GCN(dim_h=32, dim_o=len(labels)).to(device)
    gcn = train(gcn, train_loader)
    test_loss, test_acc = test(gcn, test_loader)
    print(f'Test Loss: {test_loss:.2f} | Test Acc: {test_acc*100:.2f}%')

if __name__ == '__main__':
    train_model()