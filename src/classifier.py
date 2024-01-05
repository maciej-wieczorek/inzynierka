import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GCNConv, GraphConv
from torch_geometric.nn import global_mean_pool, global_add_pool, TopKPooling, global_max_pool as gmp, global_mean_pool as gap
from torch_geometric.loader import DataLoader
from sklearn.metrics import ConfusionMatrixDisplay
from copy import deepcopy
import matplotlib.pyplot as plt

from data_builder import build_data, build_data2, write_dataset, read_dataset, balance_dataset, split_dataset, print_dataset_info

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
        
    def forward(self, x, edge_index, batch):
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
        x = F.dropout(x, p=0.2, training=self.training)
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
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.log_softmax(self.lin3(x), dim=-1)
        
        return x

class GCN(torch.nn.Module):
    """GCN"""
    def __init__(self, dim_i, dim_h, dim_o):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dim_i, dim_h).jittable()
        self.conv2 = GCNConv(dim_h, dim_h).jittable()
        self.conv3 = GCNConv(dim_h, dim_h).jittable()
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

def train(model, loader, val_loader, epochs=10, patience=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    early_stop_counter = 0
    best_val_loss = float('inf')

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

        val_loss, val_acc = test(model, val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            best_model_state = deepcopy(model.state_dict())
        else:
            early_stop_counter += 1

        print(f'Epoch {epoch:>3} | Train Loss: {total_loss:.2f} | Train Acc: {acc*100:>5.2f}% | Val Loss: {val_loss:.2f} | Val Acc: {val_acc*100:.2f}%')

        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch} with validation loss: {best_val_loss}")
            break
            
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
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

@torch.no_grad()
def conf_matrix(model, loader, labels):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    model_pred = []
    correct_pred = []

    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
    
        model_pred.extend(list(out.argmax(dim=1).cpu().numpy()))
        correct_pred.extend(list(data.y.cpu().numpy()))

    ConfusionMatrixDisplay.from_predictions(model_pred, correct_pred, display_labels=labels)
    plt.show()

def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()

def train_model():
    # dataset, labels = build_data2(r'D:\captures\VPN\VNAT_release_1')
    # dataset, labels = build_data()
    # write_dataset(dataset, labels, 'dataset1.pickle', 'labels1.pickle')
    dataset, labels = read_dataset('dataset1.pickle', 'labels1.pickle')

    dataset = balance_dataset(dataset)
    # write_dataset(dataset, labels, 'dataset1.pickle', 'labels1.pickle')

    print_dataset_info(dataset, labels, 'Full dataset')

    best_test_loss = float('inf')
    while True:
        train_dataset, test_dataset, val_dataset = split_dataset(dataset)

        # print_dataset_info(train_dataset, labels, 'Training set')
        # print_dataset_info(val_dataset, labels, 'Validation set')
        # print_dataset_info(test_dataset, labels, 'Test set')

        # Create mini-batches
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=True)
        test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GCN(dim_i=dataset[0].num_features, dim_h=32, dim_o=len(labels)).to(device)
        model = train(model, train_loader, val_loader, epochs=100)
        model = torch.jit.script(model)
        # conf_matrix(model, test_loader, labels)
        test_loss, test_acc = test(model, test_loader)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            print(f'Better model found. Saving...')
            torch.jit.save(model, 'model.pt')
        print(f'Test Loss: {test_loss:.2f} | Test Acc: {test_acc*100:.2f}%')

def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.jit.load('model.pt').to(device)
    dataset, labels = read_dataset('dataset1.pickle', 'labels1.pickle')
    # dataset, labels = build_data()
    dataset_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    conf_matrix(model, dataset_loader, labels)
    test_loss, test_acc = test(model, dataset_loader)
    print(f'Dataset Loss: {test_loss:.2f} | Dataset Acc: {test_acc*100:.2f}%')

if __name__ == '__main__':
    # train_model()
    load_model()