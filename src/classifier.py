import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GCNConv, GraphConv
from torch_geometric.nn import global_mean_pool, global_add_pool, TopKPooling, global_max_pool as gmp, global_add_pool as gap
from torch_geometric.loader import DataLoader
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from tqdm import tqdm
import math
import os
import time

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from copy import deepcopy
import matplotlib.pyplot as plt

from dataset import PacketsDatapipe, get_labels

ARTIFACTS_DIR = os.path.join('artifacts', f'training-{time.time()}')

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

def train(model, train_loader, train_num_batches, val_loader, val_num_batches, epochs=30, patience=5, learning_curve=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    early_stop_counter = 0
    best_val_loss = float('inf')
    best_model_state = None

    lc_file = None
    if learning_curve:
        lc_file = open(os.path.join(ARTIFACTS_DIR, 'learning-curve.csv'), mode='wt')
        lc_file.write('epoch, train_loss, train_acc, val_loss, val_acc\n')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        acc = 0

        # Train on batches
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch', total=train_num_batches) as t:
            for data in train_loader:
                data = data.to(device, non_blocking=True)
                optimizer.zero_grad()
                out = model(data.x, data.edge_index, data.batch)
                loss = criterion(out, data.y)
                total_loss += loss / train_num_batches
                acc += accuracy(out.argmax(dim=1), data.y) / train_num_batches
                loss.backward()
                optimizer.step()

                t.set_postfix(loss=total_loss.item() * (train_num_batches / (t.n+1)), acc=acc * (train_num_batches / (t.n+1)))
                t.update()

        val_loss, val_acc = test(model, val_loader, val_num_batches, desc='Validation')

        if learning_curve:
            train_loss, train_acc = test(model, train_loader, train_num_batches, 'Train')
            lc_file.write(f'{epoch}, {train_loss:.3f}, {train_acc:.3f}, {val_loss:.3f}, {val_acc:.3f}\n')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            best_model_state = deepcopy(model.state_dict())
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch} with validation loss: {best_val_loss}")
            break

    if lc_file is not None:
        lc_file.close()
            
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model

@torch.no_grad()
def test(model, loader, num_batches, desc='Test'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    loss = 0
    acc = 0

    with tqdm(loader, desc=desc, unit='batch', total=num_batches) as t:
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            loss += criterion(out, data.y) / num_batches
            acc += accuracy(out.argmax(dim=1), data.y) / num_batches

            t.set_postfix(loss=loss.item() * (num_batches / (t.n+1)), acc=acc * (num_batches / (t.n+1)))
            t.update()

    return loss.item() * (num_batches / (t.n)), acc * (num_batches / (t.n))

@torch.no_grad()
def conf_matrix(model, loader, num_batches, labels):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    model_pred = []
    correct_pred = []

    with tqdm(loader, desc="Confusion matrix", unit='batch', total=num_batches) as t:
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
        
            model_pred.extend(list(out.argmax(dim=1).cpu().numpy()))
            correct_pred.extend(list(data.y.cpu().numpy()))

            t.update()

    cm = confusion_matrix(model_pred, correct_pred)
    cmp = ConfusionMatrixDisplay(cm, display_labels=[labels[x] for x in sorted(list(set(correct_pred)))])
    fig, ax = plt.subplots(figsize=(12,10))
    cmp.plot(ax=ax)
    fig.savefig(os.path.join(ARTIFACTS_DIR, 'conf_matrix'))

    num_rows, num_cols = cm.shape
    rows, cols = np.indices((num_rows, num_cols))
    cm_latex = np.column_stack((cols.flatten(), rows.flatten(), cm.flatten()))
    np.savetxt(os.path.join(ARTIFACTS_DIR, 'conv_matrix_latex.txt'), cm_latex, fmt='%d', delimiter=' ', comments='')

def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()

packet_list_dataset_location = r'App\src\build_release\packet_list_dataset'
size_delay_dataset_location = r'App\src\build_release\size_delay_dataset'
dataset_in_memory_cache = False
batch_size = 64
balanced = True
num_workers = 0

def train_model():

    train_weight = 0.7
    test_weight = 0.2
    val_weight = 0.1

    train_dataset, test_dataset, val_dataset = PacketsDatapipe(size_delay_dataset_location, batch_size=batch_size, \
            weights=[train_weight, test_weight, val_weight], balanced=balanced, in_memory=dataset_in_memory_cache)

    labels = get_labels()

    train_batches = math.ceil(len(train_dataset) / batch_size)
    test_batches = math.ceil(len(test_dataset) / batch_size)
    val_batches = math.ceil(len(val_dataset) / batch_size)

    best_test_loss = float('inf')
    # while True:
    rs = MultiProcessingReadingService(num_workers=num_workers)
    train_loader = DataLoader2(train_dataset, reading_service=rs)
    test_loader = DataLoader2(test_dataset, reading_service=rs)
    val_loader = DataLoader2(val_dataset, reading_service=rs)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(dim_i=next(iter(train_dataset)).num_features, dim_h=32, dim_o=len(labels)).to(device)

    os.makedirs(ARTIFACTS_DIR)
    model = train(model, train_loader, train_batches, val_loader, val_batches, epochs=30, learning_curve=True)
    conf_matrix(model, test_loader, test_batches, labels)
    test_loss, test_acc = test(model, test_loader, test_batches)
    print(f'Test Loss: {test_loss:.2f} | Test Acc: {test_acc*100:.2f}%')
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        print(f'Better model found. Saving...')
        script_module = torch.jit.script(model)
        script_module.save(os.path.join(ARTIFACTS_DIR, 'model.pt'))

def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.jit.load('size_delay_model.pt').to(device)

    dataset = PacketsDatapipe(size_delay_dataset_location, batch_size=batch_size, balanced=balanced, in_memory=dataset_in_memory_cache)
    rs = MultiProcessingReadingService(num_workers=num_workers)
    dataset_loader = DataLoader2(dataset, reading_service=rs)

    labels = get_labels()

    train_batches = math.ceil(len(dataset) / batch_size)

    conf_matrix(model, dataset_loader, train_batches, labels)
    test_loss, test_acc = test(model, dataset_loader, train_batches)
    print(f'Dataset Loss: {test_loss:.2f} | Dataset Acc: {test_acc*100:.2f}%')

if __name__ == '__main__':
    train_model()
    # load_model()