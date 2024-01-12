import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.batch import Batch
from torchdata.datapipes.iter import FileLister, FileOpener, StreamReader, InBatchShuffler, IterableWrapper, Cycler, Zipper, SampleMultiplexer, IterDataPipe, Shuffler
from torchdata.datapipes.utils import StreamWrapper
import os
import shutil
import io
import random
import numpy as np

def get_labels():
    return ["web", "video-stream", "file-transfer", "chat", "voip", "remote-desktop", "ssh", "other"]

def filestream_to_graph(file):
    data_info = np.fromfile(file, dtype=np.int64, count=4)
    x_shape_n, x_shape_m, x_data_type, edge_index_size = data_info

    if x_data_type == 0:
        x = torch.from_numpy(np.fromfile(file, dtype=np.float32, count=x_shape_n * x_shape_m).reshape(x_shape_n, x_shape_m))
    else:
        x = torch.from_numpy(np.fromfile(file, dtype=np.int8, count=x_shape_n * x_shape_m).reshape(x_shape_n, x_shape_m)).float() / 255

    edge_index = torch.from_numpy(np.fromfile(file, dtype=np.int64, count=edge_index_size).reshape(2, edge_index_size // 2))

    y = torch.from_numpy(np.fromfile(file, dtype=np.int64, count=1))

    return Data(
        x=x,
        edge_index=edge_index,
        y=y
    )

def graph_batch(batch):
    random.shuffle(batch)
    return Batch.from_data_list(batch)

def data_filter(file_name):
    return file_name.endswith('.pt')

def get_data_list(elem):
    tensors_model = torch.jit.load(io.BytesIO(elem[1]))
    tensors = list(tensors_model.parameters())
    
    data_list = []
    i = 0
    while i < len(tensors):
        if tensors[i].dtype ==  torch.int8:
            tensors[i] = tensors[i].float() / 255
        data_list.append(Data(
            x=tensors[i],
            edge_index=tensors[i+1],
            y=tensors[i+2],
        ))
        i += 3

    return data_list

def get_offsets(path):
    return np.fromfile(path, dtype=np.uint64).reshape(-1, 2)

def offest_to_filestream(offset):
    data_file, offset = offset
    file = open(data_file, mode='rb')
    file.seek(offset)
    return file

class CustomShufflerIterDataPipe(IterDataPipe):
    def __init__(self, source) -> None:
        super().__init__()
        self.source = source

    def __iter__(self):
        np.random.shuffle(self.source)
        yield from self.source
    
    def __len__(self):
        return len(self.source)

def PacketsDatapipe(root, batch_size=64, balanced=False, max_num_graphs=None, in_memory=False):
    index = get_offsets(os.path.join(root, 'offsets.bin'))
    offsets = index[:, 0]

    if balanced:
        labels = index[:, 1]
        unique_labels, label_counts = np.unique(index[:, 1], return_counts=True)

        min_label = min(label_counts)
        balanced_offsets = np.concatenate([np.random.choice(offsets[labels == label], min_label) for label in unique_labels])
        dp_input_offsets = balanced_offsets
        num_graphs = len(unique_labels) * min_label
    else:
        dp_input_offsets = offsets
        num_graphs = len(index)

    if max_num_graphs is not None:
        num_graphs = min(num_graphs, max_num_graphs)

    dp_file_name = Cycler(IterableWrapper([os.path.join(root, 'data.bin')]))

    dp = CustomShufflerIterDataPipe(dp_input_offsets)
    dp = dp.sharding_filter()
    dp = Zipper(dp_file_name, dp)
    dp = dp.map(offest_to_filestream)
    dp = StreamWrapper(dp)
    dp = dp.map(filestream_to_graph)
    if batch_size > 1:
        dp = dp.batch(batch_size=batch_size, drop_last=True)
        dp = dp.map(Batch.from_data_list)

    dp = dp.header(num_graphs).set_length(num_graphs)
    dp = dp.prefetch()

    if in_memory:
        dp = dp.in_memory_cache()

    return dp