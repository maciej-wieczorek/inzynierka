import torch
from torch_geometric.data import Data, InMemoryDataset
from torchdata.datapipes.iter import FileLister, FileOpener, StreamReader
import os
import shutil
import io

def get_labels():
    return ["web", "video-stream", "file-transfer", "chat", "voip", "remote-desktop", "ssh", "other"]

class PacketsDataset(InMemoryDataset):
    def __init__(self, root, data_location, transform=None, pre_transform=None, pre_filter=None):
        self.data_location = data_location
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        if os.path.exists(self.data_location):
            return list(filter(lambda file_name: file_name.endswith('.pt'), os.listdir(self.data_location)))
        else:
            return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        for file_name in self.raw_file_names:
            shutil.copy(os.path.join(self.data_location, file_name), os.path.join(self.raw_dir, file_name))

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        for raw_path in self.raw_paths:
            tensors_model = torch.jit.load(raw_path)
            tensors = list(tensors_model.parameters())
            
            i = 0
            while i < len(tensors):
                data_list.append(Data(
                    x=tensors[i],
                    edge_index=tensors[i+1],
                    y=tensors[i+2],
                ))
                i += 3

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])


def data_filter(file_name):
    return file_name.endswith('.pt')

def get_data_list(elem):
    tensors_model = torch.jit.load(io.BytesIO(elem[1]))
    tensors = list(tensors_model.parameters())
    
    data_list = []
    i = 0
    while i < len(tensors):
        data_list.append(Data(
            x=tensors[i].float() / 255,
            edge_index=tensors[i+1],
            y=tensors[i+2],
        ))
        i += 3

    return data_list

def PacketsDatapipe(root, batch_size):
    dp = FileLister(root=root).filter(data_filter)
    dp = FileOpener(dp, mode='rb')
    dp = StreamReader(dp)
    dp = dp.map(get_data_list).unbatch()
    dp = dp.shuffle()
    dp = dp.batch_graphs(batch_size=batch_size)
    # dp = dp.in_batch_shuffle()
    # dp = dp.in_memory_cache()

    dp = dp.set_length(sum(map(lambda x : int(x.split('-')[1].split('_')[0]), list(filter(data_filter, os.listdir(root))))))

    return dp