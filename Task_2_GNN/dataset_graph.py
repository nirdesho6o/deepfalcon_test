import h5py
import torch
from torch_geometric.data import Dataset
from build_graph import image_to_graph


class JetGraphDataset(Dataset):

    def __init__(self, file_path, limit=5000):

        self.file = h5py.File(file_path, "r")

        self.images = self.file["X_jets"]
        self.labels = self.file["y"]

        self.limit = limit

    def len(self):
        return self.limit

    def get(self, idx):

        image = torch.tensor(self.images[idx]).permute(2,0,1)
        label = int(self.labels[idx])

        graph = image_to_graph(image, label)

        return graph