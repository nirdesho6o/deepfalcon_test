import h5py
import torch
import numpy as np
import os
from tqdm import tqdm
from torch_geometric.data import Dataset
from build_graph import image_to_graph


class JetGraphDataset(Dataset):

    def __init__(self, file_path, limit=500):

        super().__init__()

        CACHE_FILE = "jet_graph_cache.pt"

        if os.path.exists(CACHE_FILE):

            print("Loading cached graphs...")
            self.graphs = torch.load(CACHE_FILE)

        else:

            print("Loading subset from HDF5...")

            with h5py.File(file_path, "r") as f:
                images = f["X_jets"][:limit]
                labels = f["y"][:limit]

            print("Building graphs...")

            self.graphs = []

            for i in tqdm(range(limit)):

                image = torch.tensor(images[i], dtype=torch.float32).permute(2,0,1)
                label = int(labels[i])

                graph = image_to_graph(image, label)

                self.graphs.append(graph)

            torch.save(self.graphs, CACHE_FILE)
            print("Saved cache.")


    def len(self):
        return len(self.graphs)


    def get(self, idx):
        return self.graphs[idx]