import h5py
from sympy import limit
import torch
from torch_geometric.data import Dataset
from build_graph import image_to_graph
import os
CACHE_FILE = "jet_graph_cache.pt"

class JetGraphDataset(Dataset):

    def __init__(self, file_path, limit=2000):

        if os.path.exists(CACHE_FILE):

            print("Loading cached graphs...")
            self.graphs = torch.load(CACHE_FILE)

        else:

            print("Precomputing graphs...")

            self.graphs = []

            for i in range(limit):

                image = torch.tensor(self.images[i], dtype=torch.float32).permute(2,0,1)
                label = int(self.labels[i])

                graph = image_to_graph(image, label)

                self.graphs.append(graph)
            torch.save(self.graphs, CACHE_FILE)

        print("Finished building graphs.")


    def len(self):
        return self.limit


    def get(self, idx):
        return self.graphs[idx]