import h5py
import torch
from torch.utils.data import Dataset, DataLoader

class JetImageDataset(Dataset):
    def __init__(self, file_path, limit=1000):
        print(f"Loading {limit} jets from HDF5...")
        with h5py.File(file_path, "r") as f:
            # The raw shape is (N, 125, 125, 3)
            self.images = f["X_jets"][:limit]
            self.labels = f["y"][:limit]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

        img = torch.log1p(img)
        
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return img, label