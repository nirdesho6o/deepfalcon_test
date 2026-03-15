import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import torchvision.transforms.functional as TF


class JetDataset(Dataset):

    def __init__(self, file_path):

        self.file = h5py.File(file_path, "r")

        self.images = self.file["X_jets"]
        self.labels = self.file["y"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        x = self.images[idx]            # (125,125,3)

        x = np.transpose(x, (2,0,1))    # (3,125,125)

        x = x.astype(np.float32)

        # log normalization
        x = np.log1p(x)

        x = torch.tensor(x)

        # resize to 128×128
        x = TF.resize(x, [128,128])

        # scale to [0,1]
        x = x / x.max()

        return x

if __name__ == "__main__":

    dataset = JetDataset("../data/quark-gluon.hdf5")

    print("Dataset size:", len(dataset))

    sample = dataset[0]

    print("Sample shape:", sample.shape)