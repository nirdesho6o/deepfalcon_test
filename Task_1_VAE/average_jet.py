import h5py
import numpy as np
import matplotlib.pyplot as plt

file_path = "data/quark-gluon.hdf5"

with h5py.File(file_path, "r") as f:

    jets = f["X_jets"]

    num_samples = 2000

    data = jets[:num_samples]      # shape (N,125,125,3)

data = np.transpose(data, (0,3,1,2))   # (N,3,125,125)

# log transform
data = np.log1p(data)

# compute average
avg_img = data.mean(axis=0)

fig, axs = plt.subplots(1,3, figsize=(9,3))

titles = ["ECAL","HCAL","Tracks"]

for i in range(3):
    axs[i].imshow(avg_img[i])
    axs[i].set_title(titles[i])
    axs[i].axis("off")

plt.tight_layout()
plt.savefig("figures/average_jet.png", dpi=300)
plt.show()