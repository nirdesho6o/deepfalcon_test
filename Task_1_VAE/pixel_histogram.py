import h5py
import numpy as np
import matplotlib.pyplot as plt

file_path = "../data/quark-gluon.hdf5"

with h5py.File(file_path, "r") as f:
    
    jets = f["X_jets"]
    
    num_samples = 5000
    
    data = jets[:num_samples]      # shape (N,125,125,3)

# flatten all pixels
pixels = data.flatten()

# log transform (same preprocessing)
pixels = np.log1p(pixels)

plt.figure(figsize=(6,4))

plt.hist(pixels, bins=100, log=True)

plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency (log scale)")
plt.title("Pixel Value Distribution in Jet Images")

plt.tight_layout()

plt.savefig("figures/pixel_distribution.png", dpi=300)

plt.show()