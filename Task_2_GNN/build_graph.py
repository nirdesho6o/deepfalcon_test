import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph

def image_to_graph(image, label, k=5):
    image = image.numpy()

    # Sum across channels to get total energy per pixel
    total_energy = image.sum(axis=0)
    mask = total_energy > 0
    coords = np.argwhere(mask)
    
    if len(coords) == 0:
        # Failsafe for empty images
        return Data(x=torch.zeros((1, 6)), edge_index=torch.zeros((2, 0), dtype=torch.long), y=torch.tensor([label]))

    # 1. Compute center BEFORE dropping nodes, optionally weighted by energy
    center_y, center_x = coords[:, 0].mean(), coords[:, 1].mean()

    # 2. Sort by energy and take Top-K instead of random selection
    MAX_NODES = 100
    if len(coords) > MAX_NODES:
        # Get energy for all non-zero pixels
        pixel_energies = total_energy[coords[:, 0], coords[:, 1]]
        # Get indices of the top MAX_NODES energies
        top_indices = np.argsort(pixel_energies)[-MAX_NODES:] 
        coords = coords[top_indices]

    features = []
    for y, x in coords:
        dx = x - center_x
        dy = y - center_y
        r = np.sqrt(dx**2 + dy**2)

        ecal = image[0, y, x]
        hcal = image[1, y, x]
        track = image[2, y, x]

        features.append([dx, dy, r, ecal, hcal, track])

    features = torch.tensor(features, dtype=torch.float32)

    # 3. Proper HEP Normalization: Log(1 + energy)
    # This prevents extreme values from blowing up gradients without losing absolute scale
    features[:, 3:] = torch.log1p(features[:, 3:])

    # 4. Build KNN graph based on physical coordinates (dx, dy)
    edge_index = knn_graph(features[:, :2], k=k)

    data = Data(x=features, edge_index=edge_index, y=torch.tensor([label]))
    return data