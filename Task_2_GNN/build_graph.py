import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph

def image_to_graph(image, label, k=5):
    image = image.numpy()

   
    total_energy = image.sum(axis=0)
    mask = total_energy > 0
    coords = np.argwhere(mask)
    
    if len(coords) == 0:
        
        return Data(x=torch.zeros((1, 6)), edge_index=torch.zeros((2, 0), dtype=torch.long), y=torch.tensor([label]))

    center_y, center_x = coords[:, 0].mean(), coords[:, 1].mean()

  
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

   
    features[:, 3:] = torch.log1p(features[:, 3:])

    edge_index = knn_graph(features[:, :2], k=k)

    data = Data(x=features, edge_index=edge_index, y=torch.tensor([label]))
    return data