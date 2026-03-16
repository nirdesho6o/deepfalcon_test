import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph


def image_to_graph(image, label, k=8):
    """
    Convert a (3,125,125) jet image to a graph.
    """

    image = image.numpy()

    mask = image.sum(axis=0) > 0
    coords = np.argwhere(mask)
    MAX_NODES = 80

    if len(coords) > MAX_NODES:
        idx = np.random.choice(len(coords), MAX_NODES, replace=False)
        coords = coords[idx]

    features = []

    for y, x in coords:
        ecal = image[0, y, x]
        hcal = image[1, y, x]
        track = image[2, y, x]

        features.append([x, y, ecal, hcal, track])

    features = torch.tensor(features, dtype=torch.float)

    edge_index = knn_graph(features[:, :2], k=5)

    data = Data(x=features, edge_index=edge_index, y=torch.tensor([label]))

    return data