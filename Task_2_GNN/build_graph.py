import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph


def image_to_graph(image, label, k=8):
    """
    Convert a (3,125,125) jet image to a graph.
    """

    image = image.numpy()

    # sum channels to find active pixels
    mask = image.sum(axis=0) > 0
    coords = np.argwhere(mask)

    features = []

    for y, x in coords:
        ecal = image[0, y, x]
        hcal = image[1, y, x]
        track = image[2, y, x]

        features.append([x, y, ecal, hcal, track])

    features = torch.tensor(features, dtype=torch.float)

    # build kNN graph using spatial coordinates
    edge_index = knn_graph(features[:, :2], k=k)

    data = Data(x=features, edge_index=edge_index, y=torch.tensor([label]))

    return data