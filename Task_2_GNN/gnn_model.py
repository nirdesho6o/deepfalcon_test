import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_mean_pool


class JetGNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = GraphConv(5, 64)
        self.conv2 = GraphConv(64, 64)

        self.lin1 = nn.Linear(64, 32)
        self.lin2 = nn.Linear(32, 2)

    def forward(self, x, edge_index, batch):

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        x = global_mean_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return x