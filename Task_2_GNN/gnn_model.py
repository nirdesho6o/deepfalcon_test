import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm, global_max_pool, global_mean_pool

class JetGNN(nn.Module):
    def __init__(self):
        super().__init__()

    
        self.conv1 = SAGEConv(6, 64)
        self.bn1 = BatchNorm(64)
        
        self.conv2 = SAGEConv(64, 64)
        self.bn2 = BatchNorm(64)
        
        self.conv3 = SAGEConv(64, 64)
        self.bn3 = BatchNorm(64)

        
        self.lin1 = nn.Linear(128, 64)
        self.lin2 = nn.Linear(64, 2)

        
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x, edge_index, batch):
        
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        # Layer 3
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)


        x_max = global_max_pool(x, batch)
        x_mean = global_mean_pool(x, batch)

        x = torch.cat([x_max, x_mean], dim=1) 


        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)

        return x