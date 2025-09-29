import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)  # First GCN layer
        self.conv2 = GCNConv(hidden_channels, hidden_channels*2)
        self.conv3 = GCNConv(hidden_channels*2, out_channels)  # Second GCN layer

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)  # Aggregate neighbor info
        x = F.relu(x)                  # Apply ReLU activation
        x = F.dropout(x, p=0.5, training=self.training)  # Dropout for regularization
        x = self.conv2(x, edge_index)  # Aggregate neighbor info
        x = F.relu(x)                  # Apply ReLU activation
        x = F.dropout(x, p=0.5, training=self.training)  # Dropout for regularization
        x = self.conv3(x, edge_index)  # Output logit
        return x