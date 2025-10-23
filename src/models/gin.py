import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import GINConv

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout

        if num_layers == 1:
            nn = Sequential(
                Linear(in_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, out_channels),
            )
            self.convs.append(GINConv(nn))
        else:
            # Input layer
            nn = Sequential(
                Linear(in_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINConv(nn))

            # Hidden layers
            for _ in range(num_layers - 2):
                nn = Sequential(
                    Linear(hidden_channels, hidden_channels),
                    ReLU(),
                    Linear(hidden_channels, hidden_channels),
                )
                self.convs.append(GINConv(nn))
            
            # Output layer MLP for GINConv
            nn = Sequential(
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, out_channels),
            )
            self.convs.append(GINConv(nn))

    def forward(self, x, edge_index, batch=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu()
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x
