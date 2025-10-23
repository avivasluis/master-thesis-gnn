import torch
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import GINConv
import torch.nn.functional as F

class GIN(torch.nn.Module):
    """Graph Isomorphism Network (GIN) model with three GINConv layers and a final linear layer.

    This implementation mirrors the structure of the existing `GCN` class: two hidden
    representation stages with dropout/activation, followed by an output layer.
    Forward signature stays consistent with other models in `src/models` so that the
    training utilities can be reused without modification.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()

        # First GIN layer
        mlp1 = Sequential(
            Linear(in_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels)
        )
        self.conv1 = GINConv(mlp1)

        # Second GIN layer (doubling hidden dims like existing GCN/MLP for fairness)
        mlp2 = Sequential(
            Linear(hidden_channels, hidden_channels * 2),
            ReLU(),
            Linear(hidden_channels * 2, hidden_channels * 2)
        )
        self.conv2 = GINConv(mlp2)

        # Linear readout to out_channels (e.g. 1 for binary classification)
        self.lin = Linear(hidden_channels * 2, out_channels)

    def forward(self, x, edge_index):
        """Standard forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node features with shape [num_nodes, in_channels].
        edge_index : torch.Tensor
            Edge indices in COO format with shape [2, num_edges].
        """
        # First layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Second layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Output layer
        x = self.lin(x)
        return x
