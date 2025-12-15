import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import GINConv
from torch_geometric.utils import trim_to_layer

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout
        self.num_layers = num_layers

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

    def forward(self, x, edge_index, num_sampled_nodes_per_hop=None, num_sampled_edges_per_hop=None):
        """Forward pass with optional hierarchical sampling support.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape ``[num_nodes, in_channels]``.
        edge_index : torch.Tensor
            Graph connectivity in COO format with shape ``[2, num_edges]``.
        num_sampled_nodes_per_hop : List[int], optional
            Number of sampled nodes per hop for hierarchical sampling optimization.
        num_sampled_edges_per_hop : List[int], optional
            Number of sampled edges per hop for hierarchical sampling optimization.
        """
        for i, conv in enumerate(self.convs):
            # Apply hierarchical trimming if provided (for efficient mini-batch training)
            if num_sampled_nodes_per_hop is not None:
                x, edge_index, _ = trim_to_layer(
                    i,
                    num_sampled_nodes_per_hop,
                    num_sampled_edges_per_hop,
                    x,
                    edge_index,
                )
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu()
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x
