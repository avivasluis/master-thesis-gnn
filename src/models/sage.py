import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import trim_to_layer

class GraphSAGE(torch.nn.Module):
    """GraphSAGE model.

    Parameters
    ----------
    in_channels : int
        Dimensionality of input node features.
    hidden_channels : int
        Hidden dimension size.
    out_channels : int
        Dimensionality of output layer (e.g., number of classes).
    num_layers : int, optional (default=2)
        Number of SAGEConv layers to use. If ``num_layers == 1`` the model
        consists of a single convolution from ``in_channels`` to ``out_channels``.
    dropout : float, optional (default=0.5)
        Dropout probability applied after every hidden layer.
    aggr : str, optional (default="mean")
        Aggregation method to use in ``SAGEConv`` (e.g., "mean", "max", "sum").
    """

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 aggr: str = "mean"):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.dropout = dropout
        self.num_layers = num_layers

        if num_layers == 1:
            self.convs.append(SAGEConv(in_channels, out_channels, aggr=aggr))
        else:
            # Input layer
            self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))

            # Hidden layers
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))

            # Output layer
            self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggr))

    def forward(self, x, edge_index, num_sampled_nodes_per_hop=None, num_sampled_edges_per_hop=None):
        """Forward pass.

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

