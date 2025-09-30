from torch_geometric.utils import degree
from torch_geometric.data import Data

import torch
import torch.nn.functional as F


def create_node_feature_table(edge_index, n_nodes):
    degrees = degree(edge_index[0], num_nodes=n_nodes, dtype=torch.long)
    x = F.one_hot(degrees)
    return x


def get_subgraph_first_n_nodes(data: Data, n_nodes: int):
    """
    Creates a subgraph containing the first N nodes of a given graph.

    Args:
        data (torch_geometric.data.Data): The original graph data object.
        n_nodes (int): The number of nodes to include in the subgraph.

    Returns:
        torch_geometric.data.Data: A new data object representing the subgraph.
    """
    if not 10 < n_nodes <= data.num_nodes:
        raise ValueError(f"n_nodes must be between 10 and {data.num_nodes}, but got {n_nodes}")

    # Create a new Data object for the subgraph
    subgraph_data = Data()

    # Filter edges to include only those within the first n_nodes
    edge_mask = (data.edge_index[0] < n_nodes) & (data.edge_index[1] < n_nodes)
    edge_index = data.edge_index[:, edge_mask]

    degrees = degree(edge_index[0], num_nodes=n_nodes, dtype=torch.long)
    subgraph_data.x = F.one_hot(degrees)

    subgraph_data.y = data.y[:n_nodes]

    subgraph_data.edge_index = edge_index
    
    # Recalculate masks for the new subgraph size (80% train, 10% val, 10% test)
    n_train = int(n_nodes * 0.8)
    n_val = int(n_nodes * 0.1)
    if n_val == 0: n_val = 1
    n_test = n_nodes - n_train - n_val
    if n_test <= 0:
        n_test = 1
        n_train = n_nodes - n_val - n_test

    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)

    train_mask[:n_train] = True
    val_mask[n_train : n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    
    subgraph_data.masks = {
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
    }

    return subgraph_data