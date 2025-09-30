import os
import argparse
import pandas as pd
import numpy as np

from collections import defaultdict
from itertools import combinations
import ast

import torch
import torch.nn.functional as F

from torch_geometric.utils import degree, to_undirected
from torch_geometric.data import Data 


def return_density(n_nodes, n_edges):
    n_max_edges = (n_nodes*(n_nodes-1))/2
    graph_density = (n_edges/n_max_edges)*100
    return graph_density


def create_edge_index(train_df, THRESHOLD):
    # Build the inverted index
    category_to_nodes = defaultdict(list)

    for node_id, categories in zip(train_df['node_id'], train_df['categories']):
        for category in categories:
            category_to_nodes[category].append(node_id)

    pair_counts = defaultdict(int)
    for category, nodes in category_to_nodes.items():
        if len(nodes) > 1:
            for u, v in combinations(nodes, 2):
                # Ensure the pair is always in the same order (e.g., smaller id first)
                pair = tuple(sorted((u, v)))
                pair_counts[pair] += 1
    

    # Create edges based on a threshold of common categories
    edges = set()
    for pair, count in pair_counts.items():
        if count >= THRESHOLD:
            edges.add(pair)

    # print(f"Using a threshold of {THRESHOLD}, generated {len(edges)} edges.")
    # print("Edges:", edges)

    # Process the edge_index structure
    edge_array = np.array(list(edges))
    edge_index = to_undirected(torch.from_numpy(edge_array.T).to(torch.long))
    return edge_index


def create_node_feature_table(edge_index, n_nodes):
    degrees = degree(edge_index[0], num_nodes=n_nodes, dtype=torch.long)
    x = F.one_hot(degrees)
    return x

def data_partition(nodes_id):
    train_len = int(len(nodes_id)*0.8)
    val_len = int(len(nodes_id)*0.1)

    train_mask = nodes_id < train_len
    val_mask = (nodes_id >= train_len) & (nodes_id < train_len + val_len)
    test_mask = nodes_id >= train_len + val_len

    masks = {
    'train_mask': train_mask,
    'val_mask': val_mask,
    'test_mask': test_mask
    }

    return masks


def main(args):
    # Construct paths based on the dataset argument
    data_path = os.path.join('data','2_intermediate','rel-amazon','tasks','user-churn','single_categories_string.csv')
    output_path = os.path.join('data','3_processed','rel-amazon','tasks','user-churn')

    # Pre-Process the data
    train = pd.read_csv(data_path)
    train['categories'] = train['categories_expanded'].apply(ast.literal_eval)
    train = train.drop(['product_id', 'categories_expanded'], axis = 1)
    train['node_id'] = train.index
    train = train[['node_id','timestamp', 'customer_id', 'categories', 'churn']]

    # print(f'train_df: {train}')
    n_nodes = len(train)

    edge_index = create_edge_index(train, args.threshold)
    x = create_node_feature_table(edge_index, n_nodes)
    y = torch.tensor(train['churn'].values, dtype=torch.long)
    masks = data_partition(train.index)
    data = Data(x=x.to(torch.float), edge_index=edge_index, y=y, masks = masks)

    # print(f'data object: {data}')

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Save the Data object
    torch.save(data, os.path.join(output_path, f'data_thres_{args.threshold}.pt'))
    # print(f"Data object saved to {os.path.join(output_path, f'data_thres_{args.threshold}.pt')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create PyG Data Object")
    parser.add_argument('--threshold', type=int, default=1, help=" Threshold of common categories ")
    args = parser.parse_args()
    
    main(args)