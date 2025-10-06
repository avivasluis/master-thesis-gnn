import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import argparse
import os

import torch
import torch.nn.functional as F


from torch_geometric.utils import degree, to_undirected
from torch_geometric.data import Data 

from build_graph_categories_string import return_density

def create_edge_index(train, threshold):
    mlb = MultiLabelBinarizer()
    one_hot_encoded_cats = mlb.fit_transform(train['categories_expanded'])
    similarity_matrix = cosine_similarity(one_hot_encoded_cats)

    edges = np.nonzero(np.triu(similarity_matrix > threshold, k=1))
    edge_index = np.stack(edges, axis=0)
    edge_index = torch.tensor(edge_index, dtype = torch.long)
    edge_index = to_undirected(edge_index)
    print(f'edge_index: \n{edge_index}\n')
    print(f'type(edge_index): \n{type(edge_index)}\n')
    print(f'edge_index.shape: \n{edge_index.shape}\n')
    print(f'threshold: \n{threshold}\n')
    return edge_index

def create_node_feature_table(edge_index, n_nodes):
    degrees = degree(edge_index[0], num_nodes=n_nodes, dtype=torch.long)
    x = F.one_hot(degrees).to(torch.float)
    return x

def main(args):
    # Construct paths based on the dataset argument
    data_path = os.path.join('data','2_intermediate','rel-amazon','tasks','user-churn','single_categories_string.csv')
    output_path = os.path.join('data','3_processed','rel-amazon','tasks','user-churn','common_purchased_categories_one_hot_vector')

    # Pre-Process the data
    train = pd.read_csv(data_path)
    train['categories_expanded'] = train['categories_expanded'].apply(ast.literal_eval)
    print(f'train: \n{train}\n')

    n_nodes = len(train)

    edge_index = create_edge_index(train, args.threshold)
    density = return_density(n_nodes, len(edge_index[0])/2)
    print(f'density: \n{density:.4f}% \n')
    x = create_node_feature_table(edge_index, n_nodes)
    y = torch.tensor(train['churn'].values, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)

    print(f'Data object: \n{data}\n')

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Save the Data object
    torch.save(data, os.path.join(output_path, f'data_thres_{args.threshold}.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create PyG Data Object")
    parser.add_argument('--threshold', type=float, default=0.9, help=" Threshold of common categories ")
    args = parser.parse_args()
    
    main(args)