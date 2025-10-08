import numpy as np 
import pandas as pd 
import ast
import argparse
import os
import polars as pl

import shlex
from sentence_transformers import SentenceTransformer

from typing import List, Optional
import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.utils import degree, to_undirected
from torch_geometric.data import Data 

class GloveTextEmbedding:
    def __init__(self, device: Optional[torch.device] = None):
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d",
            device=device
        )

    def __call__(self, sentences: List[str]) -> Tensor:
        return torch.from_numpy(self.model.encode(sentences, normalize_embeddings=True))

    def similarity(self, embeddings):
      return self.model.similarity(embeddings, embeddings)

    def single_embedding(self, embeddings):
        return torch.sum(embeddings, 0)


def special_print(var, name):
    print('-*'*100)
    print(f'\n {name} -> \n{var}\n')
    print('-*'*100)
    print()


def parse_string_list(s):
    """
    Parses a string that represents a list of strings.
    Example input: "['String one' 'String two']"
    Output: ['String one', 'String two']
    """
    if not isinstance(s, str):
        return []
    
    s = s.strip()
    # Remove outer brackets
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]
    
    # shlex.split is good for parsing shell-like quoted strings
    # It can handle strings with spaces if they are quoted.
    try:
        return shlex.split(s)
    except ValueError:
        # This can happen with unclosed quotes, for example.
        return []

"""
def create_data_edge_matrix(train, column_name, text_embedder):
    train_exploded = train.explode(column_name)
    train_exploded = train_exploded.dropna(subset=[column_name])

    #special_print(train_exploded, 'train_exploded')

    all_titles = train_exploded[column_name].tolist()

    all_embeddings = text_embedder(all_titles)

    #special_print(all_embeddings, 'all_embeddings')
    #special_print(type(all_embeddings), 'type(all_embeddings)')
    #special_print(all_embeddings.size(), 'all_embeddings.size()')


    train_exploded['embeddings'] = list(all_embeddings.cpu().numpy())

    aggregated_embeddings = train_exploded.groupby(train_exploded.index)['embeddings'].apply(
        lambda embeddings: np.sum(embeddings.values, axis=0)
    )

    #special_print(aggregated_embeddings, 'aggregated_embeddings')
    #special_print(type(aggregated_embeddings), 'type(aggregated_embeddings)')
    #special_print(aggregated_embeddings.size, 'aggregated_embeddings.size')

    embedding_dim = text_embedder.model.get_sentence_embedding_dimension()
    zero_embedding = np.zeros(embedding_dim)

    # Create a Series of zero vectors with the same index as the original dataframe.
    final_embeddings = pd.Series([zero_embedding] * len(train), index=train.index)

    # Update this series with the computed embeddings. Any row that had no titles
    # will keep its zero vector.
    final_embeddings.update(aggregated_embeddings)
    numpy_embeddings = np.stack(final_embeddings.values)
    tensor_embeddings = torch.from_numpy(numpy_embeddings)

    special_print(tensor_embeddings, 'tensor_embeddings')
    special_print(type(tensor_embeddings), 'type(tensor_embeddings)')
    special_print(tensor_embeddings.size(), 'tensor_embeddings.size()')

    return tensor_embeddings
"""

def create_data_edge_matrix(train: pd.DataFrame,
                            column_name: str,
                            text_embedder,
                            device) -> torch.Tensor:
    """
    Vectorised version that keeps the data on-device and
    aggregates with a single scatter_add.
    """
    # 1. Collect the lists per row (no explode)
    lists = train[column_name].fillna('').tolist()

    # 2. Flatten them and remember how many tokens each row has
    # Ensure the tensor lives on the same device as the embeddings to avoid
    # "Expected all tensors to be on the same device" runtime errors.
    lengths = torch.tensor([len(lst) for lst in lists], dtype=torch.long, device=device)
    flat_titles = [title for sub in lists for title in sub]

    # Short-circuit if the whole column is empty
    if not flat_titles:
        dim = text_embedder.model.get_sentence_embedding_dimension()
        return torch.zeros((len(train), dim))

    # 3. Encode every title – stays on the same device the model uses
    embeddings = text_embedder(flat_titles).to(device)              # (N_flat, d)

    # 4. Build an index that maps every title back to its row
    row_idx = torch.repeat_interleave(
        torch.arange(len(train), device=embeddings.device),
        lengths
    )                                                    # (N_flat, )

    # 5. Aggregate with one scatter_add
    agg = torch.zeros(len(train), embeddings.size(1),
                      device=embeddings.device, dtype=embeddings.dtype)
    agg.scatter_add_(0, row_idx.unsqueeze(-1).expand_as(embeddings), embeddings)

    #special_print(agg, 'agg')
    #special_print(type(agg), 'type(agg)')
    #special_print(agg.size(), 'agg.size()')

    # rows that had no titles already contain zeros, nothing else to do
    return agg                                           # (n_rows, d)

"""
def create_edge_index(data_edge_matrix, threshold, text_embedder):
    similarity_matrix = text_embedder.similarity(data_edge_matrix)

    edges = np.nonzero(np.triu(similarity_matrix > threshold, k=1))
    edge_index = np.stack(edges, axis=0)
    edge_index = torch.tensor(edge_index, dtype = torch.long)
    edge_index = to_undirected(edge_index)
    #print(f'edge_index: \n{edge_index}\n')
    #print(f'type(edge_index): \n{type(edge_index)}\n')
    #print(f'edge_index.shape: \n{edge_index.shape}\n')
    #print(f'threshold: \n{threshold}\n')
    return edge_index
"""


def build_edge_index_from_cosine(x: torch.Tensor,
                                 threshold: float,
                                 symmetrize: bool = True) -> torch.Tensor:
    """
    x : (n, d)  – row-wise node/text embeddings (need **not** be unit-normed)
    threshold : keep edges with cos-sim > threshold
    symmetrize: if True make the graph undirected by mirroring the upper-tri.

    Returns a 2×E tensor in COO format suitable for `torch_geometric`.
    """
    # 1. Normalize so that dot product == cosine similarity
    x = F.normalize(x, dim=-1)

    # 2. Dense cosine similarity  – works for n ≤ ~10 000 on 16 GB GPU/host RAM
    sim = x @ x.T                                    # (n, n)

    # 3. Mask the strictly upper triangular part & threshold
    mask = torch.triu(sim, diagonal=1) > threshold   # (n, n) boolean

    # 4. Indices where mask == True → edge list
    edge_index = mask.nonzero(as_tuple=False).T      # shape (2, E)

    # 5. Optional: mirror to get an undirected graph
    return to_undirected(edge_index) if symmetrize else edge_index


def create_node_feature_table(edge_index, n_nodes):
    degrees = degree(edge_index[0], num_nodes=n_nodes, dtype=torch.long)
    x = F.one_hot(degrees).to(torch.float)
    return x


def return_data_partition_masks(nodes_id):
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


def return_density(n_nodes, n_edges):
    n_max_edges = (n_nodes*(n_nodes-1))/2
    density = (n_edges / n_max_edges) * 100
    return density


def save_data_object(data, directory_name, thrs, density, output_base_path):
    output_dir = os.path.join(output_base_path, directory_name)
    os.makedirs(output_dir, exist_ok = True)

    file_name = f'thr_{thrs}__{density:.1f}%.pt'

    output_path = os.path.join(output_dir, file_name)
    torch.save(data, output_path)
    print(f'Saved: {directory_name}/{file_name}')


def main(args):
    #print(f'\n args -> \n {args}')
    #special_print(args.train, 'train')
    #special_print(args.train.info(), 'train.info()')
    #special_print(args.train[args.column_data][0], f'data for a row in train[{args.column_data}]')
    #special_print(type(args.train[args.column_data][0]), f'type of data for a row in train[{args.column_data}]')
    #special_print(len(args.train[args.column_data][0]), f'len for a row in train[{args.column_data}]')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_embedder = GloveTextEmbedding(device=device)

    #special_print(device, 'device')

    data_edge_matrix = create_data_edge_matrix(args.train, args.column_data, text_embedder, device)

    edge_index = build_edge_index_from_cosine(data_edge_matrix, args.thr)

    n_nodes = len(args.train)
    n_edges = len(edge_index[0])/2

    x = create_node_feature_table(edge_index, n_nodes)
    y = torch.tensor(args.train['churn'].values, dtype=torch.long)
    masks = return_data_partition_masks(args.train.index)
    density = return_density(n_nodes, n_edges)

    data = Data(x = x, edge_index = edge_index, y = y, masks = masks, density = round(density,2))

    save_data_object(data, f'{args.column_data}', args.thr, density, args.output_path)
    special_print(data, 'data')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process expanded training data into a graph using a feature.")
    parser.add_argument('output_path', type=str, help="Directory for output data")
    parser.add_argument('thr', type=float, help="Threshold to generate the edges on the graph")
    parser.add_argument('column_data', type=str, help="Name of the column data to use")     

    parser.add_argument('--base_data_path', type=str, default=r'data\2_intermediate\rel-amazon\tasks\user-churn', help="Directory base for all the expanded files")
    parser.add_argument('--kaggle', action='store_true', help="Flag to indicate whether the script will run on kaggle or not")
    parser.add_argument('--use_subsection', action='store_true', help="Flag to indicate whether generate graph using only subsection")
    parser.add_argument('--sample_size', type=int, default=5, help="Number of samples to process from the train set")

    args = parser.parse_args()

    train_file = os.path.join(args.base_data_path, args.column_data, 'expanded_train.csv')

    if args.use_subsection:
        args.train = pd.read_csv(train_file, converters={args.column_data: parse_string_list}, index_col=0).head(args.sample_size)
    else:
        args.train = pd.read_csv(train_file, converters={args.column_data: parse_string_list}, index_col=0)

    os.makedirs(args.output_path, exist_ok=True)

    main(args)