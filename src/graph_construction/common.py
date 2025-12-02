from __future__ import annotations

"""Common helper utilities shared by graph-construction pipelines.

The functions here are intentionally dataset-agnostic.  They operate on generic
`pd.DataFrame` inputs and simple Python / NumPy / torch objects so that any
notebook-style experiment can import them without modification.
"""

import os
import shlex
from collections import defaultdict
from html import unescape
from pprint import pprint
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import degree, to_undirected
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math

__all__ = [
    "special_print",
    "parse_string_list",
    "return_density",
    'compute_assortativity_categorical',
    "build_edge_index",
    "return_data_partition_masks",
    "find_threshold_for_target_density",
    "create_node_feature_table_degree",
    "create_node_feature_table_data_user_churn",
    "save_data_object",
    "make_stratified_masks",
]


# ---------------------------------------------------------------------------
# misc helpers
# ---------------------------------------------------------------------------

def special_print(var, name: str | None = None, *, use_pprint: bool = False) -> None:
    """Pretty print a variable with clear separators – convenient for notebooks.

    Parameters
    ----------
    var
        The object to show.
    name
        Optional label printed before the value.
    use_pprint
        If ``True`` use :pyfunc:`pprint.pprint`; otherwise use ``print``.
    """
    print()  # blank line
    print("_" * 150)
    if name is not None:
        header = f"\n {name} -> \n"
        print(header)
    if use_pprint:
        pprint(var)
    else:
        print(var)
    print("-" * 150)
    print()


# ---------------------------------------------------------------------------
# string ⇄ list conversion – required when CSV is stored with list-like strings
# ---------------------------------------------------------------------------

def parse_string_list(s: str | list[str] | np.ndarray | None) -> list[str]:
    """Parse a string that represents a list of substrings.

    Accepts the typical output of ``DataFrame.to_csv`` where Python lists end up
    as strings such as ``"['foo' 'bar']"``.
    """
    if not isinstance(s, str):
        return [] if s is None else list(s)

    s = unescape(s).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]

    try:
        return shlex.split(s)
    except ValueError:
        # Malformed quotes, just fall back to whitespace split.
        return s.split()


# ---------------------------------------------------------------------------
# density / edge helpers
# ---------------------------------------------------------------------------

def return_density(n_nodes: int, n_edges: int | float) -> float:
    """Return (undirected) graph density in percent (0-100)."""
    n_max_edges = (n_nodes * (n_nodes - 1)) / 2
    return (n_edges / n_max_edges) * 100


def compute_assortativity_categorical(edge_index: torch.Tensor, y: torch.Tensor, num_classes: int = None):
    # Treat as undirected by counting both directions, per Newman (2003)
    if num_classes is None:
        num_classes = int(y.max().item()) + 1

    src, dst = edge_index
    # Remove self-loops (optional but typical for this metric)
    mask = src != dst
    src, dst = src[mask], dst[mask]

    # Make edges undirected by adding both directions
    src2 = torch.cat([src, dst], dim=0)
    dst2 = torch.cat([dst, src], dim=0)

    pairs = torch.stack([y[src2], y[dst2]], dim=1)  # [2M, 2] class pairs
    idx = pairs[:, 0] * num_classes + pairs[:, 1]
    counts = torch.bincount(idx, minlength=num_classes * num_classes).double()
    e = counts.view(num_classes, num_classes)
    e = e / e.sum()  # mixing matrix with sum=1 over all (i,j)

    a = e.sum(dim=1)  # row sums
    b = e.sum(dim=0)  # col sums (equal to a in undirected case)
    expected = (a * b).sum()
    r = (e.diag().sum() - expected) / (1 - expected + 1e-12)
    return r.item()


def build_edge_index(similarity_matrix: np.ndarray, threshold: float) -> torch.Tensor:
    """Return a *coalesced*, undirected ``edge_index`` tensor from the matrix."""
    # only upper-triangular comparison, k=1 excludes self-loops
    edges = np.nonzero(np.triu(similarity_matrix > threshold, k=1))
    edge_index = np.stack(edges, axis=0)
    edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    return to_undirected(edge_index)


# ---------------------------------------------------------------------------
# train/val/test mask split
# ---------------------------------------------------------------------------

def make_stratified_masks(
    y: np.ndarray | torch.Tensor,
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 123,
) -> Mapping[str, torch.Tensor]:
    """Return boolean masks with stratified 80/10/10 splits.

    Parameters
    ----------
    y
        1-D label tensor/array.  Must be the **same order** as rows/nodes.
    train_ratio, val_ratio
        Fractions that sum to ≤1.  Test ratio is inferred.
    seed
        Random seed to make the split reproducible across different runs and
        across different graph constructions (so long as the node order stays
        identical).
    """
    y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)

    idx = np.arange(len(y_np))

    idx_train, idx_tmp, y_train, y_tmp = train_test_split(
        idx, y_np, stratify=y_np, test_size=1 - train_ratio, random_state=seed
    )

    relative_val = val_ratio / (1 - train_ratio)  # val share within tmp
    idx_val, idx_test, _, _ = train_test_split(
        idx_tmp, y_tmp, stratify=y_tmp, test_size=1 - relative_val, random_state=seed
    )

    mask = lambda ids: torch.as_tensor(np.isin(idx, ids), dtype=torch.bool)

    return {
        "train_mask": mask(idx_train),
        "val_mask": mask(idx_val),
        "test_mask": mask(idx_test),
    }



def return_data_partition_masks(nodes_id: np.ndarray | torch.Tensor) -> Mapping[str, torch.Tensor]:
    """Return boolean masks for *row-wise* splits 80/10/10.

    ``nodes_id`` is assumed to be monotonic increasing 0..N-1.
    """
    n = len(nodes_id)
    train_len = int(n * 0.8)
    val_len = int(n * 0.1)

    train_mask = nodes_id < train_len
    val_mask = (nodes_id >= train_len) & (nodes_id < train_len + val_len)
    test_mask = nodes_id >= train_len + val_len

    # convert to torch.BoolTensor so downstream code can use them directly
    to_tensor = lambda x: torch.as_tensor(x, dtype=torch.bool)
    return {
        "train_mask": to_tensor(train_mask),
        "val_mask": to_tensor(val_mask),
        "test_mask": to_tensor(test_mask),
    }


# ---------------------------------------------------------------------------
# node features (one-hot degree)
# ---------------------------------------------------------------------------

def create_node_feature_table_degree(edge_index: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """Return log-binned degree feature matrix (shape [N, num_bins])."""
    deg = degree(edge_index[0], num_nodes=n_nodes, dtype=torch.float32)

    max_bin = int(math.ceil(math.log10(n_nodes)))  # cleaner for scalar

    # Log-scale binning: 0, 1-10, 11-100, 101-1000, 1001-10000, 10001+
    # Using log10(deg + 1) and flooring gives natural bins
    log_deg = torch.floor(torch.log10(deg + 1)).long()  # +1 to handle degree 0
    log_deg = torch.clamp(log_deg, max=max_bin)  # Cap at max_bin
    x = F.one_hot(log_deg, num_classes=max_bin+1).to(torch.float32)  
    return x

def create_node_feature_table_data_user_churn(train_feature_matrix: pd.DataFrame, masks_dict: dict) -> torch.Tensor:
    """Return Tensor (num_nodes, 16) for data informed node feature table"""
    train_feature_matrix_num = train_feature_matrix.drop(['timestamp', 'customer_id', 'churn'], axis = 'columns')
    scaler = StandardScaler()
    train_mask = masks_dict['train_mask']
    
    x = train_feature_matrix_num.values
    scaler.fit(x[train_mask.cpu().numpy()])
    x_scaled = scaler.transform(x)
    node_features = torch.tensor(x_scaled, dtype=torch.float32)
    return node_features

# ---------------------------------------------------------------------------
# threshold search – reused by every pipeline
# ---------------------------------------------------------------------------

def find_threshold_for_target_density(
    similarity_matrix: np.ndarray,
    n_nodes: int,
    target_density: float,
    *,
    tolerance: float = 1.0,
    max_iter: int = 100,
) -> float:
    """Binary-search a similarity threshold so that the resulting graph density
    is **as close as possible** to `target_density` (percentage).
    """
    low, high = 0.0, 1.0
    best_threshold = 0.0
    min_diff = float("inf")

    for _ in range(max_iter):
        mid = (low + high) / 2
        if high - low < 1e-6:
            break  # search space exhausted

        edge_index = build_edge_index(similarity_matrix, mid)
        n_edges = edge_index.size(1) / 2  # undirected – each edge counted twice
        density = return_density(n_nodes, n_edges)
        diff = abs(density - target_density)

        if diff < min_diff:
            min_diff = diff
            best_threshold = mid
        if diff <= tolerance:
            break

        if density > target_density:
            low = mid  # need fewer edges → higher threshold
        else:
            high = mid

    return best_threshold


# ---------------------------------------------------------------------------
# persist helper
# ---------------------------------------------------------------------------

def save_data_object(
    data: Data | np.ndarray,
    directory_name: str,
    threshold: float,
    density: float,
    output_base_path: str | os.PathLike = "./graphs",
    similarity_matrix_flag: bool = False,
    node_feature_degree: bool = True
) -> str:
    """Save a :class:`torch_geometric.data.Data` and return the filepath."""
    if similarity_matrix_flag:
        output_dir = os.path.join(output_base_path, directory_name)
        os.makedirs(output_dir, exist_ok=True)

        file_name = "similarity_matrix.npy"
        filepath = os.path.join(output_dir, file_name)
        np.save(filepath, data)
    else:
        if node_feature_degree:
            output_dir = os.path.join(output_base_path, directory_name, 'node_feature_degree')
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = os.path.join(output_base_path, directory_name, 'node_feature_data')
            os.makedirs(output_dir, exist_ok=True)
        
        file_name = f"thr_{threshold:.2f}__{density:.2f}%.pt"
        filepath = os.path.join(output_dir, file_name)
        torch.save(data, filepath)
    print(f"Saved: {os.path.relpath(filepath, output_base_path)}")
    return filepath
