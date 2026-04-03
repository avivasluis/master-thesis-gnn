from __future__ import annotations

"""Common helper utilities shared by graph-construction pipelines.

The functions here are intentionally dataset-agnostic.  They operate on generic
`pd.DataFrame` inputs and simple Python / NumPy / torch objects so that any
notebook-style experiment can import them without modification.

This module is organized into two sections:
1. Similarity matrix helpers - used by the pipeline modules (categorical, numeric, review_count)
2. Graph construction helpers - used by notebooks for building graphs from similarity matrices
"""

import os
import shlex
from collections import defaultdict
from html import unescape
from itertools import combinations
from pprint import pprint
from typing import Any, Iterable, Mapping, Sequence

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
    # Similarity matrix helpers
    "special_print",
    "parse_string_list",
    "is_list_column",
    "save_similarity_matrix",
    "normalize_matrix",
    "inspect_similar_nodes",
    # Graph construction helpers (for notebooks)
    "return_density",
    'compute_assortativity_categorical',
    "build_edge_index",
    "return_data_partition_masks",
    "find_threshold_for_target_density",
    "create_node_feature_table_degree",
    "create_node_feature_table_data_user_churn",
    "save_data_object",
    "make_stratified_masks",
    "make_temporal_masks",
    # Experiment configuration helpers
    "generate_alpha_configs",
]


# ===========================================================================
# SECTION 1: Similarity matrix helpers
# ===========================================================================

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


def is_list_column(series: pd.Series) -> bool:
    """Return True if the column holds list/array values (not string-encoded lists).

    Scalar columns (single string, number, etc.) return False. Rows that are
    all missing yield False.
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    sample = non_null.iloc[0]
    return isinstance(sample, (list, np.ndarray)) and not isinstance(sample, str)


def save_similarity_matrix(
    similarity_matrix: np.ndarray,
    labels: np.ndarray | None,
    output_dir: str | os.PathLike,
    column_name: str,
) -> tuple[str, str | None]:
    """Save similarity matrix and labels to disk.

    Parameters
    ----------
    similarity_matrix
        The N x N similarity matrix to save.
    labels
        The label vector of shape [N], or None if no labels.
    output_dir
        Base output directory.
    column_name
        Name of the column (used as subdirectory name).

    Returns
    -------
    tuple[str, str | None]
        Paths to saved similarity_matrix.npy and labels.npy (or None if no labels).
    """
    out_path = os.path.join(output_dir, column_name)
    os.makedirs(out_path, exist_ok=True)

    sim_path = os.path.join(out_path, "similarity_matrix.npy")
    np.save(sim_path, similarity_matrix)
    print(f"Saved: {sim_path}")

    labels_path = None
    if labels is not None:
        labels_path = os.path.join(out_path, "labels.npy")
        np.save(labels_path, labels)
        print(f"Saved: {labels_path}")

    return sim_path, labels_path


def normalize_matrix(M: np.ndarray) -> np.ndarray:
    """Normalize a matrix to [0, 1] range using min-max scaling.
    
    Handles NaN values gracefully using nanmin/nanmax.
    Returns the original matrix unchanged if min == max.
    """
    M_min, M_max = np.nanmin(M), np.nanmax(M)
    return (M - M_min) / (M_max - M_min) if M_max > M_min else M


def _format_feature_cell(
    val: Any,
    *,
    is_list: bool,
    max_text_length: int,
) -> str:
    """Turn a cell value into a single-line string for inspection output."""
    if is_list:
        if val is None:
            return ""
        if not isinstance(val, (list, np.ndarray, str)) and pd.isna(val):
            return ""
        if isinstance(val, str):
            val = parse_string_list(val)
        if isinstance(val, (list, np.ndarray)):
            parts = [str(x) for x in val if x is not None and str(x).strip()]
            s = " | ".join(parts) if parts else ""
        else:
            s = str(val)
    else:
        s = "" if pd.isna(val) else str(val)
    if len(s) > max_text_length:
        return s[: max_text_length - 3] + "..."
    return s


def inspect_similar_nodes(
    similarity_matrix: np.ndarray,
    query_node: int,
    df: pd.DataFrame,
    feature_column: str,
    *,
    top_n: int = 10,
    show_labels: bool = True,
    label_column: str = "churn",
    max_text_length: int = 500,
) -> pd.DataFrame:
    """Print and return the top-N most similar nodes for a query row index.

    ``similarity_matrix`` row ``i`` must correspond to ``df.iloc[i]``. The query
    node itself is excluded by **index**, not by similarity value, so duplicate
    matches with similarity 1.0 are still listed.

    Parameters
    ----------
    similarity_matrix
        Square matrix of shape ``(N, N)``.
    query_node
        Row index of the node to inspect (0-based).
    df
        DataFrame with one row per node, same order as the matrix.
    feature_column
        Column whose values are shown (the feature used to build the matrix).
    top_n
        How many neighbors to show after excluding the query node.
    show_labels
        If ``True`` and ``label_column`` exists in ``df``, include labels.
    label_column
        Label column name (e.g. ``churn``).
    max_text_length
        Truncate displayed feature strings to this many characters.

    Returns
    -------
    pd.DataFrame
        Columns: ``node_index``, ``similarity``, ``feature``; plus ``label`` if
        applicable.
    """
    if feature_column not in df.columns:
        raise ValueError(f"Column '{feature_column}' not found in DataFrame.")

    sim = np.asarray(similarity_matrix)
    if sim.ndim != 2 or sim.shape[0] != sim.shape[1]:
        raise ValueError("similarity_matrix must be a square 2-D array.")

    n = sim.shape[0]
    if len(df) != n:
        raise ValueError(
            f"DataFrame length ({len(df)}) must match matrix size ({n})."
        )
    if query_node < 0 or query_node >= n:
        raise IndexError(f"query_node must be in [0, {n - 1}], got {query_node}.")

    scores = sim[query_node]
    # Descending order by similarity; exclude query row by index (not by score).
    order = np.argsort(-scores)
    order = order[order != query_node]
    neighbor_idx = order[:top_n]
    neighbor_scores = scores[neighbor_idx]

    is_list = is_list_column(df[feature_column])

    query_text = _format_feature_cell(
        df[feature_column].iloc[query_node],
        is_list=is_list,
        max_text_length=max_text_length,
    )
    special_print(query_text, f"Query node {query_node} — {feature_column}")

    rows: list[dict[str, Any]] = []
    for j, sim_ij in zip(neighbor_idx, neighbor_scores):
        row: dict[str, Any] = {
            "node_index": int(j),
            "similarity": float(sim_ij),
            "feature": _format_feature_cell(
                df[feature_column].iloc[j],
                is_list=is_list,
                max_text_length=max_text_length,
            ),
        }
        if show_labels and label_column in df.columns:
            row["label"] = df[label_column].iloc[j]
        rows.append(row)

    out_df = pd.DataFrame(rows)
    special_print(out_df.to_string(index=False), f"Top {len(out_df)} similar nodes (excluding index {query_node})")
    return out_df


# ===========================================================================
# SECTION 2: Graph construction helpers (for notebooks)
# ===========================================================================

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


def make_temporal_masks(
    df: pd.DataFrame,
    *,
    timestamp_column: str = "timestamp",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Mapping[str, torch.Tensor]:
    """Return boolean masks with temporal 80/10/10 splits based on timestamps.

    Parameters
    ----------
    df
        DataFrame with one row per node. Must contain a timestamp column.
    timestamp_column
        Name of the timestamp column used to order nodes chronologically.
    train_ratio, val_ratio
        Fractions that sum to ≤1. Test ratio is inferred.
    """
    if timestamp_column not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_column}' not found in DataFrame.")

    n = len(df)
    if n == 0:
        raise ValueError("Cannot create temporal masks for empty DataFrame.")

    if train_ratio < 0 or val_ratio < 0 or train_ratio + val_ratio > 1:
        raise ValueError("train_ratio and val_ratio must be non-negative and sum to ≤ 1.")

    # Ensure we work on a copy to avoid modifying caller's DataFrame
    ts = pd.to_datetime(df[timestamp_column])

    # argsort gives indices that would sort timestamps in ascending order
    sorted_idx = np.argsort(ts.values)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    idx_train = sorted_idx[:train_end]
    idx_val = sorted_idx[train_end:val_end]
    idx_test = sorted_idx[val_end:]

    all_idx = np.arange(n)

    def to_mask(indices: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.isin(all_idx, indices), dtype=torch.bool)

    return {
        "train_mask": to_mask(idx_train),
        "val_mask": to_mask(idx_val),
        "test_mask": to_mask(idx_test),
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
    tolerance: float = 0.1,
    max_iter: int = 100,
    verbose: bool = True,
) -> float:
    """Binary-search a similarity threshold so that the resulting graph density
    is **as close as possible** to `target_density` (percentage).
    
    Parameters
    ----------
    similarity_matrix
        NxN similarity matrix.
    n_nodes
        Number of nodes in the graph.
    target_density
        Target density as a percentage (0-100).
    tolerance
        Stop early if density difference is within this tolerance.
    max_iter
        Maximum number of binary search iterations.
    verbose
        If True, print progress messages during search.
    """
    low, high = 0.0, 1.0
    best_threshold = 0.0
    min_diff = float("inf")

    if verbose:
        print(f"Searching for threshold to achieve target density: {target_density:.4f}")

    for _ in range(max_iter):
        mid = (low + high) / 2
        if high - low < 1e-6:
            if verbose:
                print("Search space is too small, stopping.")
            break

        edge_index = build_edge_index(similarity_matrix, mid)
        n_edges = edge_index.size(1) / 2  # undirected – each edge counted twice
        density = return_density(n_nodes, n_edges)
        diff = abs(density - target_density)

        if diff < min_diff:
            min_diff = diff
            best_threshold = mid
        if min_diff <= tolerance:
            if verbose:
                print(f"Found threshold with density difference within tolerance ({tolerance}).")
            break

        if density > target_density:
            low = mid  # need fewer edges → higher threshold
        else:
            high = mid

    if verbose:
        final_edge_index = build_edge_index(similarity_matrix, best_threshold)
        final_n_edges = final_edge_index.size(1) / 2
        final_density = return_density(n_nodes, final_n_edges)
        print(f"Search finished. Best threshold: {best_threshold:.4f} with density {final_density:.4f}")

    return best_threshold


# ---------------------------------------------------------------------------
# persist helper (legacy – for backward compatibility with notebooks)
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
    """Save a :class:`torch_geometric.data.Data` and return the filepath.
    
    Note: This function is kept for backward compatibility with notebooks.
    For new code, consider using save_similarity_matrix() for matrices.
    """
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


# ===========================================================================
# SECTION 3: Experiment configuration helpers
# ===========================================================================

def generate_alpha_configs(
    names: Sequence[str],
    n_random: int = 50,
    sparsity_alpha: float = 1.0,
    start_random_count: int = 0,
    use_random: bool = True
) -> list[dict]:
    """Generate alpha weight configurations for combining similarity matrices.
    
    Creates a list of alpha dictionaries {name: weight} covering random mixtures
    sampled from a Dirichlet distribution.
    
    Parameters
    ----------
    names
        List of feature/matrix names to generate weights for.
    n_random
        Number of random configurations to generate.
    sparsity_alpha
        Dirichlet concentration parameter:
        - < 1.0 pushes weights towards 0 (sparse mixtures)
        - > 1.0 pushes weights towards center (dense/uniform mixtures)
        - = 1.0 uniform sampling over the simplex
    start_random_count
        Starting index for naming random configurations.
        
    Returns
    -------
    list[dict]
        List of configuration dictionaries. Each dict has keys for each name
        (with float weights summing to 1.0) plus a 'type' key describing
        the configuration.
    """
    configs = []
    n = len(names)
    
    # Random mixtures using Dirichlet distribution
    if use_random:
        random_weights = np.random.dirichlet([sparsity_alpha] * n, n_random)
        
        for i, weights in enumerate(random_weights):
            config = {name: float(w) for name, w in zip(names, weights)}
            config['type'] = f'random_{i + start_random_count}'
            configs.append(config)
    else:
        # 1. Pure Strategies (Corners)
        # Useful to establish baseline performance of each feature
        for name in names:
            config = {n: 0.0 for n in names}
            config[name] = 1.0
            config['type'] = f'pure_{name}'
            configs.append(config)
            
        # 2. Pairwise Mixtures (Edges)
        # Useful to see if two features complement each other
        for name1, name2 in combinations(names, 2):
            config = {n: 0.0 for n in names}
            config[name1] = 0.5
            config[name2] = 0.5
            config['type'] = f'pair_{name1}_{name2}'
            configs.append(config)
            
        # 3. Balanced Mixture (Center)
        n = len(names)
        config = {name: 1.0/n for name in names}
        config['type'] = 'balanced'
        configs.append(config)
        
    return configs
