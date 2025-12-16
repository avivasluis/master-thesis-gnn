from __future__ import annotations

"""Build homogeneous graphs from a scalar *num_reviews* column using
absolute difference as the similarity measure.

This method is fully vectorized (no Python loops) enabling efficient
graph generation for 50K+ nodes.

Similarity formula: 1 / (1 + |num_reviews_i - num_reviews_j|)
"""

from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from .common import (
    build_edge_index,
    create_node_feature_table_degree,
    create_node_feature_table_data_user_churn,
    find_threshold_for_target_density,
    return_data_partition_masks,
    return_density,
    compute_assortativity_categorical,
    special_print,
    make_stratified_masks,
)

__all__ = ["build_graph"]


# ---------------------------------------------------------------------------
# similarity matrix – vectorized absolute difference
# ---------------------------------------------------------------------------

def _create_similarity_matrix(
    num_reviews: np.ndarray,
    dtype: type = np.float32,
) -> np.ndarray:
    """Compute similarity matrix based on review count difference.

    Fully vectorized using numpy broadcasting.
    Memory usage: n² × sizeof(dtype) bytes.

    Parameters
    ----------
    num_reviews
        1-D array of review counts per node (shape [n,]).
    dtype
        NumPy dtype for the output matrix. Use float32 for memory efficiency.

    Returns
    -------
    np.ndarray
        Symmetric similarity matrix of shape [n, n].
    """
    num_reviews = np.asarray(num_reviews, dtype=dtype).ravel()
    n = len(num_reviews)
    print(f"Computing {n}x{n} similarity matrix ({n*n*np.dtype(dtype).itemsize / 1e9:.2f} GB)...")

    # Broadcasting: (n, 1) - (1, n) → (n, n)
    diff = np.abs(num_reviews[:, None] - num_reviews[None, :])
    sim = (1.0 / (1.0 + diff)).astype(dtype)
    return sim


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def build_graph(
    df: pd.DataFrame,
    *,
    label_column: str = "churn",
    item_list_column: str = "num_reviews",
    target_densities: Iterable[float] = (15, 10, 7, 4),
    density_tol: float = 1.0,
    max_iter: int = 100,
    verbose: bool = True,
    dataset: str = "rel-amazon",
    task: str = "user-churn",
    time_window: str = "-6mo",
    feature_df: pd.DataFrame,
    dtype: type = np.float32,
    return_only_sim_matrix: bool = False
) -> tuple[list[Data], list[Data], np.ndarray]:
    """Build graph from review count similarity.

    This pipeline extracts `num_reviews` from `feature_df` and computes
    pairwise similarity as 1/(1+|diff|). It is optimized for large graphs.

    Parameters
    ----------
    df
        Input DataFrame – one row per node. Used for label extraction and
        structure validation.
    label_column
        Name of the column that holds the node labels (`y`).
    item_list_column
        Column name for similarity computation. For this pipeline, this is
        expected to be "num_reviews" and is extracted from `feature_df`.
    target_densities
        Iterable of desired densities (percentage). One graph is produced
        for each density.
    density_tol
        Acceptable absolute error in the achieved density.
    max_iter
        Maximum binary-search iterations per density.
    verbose
        Whether to print progress information.
    dataset, task, time_window
        Metadata stored in the output Data objects.
    feature_df
        DataFrame holding feature data for every node, including `num_reviews`.
    dtype
        NumPy dtype for similarity matrix. Default float32 saves memory.

    Returns
    -------
    tuple[list[Data], list[Data], np.ndarray]
        (data_degree_objects, data_features_objects, similarity_matrix)
    """
    n_nodes = len(df)

    if len(feature_df) != n_nodes:
        raise ValueError(
            f"Feature DataFrame length {len(feature_df)} does not match "
            f"Graph DataFrame length {n_nodes}"
        )

    # Verify alignment of customer_id if present
    if "customer_id" in df.columns and "customer_id" in feature_df.columns:
        if not np.array_equal(df["customer_id"].values, feature_df["customer_id"].values):
            raise ValueError(
                "Structure mismatch: 'customer_id' columns do not align "
                "between input DF and feature DF."
            )

    # Extract num_reviews from feature_df
    if item_list_column not in feature_df.columns:
        raise ValueError(
            f"Column '{item_list_column}' not found in feature_df. "
            f"Available columns: {list(feature_df.columns)}"
        )

    num_reviews = feature_df[item_list_column].values
    if verbose:
        special_print(
            {
                "n_nodes": n_nodes,
                "num_reviews_min": num_reviews.min(),
                "num_reviews_max": num_reviews.max(),
                "num_reviews_mean": num_reviews.mean(),
            },
            name="Review count stats",
        )

    # Vectorized similarity computation
    similarity_matrix = _create_similarity_matrix(num_reviews, dtype=dtype)
    if verbose:
        special_print(similarity_matrix.shape, "similarity_matrix.shape")

    if return_only_sim_matrix:
        return similarity_matrix
    else:
        # Labels and masks
        y = (
            torch.as_tensor(df[label_column].values, dtype=torch.long)
            if label_column in df.columns
            else None
        )
        masks = (
            make_stratified_masks(y)
            if y is not None
            else return_data_partition_masks(np.arange(n_nodes))
        )

        # Compute static features once
        x_features = create_node_feature_table_data_user_churn(feature_df, masks)

        data_degree_objects: list[Data] = []
        data_features_objects: list[Data] = []

        for target in target_densities:
            thr = find_threshold_for_target_density(
                similarity_matrix,
                n_nodes,
                target,
                tolerance=density_tol,
                max_iter=max_iter,
            )
            edge_index = build_edge_index(similarity_matrix, thr)
            n_edges = edge_index.size(1) / 2
            density = return_density(n_nodes, n_edges)
            assortativity = (
                compute_assortativity_categorical(edge_index, y) if y is not None else None
            )

            x_degree = create_node_feature_table_degree(edge_index, n_nodes)

            data_degree = Data(
                x=x_degree,
                edge_index=edge_index,
                y=y,
                masks=masks,
                density=round(density, 2),
                assortativity=assortativity,
                threshold=round(thr, 5),
                dataset=dataset,
                task=task,
                time_window=time_window,
            )

            data_features = Data(
                x=x_features,
                edge_index=edge_index,
                y=y,
                masks=masks,
                density=round(density, 2),
                assortativity=assortativity,
                threshold=round(thr, 5),
                dataset=dataset,
                task=task,
                time_window=time_window,
            )

            data_degree_objects.append(data_degree)
            data_features_objects.append(data_features)

            if verbose:
                special_print(
                    {
                        "target_density": target,
                        "achieved": density,
                        "threshold": thr,
                    },
                    name=f"density {target}% summary",
                )

        return data_degree_objects, data_features_objects, similarity_matrix

