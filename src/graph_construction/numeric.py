from __future__ import annotations

"""Build homogeneous graphs from *numeric* list columns using Wasserstein
(earth-mover) distance as similarity measure.
"""

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

from .common import (
    build_edge_index,
    create_node_feature_table,
    find_threshold_for_target_density,
    parse_string_list,
    return_data_partition_masks,
    return_density,
    special_print,
)

__all__ = ["build_graph"]

# ---------------------------------------------------------------------------
# node continuous feature helpers (stats + empty flag)
# ---------------------------------------------------------------------------

def _stats_vector(arr: Sequence[float] | np.ndarray) -> np.ndarray:
    if len(arr) == 0:
        return np.array([0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=float)
    s = pd.Series(arr, dtype=float)
    descr = s.describe(percentiles=[0.25, 0.50, 0.75]).fillna(0)
    return np.concatenate([descr.values, [0]])  # + flag


def _create_mlp_features(df: pd.DataFrame, column: str) -> torch.Tensor:
    stats_series = df[column].apply(_stats_vector)
    feature_matrix = np.stack(stats_series.values)

    nodes_id = np.arange(len(feature_matrix))
    masks = return_data_partition_masks(nodes_id)

    cont = feature_matrix[:, :8]
    flag = feature_matrix[:, 8:]

    scaler = StandardScaler()
    scaler.fit(cont[masks["train_mask"].numpy()])
    cont_scaled = scaler.transform(cont)

    return torch.as_tensor(np.hstack([cont_scaled, flag]), dtype=torch.float32)


# ---------------------------------------------------------------------------
# similarity matrix – Wasserstein similarity
# ---------------------------------------------------------------------------

def _create_similarity_matrix(data_lists: Sequence[np.ndarray]) -> np.ndarray:
    n = len(data_lists)
    sim = np.ones((n, n), dtype=float)
    for i in range(n):
        if i % 100 == 0:
            print(f"Processing row {i}/{n}")
        for j in range(i + 1, n):
            dist = wasserstein_distance(data_lists[i], data_lists[j])
            sim_val = 1.0 / (1.0 + dist)
            sim[i, j] = sim[j, i] = sim_val
    return sim


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def build_graph(
    df: pd.DataFrame,
    *,
    label_column: str = "churn",
    item_list_column: str,
    target_densities: Iterable[float] = (15, 10, 7, 4),
    density_tol: float = 1.0,
    max_iter: int = 100,
    verbose: bool = True,
) -> list[Data]:
    """Same contract as the categorical builder but for numerical lists."""

    df = df.copy()

    # ensure lists are numeric numpy arrays
    if isinstance(df[item_list_column].iloc[0], str):
        df[item_list_column] = df[item_list_column].apply(parse_string_list)
    df[item_list_column] = df[item_list_column].apply(lambda x: np.asarray(x, dtype=float))

    data_lists = df[item_list_column].tolist()
    if verbose:
        special_print(df.head(), "df.head()")

    similarity_matrix = _create_similarity_matrix(data_lists)
    if verbose:
        special_print(similarity_matrix.shape, "similarity_matrix.shape")

    n_nodes = len(df)
    y = torch.as_tensor(df[label_column].values, dtype=torch.long) if label_column in df else None
    masks = return_data_partition_masks(np.arange(n_nodes))

    # continuous node feature matrix (9-dim)
    x_mlp = _create_mlp_features(df, item_list_column)

    data_objects: list[Data] = []
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

        x_deg = create_node_feature_table(edge_index, n_nodes)
        x = torch.cat([x_deg, x_mlp], dim=1)  # simple concat

        data = Data(x=x, edge_index=edge_index, y=y, masks=masks, density=round(density, 2))
        data.threshold = thr
        data_objects.append(data)

        if verbose:
            special_print(
                {
                    "target_density": target,
                    "achieved": density,
                    "threshold": thr,
                },
                name=f"density {target}% summary",
            )

    return data_objects
