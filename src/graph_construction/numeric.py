from __future__ import annotations

"""Build similarity matrices from *numeric* list columns using Wasserstein
(earth-mover) distance as similarity measure.
"""

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from .common import (
    parse_string_list,
    special_print,
)

__all__ = ["build_similarity_matrix"]


# ---------------------------------------------------------------------------
# similarity matrix – Wasserstein similarity
# ---------------------------------------------------------------------------

def _create_similarity_matrix(data_lists: Sequence[np.ndarray]) -> np.ndarray:
    """Compute similarity matrix using Wasserstein distance.
    
    Similarity formula: sim = 1.0 / (1.0 + wasserstein_distance(i, j))
    """
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

def build_similarity_matrix(
    df: pd.DataFrame,
    *,
    label_column: str = "churn",
    item_list_column: str,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute similarity matrix from numeric list column using Wasserstein distance.

    Parameters
    ----------
    df
        Input DataFrame – one row per node. ``item_list_column`` must contain
        an *iterable* (list/array) of numbers.
    label_column
        Name of the column that holds the node labels (`y`).
    item_list_column
        Column with the list of numeric values.
    verbose
        Whether to print progress information.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (similarity_matrix, y) where similarity_matrix is shape [N, N] and
        y is the label vector of shape [N].
    """
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

    # Extract labels
    y = df[label_column].values if label_column in df.columns else None

    return similarity_matrix, y
