from __future__ import annotations

"""Build similarity matrices from numeric columns.

**List columns:** Wasserstein distance between distributions.

**Scalar columns:** min-max normalization then ``1 / (1 + |x_i - x_j|)``.
"""

from typing import Sequence
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from .common import (
    is_list_column,
    parse_string_list,
    special_print,
)

__all__ = ["build_similarity_matrix"]


# ---------------------------------------------------------------------------
# similarity matrix – Wasserstein similarity
# ---------------------------------------------------------------------------

def _create_scalar_similarity_matrix(
    values: np.ndarray,
    dtype: type = np.float32,
) -> np.ndarray:
    """Compute similarity from scalar values using min-max normalization and absolute difference.

    Formula: sim = 1 / (1 + |norm_i - norm_j|) after scaling values to [0, 1].
    """
    values = np.asarray(values, dtype=dtype).ravel()
    v_min = float(np.nanmin(values))
    v_max = float(np.nanmax(values))
    if not np.isfinite(v_min) or not np.isfinite(v_max) or v_max <= v_min:
        values_norm = np.zeros_like(values, dtype=dtype)
    else:
        values_norm = (values - v_min) / (v_max - v_min)
        values_norm = np.nan_to_num(values_norm, nan=0.0)

    diff = np.abs(values_norm[:, None] - values_norm[None, :])
    return (1.0 / (1.0 + diff)).astype(dtype)


def _create_similarity_matrix(data_lists: Sequence[np.ndarray]) -> np.ndarray:
    """Compute similarity matrix using Wasserstein distance.
    
    Similarity formula: sim = 1.0 / (1.0 + wasserstein_distance(i, j))
    
    Empty list handling:
        - Both empty: similarity = 1.0 (identical empty distributions)
        - One empty, one not: similarity = 0.0 (no meaningful comparison)
    """
    n = len(data_lists)
    sim = np.ones((n, n), dtype=float)
    for i in range(n):
        if i % 100 == 0:
            print(f"Processing row {i}/{n}")
        for j in range(i + 1, n):
            len_i = len(data_lists[i])
            len_j = len(data_lists[j])
            
            if len_i == 0 or len_j == 0:
                # Both empty -> perfect similarity; one empty -> no similarity
                sim_val = 1.0 if (len_i == 0 and len_j == 0) else 0.0
            else:
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
        Input DataFrame – one row per node. ``item_list_column`` may contain
        an *iterable* (list/array) of numbers per row, or a single numeric value per row.
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

    if is_list_column(df[item_list_column]):
        # ensure lists are numeric numpy arrays
        first = df[item_list_column].dropna().iloc[0]
        if isinstance(first, str):
            df[item_list_column] = df[item_list_column].apply(parse_string_list)
        df[item_list_column] = df[item_list_column].apply(lambda x: np.asarray(x, dtype=float))

        data_lists = df[item_list_column].tolist()
        if verbose:
            special_print(df.head(), "df.head()")

        similarity_matrix = _create_similarity_matrix(data_lists)
    else:
        values = df[item_list_column].values.astype(float)
        if verbose:
            special_print(df.head(), "df.head()")

        similarity_matrix = _create_scalar_similarity_matrix(values)
    if verbose:
        special_print(similarity_matrix.shape, "similarity_matrix.shape")

    # Extract labels
    y = df[label_column].values if label_column in df.columns else None

    return similarity_matrix, y
