from __future__ import annotations

"""Build similarity matrices from a scalar *num_reviews* column using
absolute difference as the similarity measure.

This method is fully vectorized (no Python loops) enabling efficient
computation for 50K+ nodes.

Similarity formula: 1 / (1 + |num_reviews_i - num_reviews_j|)
"""

import numpy as np
import pandas as pd

from .common import special_print

__all__ = ["build_similarity_matrix"]


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

def build_similarity_matrix(
    df: pd.DataFrame,
    *,
    label_column: str = "churn",
    item_list_column: str = "num_reviews",
    feature_df: pd.DataFrame,
    verbose: bool = True,
    dtype: type = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute similarity matrix from review count using absolute difference.

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
    feature_df
        DataFrame holding feature data for every node, including `num_reviews`.
    verbose
        Whether to print progress information.
    dtype
        NumPy dtype for similarity matrix. Default float32 saves memory.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (similarity_matrix, y) where similarity_matrix is shape [N, N] and
        y is the label vector of shape [N].
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

    # Extract labels
    y = df[label_column].values if label_column in df.columns else None

    return similarity_matrix, y
