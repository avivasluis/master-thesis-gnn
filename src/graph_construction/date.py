from __future__ import annotations

"""Build similarity matrices from *date* list columns using temporal
features and cosine similarity.

The algorithm extracts temporal features from each node's list of dates:
- days_since_reference: Days from reference date to the most recent date
- total_count: Number of dates in the list
- avg_interval: Average days between consecutive dates
- std_interval: Standard deviation of intervals between dates

These features are then scaled and used to compute pairwise cosine similarity.
"""

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from .common import special_print

__all__ = ["build_similarity_matrix"]


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _parse_dates(dates: Sequence) -> list:
    """Convert a sequence of date-like values to datetime.date objects.
    
    Handles various input formats including strings, timestamps, and datetime objects.
    Filters out NaT/None values.
    """
    parsed = []
    for d in dates:
        if d is None:
            continue
        if pd.isna(d):
            continue
        try:
            dt = pd.to_datetime(d)
            if pd.notna(dt):
                parsed.append(dt.date())
        except (ValueError, TypeError):
            continue
    return parsed


def _compute_date_features(
    dates: Sequence,
    reference_date,
    has_items: bool,
) -> tuple[float, float, float, float]:
    """Extract temporal features from a list of dates.
    
    Parameters
    ----------
    dates
        List of date values (can be strings, timestamps, or datetime objects).
    reference_date
        The reference date to compute days_since_reference from.
    has_items
        Whether the node has any associated items (used to distinguish between
        nodes with no items vs nodes with items but no valid dates).
    
    Returns
    -------
    tuple[float, float, float, float]
        (days_since_reference, total_count, avg_interval, std_interval)
    """
    if not has_items:
        return (-1.0, 0.0, -1.0, -1.0)
    
    parsed_dates = _parse_dates(dates)
    
    if len(parsed_dates) == 0:
        return (-1.0, 0.0, -1.0, -1.0)
    
    ref_date = pd.to_datetime(reference_date).date()
    sorted_dates = sorted(parsed_dates)
    
    # 1. Days since the most recent date
    days_since_reference = (ref_date - sorted_dates[-1]).days
    
    # 2. Total count of dates
    total_count = len(sorted_dates)
    
    if total_count > 1:
        # Compute intervals between consecutive dates
        intervals = [
            (sorted_dates[i] - sorted_dates[i - 1]).days
            for i in range(1, total_count)
        ]
        # 3. Average interval
        avg_interval = np.mean(intervals)
        # 4. Standard deviation of intervals
        std_interval = np.std(intervals)
    else:
        avg_interval = -1.0
        std_interval = -1.0
    
    return (days_since_reference, total_count, avg_interval, std_interval)


def _build_feature_matrix(
    df: pd.DataFrame,
    date_col: str,
    ref_date_col: str,
    id_col: str | None,
) -> np.ndarray:
    """Build the feature matrix from all rows in the DataFrame.
    
    Parameters
    ----------
    df
        Input DataFrame.
    date_col
        Column containing lists of dates.
    ref_date_col
        Column containing the reference date for each row.
    id_col
        Optional column to detect empty records. If provided, rows where this
        column contains an empty list are treated as having no items.
    
    Returns
    -------
    np.ndarray
        Feature matrix of shape (n_rows, 4).
    """
    features = []
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        dates = row[date_col]
        reference_date = row[ref_date_col]
        
        # Determine if the node has items
        if id_col is not None and id_col in df.columns:
            id_list = row[id_col]
            has_items = id_list is not None and len(id_list) > 0
        else:
            # If no id_col specified, use the date list length
            has_items = dates is not None and len(dates) > 0
        
        feat = _compute_date_features(dates, reference_date, has_items)
        features.append(feat)
    
    return np.array(features, dtype=float)


# ---------------------------------------------------------------------------
# Similarity matrix computation
# ---------------------------------------------------------------------------

def _create_similarity_matrix(feature_matrix: np.ndarray) -> np.ndarray:
    """Compute similarity matrix using scaled features and cosine similarity.
    
    Parameters
    ----------
    feature_matrix
        Feature matrix of shape (n_nodes, n_features).
    
    Returns
    -------
    np.ndarray
        Similarity matrix of shape (n_nodes, n_nodes).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_matrix)
    
    similarity_matrix = cosine_similarity(X_scaled)
    
    return similarity_matrix


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_similarity_matrix(
    df: pd.DataFrame,
    *,
    label_column: str = "churn",
    item_list_column: str,
    reference_date_column: str = "timestamp",
    id_column: str | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute similarity matrix from date list column using temporal features.

    The algorithm extracts 4 temporal features per node:
    - days_since_reference: Days from reference date to the most recent date
    - total_count: Number of dates in the list
    - avg_interval: Average days between consecutive dates  
    - std_interval: Standard deviation of intervals

    These features are scaled with StandardScaler and cosine similarity is
    computed between all pairs of nodes.

    Parameters
    ----------
    df
        Input DataFrame – one row per node. ``item_list_column`` must contain
        an *iterable* (list/array) of date-like values.
    label_column
        Name of the column that holds the node labels (`y`).
    item_list_column
        Column with the list of dates.
    reference_date_column
        Column with the reference date for computing days_since_reference.
        Typically this is the observation/snapshot timestamp.
    id_column
        Optional column to detect empty records. If provided, rows where this
        column contains an empty list are treated as having no items (returns
        sentinel values for features). If None, uses the date list length.
    verbose
        Whether to print progress information.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (similarity_matrix, y) where similarity_matrix is shape [N, N] and
        y is the label vector of shape [N].
    """
    df = df.copy()
    
    if verbose:
        special_print(df.head(), "df.head()")
        print(f"Building features from date column: {item_list_column}")
        print(f"Reference date column: {reference_date_column}")
        if id_column:
            print(f"ID column for empty detection: {id_column}")
    
    # Build feature matrix
    feature_matrix = _build_feature_matrix(
        df=df,
        date_col=item_list_column,
        ref_date_col=reference_date_column,
        id_col=id_column,
    )
    
    if verbose:
        special_print(feature_matrix.shape, "feature_matrix.shape")
        feature_names = [
            "days_since_reference",
            "total_count", 
            "avg_interval",
            "std_interval",
        ]
        feature_df = pd.DataFrame(feature_matrix, columns=feature_names)
        special_print(feature_df.describe(), "Feature statistics")
    
    # Compute similarity matrix
    similarity_matrix = _create_similarity_matrix(feature_matrix)
    
    if verbose:
        special_print(similarity_matrix.shape, "similarity_matrix.shape")
    
    # Extract labels
    y = df[label_column].values if label_column in df.columns else None
    
    return similarity_matrix, y
