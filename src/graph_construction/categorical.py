from __future__ import annotations

"""Build similarity matrices from categorical columns.

**List columns:** association-rule mining finds similar items; node similarity
uses overlap of similarity groups.

**Scalar columns:** binary similarity (exact match or optional pickle similarity map).
"""

import pickle
from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from .common import (
    is_list_column,
    parse_string_list,
    special_print,
)

__all__ = ["build_similarity_matrix"]


def _build_similarity_map(
    transactions: Sequence[Sequence[str]],
    *,
    min_support: float,
    min_lift: float,
) -> dict[str, set[str]]:
    """Return *symmetric* similarity map using Association-Rule Mining."""
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    rules = association_rules(
        frequent_itemsets.sort_values("support", ascending=False),
        metric="lift",
        min_threshold=min_lift,
    )

    strong_pairs: set[tuple[str, str]] = set()
    for ante, cons in zip(rules["antecedents"], rules["consequents"]):
        for a in ante:
            for b in cons:
                strong_pairs.add((a, b))

    similarity_map: dict[str, set[str]] = defaultdict(set)
    for a, b in strong_pairs:
        similarity_map[a].add(b)
        similarity_map[b].add(a)
    # every item is similar to itself
    for item in set().union(*similarity_map.values()):
        similarity_map[item].add(item)
    return similarity_map


def _load_similarity_map(path: str) -> dict[Any, set[Any]]:
    """Load a pre-computed similarity map from a pickle file."""
    with open(path, "rb") as f:
        loaded = pickle.load(f)
    # Normalize values to sets of comparable items
    out: dict[Any, set[Any]] = {}
    for k, v in loaded.items():
        if isinstance(v, set):
            out[k] = set(v)
        elif v is None:
            out[k] = {k}
        else:
            out[k] = set(v)
    return out


def _stable_cat_key(v: Any) -> Any:
    """Map missing values to a single sentinel for grouping and map lookup."""
    try:
        if pd.isna(v):
            return "__missing__"
    except (TypeError, ValueError):
        pass
    return v


def _get_expansion(value: Any, similarity_map: dict[Any, set[Any]]) -> set[Any]:
    """Return the similarity expansion for a value.

    - If the value is in the map: return {value} | map[value]
    - If the value is NOT in the map: return {value} (exact match only)
    """
    raw = similarity_map.get(value)
    if raw is None:
        return {value}
    return set(raw) | {value}


def _create_scalar_similarity_matrix(
    values: np.ndarray,
    similarity_map: dict[Any, set[Any]] | None = None,
) -> np.ndarray:
    """Compute binary similarity matrix for scalar categorical values.

    Similarity rules:
    - If no similarity_map: exact match only (vi == vj)
    - With similarity_map:
      - If a category is NOT in the map: exact match only
      - If a category IS in the map: exact match + similar categories
    - Two items are similar (1.0) if their expansions overlap
    - Missing values only match other missing values
    """
    n = len(values)
    sim = np.zeros((n, n), dtype=np.float32)

    if similarity_map is None:
        value_to_indices: dict[Any, list[int]] = defaultdict(list)
        for i, v in enumerate(values):
            value_to_indices[_stable_cat_key(v)].append(i)

        for indices in value_to_indices.values():
            for i in indices:
                for j in indices:
                    sim[i, j] = 1.0
        return sim

    # Pre-compute expansions for all unique values
    unique_vals = set(values)
    expansions: dict[Any, set[Any]] = {}
    for v in unique_vals:
        k = _stable_cat_key(v)
        if k == "__missing__":
            expansions[v] = {"__missing__"}
        else:
            expansions[v] = _get_expansion(v, similarity_map)

    for i in range(n):
        vi = values[i]
        exp_i = expansions[vi]
        for j in range(i, n):
            vj = values[j]
            exp_j = expansions[vj]
            # Similar if expansions overlap (bidirectional check)
            if not exp_i.isdisjoint(exp_j):
                sim[i, j] = sim[j, i] = 1.0
    return sim


def _create_similarity_matrix(
    data_lists: Sequence[Sequence[str]],
    similarity_map: dict[str, set[str]],
) -> np.ndarray:
    """Compute symmetric similarity matrix based on *overlap* of similarity groups."""
    sets = [set(lst) for lst in data_lists]
    bags = [pd.Series(lst).value_counts() for lst in data_lists]

    n = len(sets)
    sim = np.zeros((n, n), dtype=float)
    for i in range(n):
        if i % 100 == 0:
            print(f"Processing row {i}/{n}")
        for j in range(i, n):
            total_i = bags[i].sum()
            total_j = bags[j].sum()
            shared_i = sum(
                cnt
                for item, cnt in bags[i].items()
                if not sets[j].isdisjoint(similarity_map.get(item, {item}))
            )
            shared_j = sum(
                cnt
                for item, cnt in bags[j].items()
                if not sets[i].isdisjoint(similarity_map.get(item, {item}))
            )
            Ni = shared_i / total_i if total_i else 0.0
            Nj = shared_j / total_j if total_j else 0.0
            sim_val = (Ni + Nj) / 2
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
    min_support: float = 0.03,
    min_lift: float = 1.2,
    use_similarity_map: bool = True,
    similarity_map_path: str | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute similarity matrix from a categorical column.

    Parameters
    ----------
    df
        Input DataFrame – one row per node.  ``item_list_column`` may contain
        an *iterable* (list/array) of strings per row, or a single categorical
        value per row.
    label_column
        Name of the column that holds the node labels (`y`).
    item_list_column
        Column with categorical items (list or scalar).
    min_support, min_lift
        Hyper-parameters for association-rule mining (list columns only).
    use_similarity_map
        If True (default), use association-rule mining to find similar items
        and expand similarity groups. If False, use direct set overlap where
        items are only considered similar to themselves (list columns only).
    similarity_map_path
        Path to a pickle file ``dict[key, set[related]]`` for scalar columns.
        When set, pairwise similarity is binary: 1 if values match or are in the
        same group; ignored for list columns.
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

    if is_list_column(df[item_list_column]):
        if isinstance(df[item_list_column].dropna().iloc[0], str):
            df[item_list_column] = df[item_list_column].apply(parse_string_list)

        transactions = df[item_list_column].tolist()

        if use_similarity_map:
            similarity_map = _build_similarity_map(
                transactions, min_support=min_support, min_lift=min_lift
            )
            if verbose:
                special_print(similarity_map, "similarity_map", use_pprint=True)
        else:
            similarity_map = {}  # Empty map = items only similar to themselves
            if verbose:
                special_print("Skipping similarity_map (direct overlap mode)", "Info")

        similarity_matrix = _create_similarity_matrix(transactions, similarity_map)
    else:
        values = df[item_list_column].values
        ext_similarity_map = None
        if similarity_map_path:
            ext_similarity_map = _load_similarity_map(similarity_map_path)
            if verbose:
                special_print(
                    f"Loaded similarity map from {similarity_map_path}",
                    "Info",
                )

        similarity_matrix = _create_scalar_similarity_matrix(values, ext_similarity_map)

    if verbose:
        special_print(similarity_matrix.shape, "similarity_matrix.shape")

    # Extract labels
    y = df[label_column].values if label_column in df.columns else None

    return similarity_matrix, y
