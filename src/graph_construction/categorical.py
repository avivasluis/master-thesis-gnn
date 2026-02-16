from __future__ import annotations

"""Build similarity matrices from a DataFrame column that contains **lists of
categorical strings** per node.

The algorithm uses association-rule mining to find similar items, then computes
pairwise node similarity based on overlap of similarity groups.
"""

from collections import defaultdict
from typing import Sequence

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from .common import (
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
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute similarity matrix from categorical list column.

    Parameters
    ----------
    df
        Input DataFrame – one row per node.  ``item_list_column`` must contain
        an *iterable* (list/array) of strings.
    label_column
        Name of the column that holds the node labels (`y`).
    item_list_column
        Column with the list of categorical items.
    min_support, min_lift
        Hyper-parameters for association-rule mining.
    use_similarity_map
        If True (default), use association-rule mining to find similar items
        and expand similarity groups. If False, use direct set overlap where
        items are only considered similar to themselves.
    verbose
        Whether to print progress information.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (similarity_matrix, y) where similarity_matrix is shape [N, N] and
        y is the label vector of shape [N].
    """
    df = df.copy()
    if isinstance(df[item_list_column].iloc[0], str):
        df[item_list_column] = df[item_list_column].apply(parse_string_list)

    transactions = df[item_list_column].tolist()
    if verbose:
        special_print(df.head(), "df.head()")

    if use_similarity_map:
        similarity_map = _build_similarity_map(transactions, min_support=min_support, min_lift=min_lift)
        if verbose:
            special_print(similarity_map, "similarity_map", use_pprint=True)
    else:
        similarity_map = {}  # Empty map = items only similar to themselves
        if verbose:
            special_print("Skipping similarity_map (direct overlap mode)", "Info")

    similarity_matrix = _create_similarity_matrix(transactions, similarity_map)
    if verbose:
        special_print(similarity_matrix.shape, "similarity_matrix.shape")

    # Extract labels
    y = df[label_column].values if label_column in df.columns else None

    return similarity_matrix, y
