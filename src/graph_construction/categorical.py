from __future__ import annotations

"""Build *homogeneous* graphs from a DataFrame column that contains **lists of
categorical strings** per node.

The algorithm is the same as presented in the *process-categories-complete-algorithm*
notebook but rewritten as a reusable function.
"""

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
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


def build_graph(
    df: pd.DataFrame,
    *,
    label_column: str = "churn",
    item_list_column: str,
    min_support: float = 0.03,
    min_lift: float = 1.2,
    target_densities: Iterable[float] = (15, 10, 7, 4),
    density_tol: float = 1.0,
    max_iter: int = 100,
    verbose: bool = True,
) -> list[Data]:
    """Construct one :class:`torch_geometric.data.Data` object **per** target density.

    Parameters
    ----------
    df
        Input DataFrame – one row per node.  ``item_list_column`` must contain
        an *iterable* (list/array) of strings.
    label_column
        Name of the column that holds the node labels (`y`).  Can be ``None`` if
        you want unlabeled graphs.
    item_list_column
        Column with the list of categorical items.
    min_support, min_lift
        Hyper-parameters for association-rule mining.
    target_densities
        Iterable of desired densities (percentage).  One graph is produced for
        each density.
    density_tol
        Acceptable absolute error in the achieved density.
    max_iter
        Maximum binary-search iterations per density.
    verbose
        Whether to print progress information.
    """

    df = df.copy()
    if isinstance(df[item_list_column].iloc[0], str):
        df[item_list_column] = df[item_list_column].apply(parse_string_list)

    transactions = df[item_list_column].tolist()
    if verbose:
        special_print(df.head(), "df.head()")

    similarity_map = _build_similarity_map(transactions, min_support=min_support, min_lift=min_lift)
    if verbose:
        special_print(similarity_map, "similarity_map", use_pprint=True)

    similarity_matrix = _create_similarity_matrix(transactions, similarity_map)
    if verbose:
        special_print(similarity_matrix.shape, "similarity_matrix.shape")

    n_nodes = len(df)
    y = torch.as_tensor(df[label_column].values, dtype=torch.long) if label_column in df else None
    masks = return_data_partition_masks(np.arange(n_nodes))

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

        x = create_node_feature_table(edge_index, n_nodes)
        data = Data(x=x, edge_index=edge_index, y=y, masks=masks, density=round(density, 2))
        data.threshold = thr  # attach extra attributes for convenience
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

    return data_objects, similarity_matrix