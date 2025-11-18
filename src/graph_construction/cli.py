#!/usr/bin/env python
"""Light-weight CLI for building graphs with either pipeline.

Example
-------
python -m graph_construction.cli \
    --csv data/expanded_train.csv \
    --column product_category \
    --type categorical \
    --out graphs/
"""
from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import sys

import pandas as pd
import re
from html import unescape
import numpy as np

# Ensure package root is on sys.path when running the file directly
PACKAGE_NAME = "graph_construction"
this_file = Path(__file__).resolve()
package_root = this_file.parent.parent  # .../src
if package_root.as_posix() not in sys.path:
    sys.path.insert(0, package_root.as_posix())

from graph_construction.common import save_data_object, special_print

PIPELINES = {
    "categorical": "graph_construction.categorical",
    "numeric": "graph_construction.numeric",
}

PREPROCESSORS = {
    "product_brand": lambda lst: [clean_author(s) for s in lst if s],
    "product_category": lambda arr: flatten_and_filter(arr).tolist(),
}

def clean_author(raw: str) -> str:
    """Simplified author cleaner – strips the common "Visit Amazon's <Name> Page" suffix and
    trims whitespace."""
    if not isinstance(raw, str):
        return raw
    raw = raw.strip()
    m = re.match(r"Visit Amazon's\s+(.+?)\s+Page$", raw, flags=re.I)
    return m.group(1).strip() if m else raw

def flatten_and_filter(x) -> np.ndarray:
    """Light-weight version of the helper shown in the notebook.
    1. Flattens nested lists/arrays
    2. Drops empty strings and the standalone word 'Books'
    3. HTML-unescapes strings
    Returns a numpy array of strings."""
    # ---- flatten
    if isinstance(x, np.ndarray) and x.dtype.kind != "O":
        flat = x.ravel()
    else:
        try:
            flat = np.hstack(x).ravel()
        except Exception:
            flat = np.atleast_1d(x).ravel()
    # ---- filter
    clean = [unescape(s) for s in flat if isinstance(s, str) and s.strip() and s.strip().lower() != "books"]
    return np.array(clean, dtype=object)

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Graph constructor")
    p.add_argument("--csv", required=True, help="Path to expanded training CSV")
    p.add_argument("--column", required=True, help="Name of the list column to use")
    p.add_argument("--type", choices=PIPELINES.keys(), required=True, help="Pipeline type")
    p.add_argument("--label-column", default="churn", help="Target/label column name (optional)")
    p.add_argument("--out", default="./graphs", help="Output directory for .pt files")
    p.add_argument("--densities", default="15,10,7,4", help="Comma-separated list of target densities (%)")
    # Categorical-specific hyper-parameters
    p.add_argument("--min_support", type=float, default=0.03, help="min_support for association rules")
    p.add_argument("--min_lift", type=float, default=1.2, help="min_lift for association rules")
    p.add_argument("--no-preprocess", action="store_true", help="Disable automatic preprocessing of the list column before graph construction")
    return p.parse_args()

def main() -> None:
    args = _parse_args()
    target_densities = [float(x) for x in args.densities.split(",")]

    module = importlib.import_module(PIPELINES[args.type])
    build_graph = getattr(module, "build_graph")

    path = Path(args.csv)
    df = pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)
    special_print(df.head(), "Loaded DF")

    # --------------------------------------------------
    # Optional preprocessing of the list column
    # --------------------------------------------------
    if not args.no_preprocess and args.column in PREPROCESSORS:
        special_print(f"Applying preprocessing for column '{args.column}'", "Info")
        df[args.column] = df[args.column].apply(PREPROCESSORS[args.column])

    build_kwargs = dict(
        df=df,
        label_column=args.label_column,
        item_list_column=args.column,
        target_densities=target_densities,
    )

    if args.type == "categorical":
        build_kwargs.update(min_support=args.min_support, min_lift=args.min_lift)

    datas, similarity_matrix = build_graph(**build_kwargs)

    save_data_object(
        similarity_matrix,
        directory_name=args.column,
        threshold=0.0,
        density=0.0,
        output_base_path=args.out,
        similarity_matrix_flag = True
    )

    for data in datas:
        save_data_object(
            data,
            directory_name=args.column,
            threshold=data.threshold,
            density=data.density,
            output_base_path=args.out,
        )

if __name__ == "__main__":
    main()
