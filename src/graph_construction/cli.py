#!/usr/bin/env python
"""CLI for computing and saving similarity matrices from database columns.

This tool computes a similarity matrix for a single column/feature and saves
both the matrix and the label vector to disk. Graph construction from combined
matrices happens downstream in notebooks.

Example
-------
python -m graph_construction.cli \
    --csv data/expanded_train.parquet \
    --column product_category \
    --type categorical \
    --label_column churn \
    --time_window -6mo \
    --feature_df_path data/feature_matrix.parquet \
    --out graphs/
"""
from __future__ import annotations

import argparse
import importlib
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

from graph_construction.common import save_similarity_matrix, special_print

PIPELINES = {
    "categorical": "graph_construction.categorical",
    "numeric": "graph_construction.numeric",
    "review_count": "graph_construction.review_count",
    "text": "graph_construction.text",
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
    """
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
    p = argparse.ArgumentParser(description="Similarity matrix generator")
    p.add_argument("--csv", required=True, help="Path to expanded training CSV/Parquet")
    p.add_argument("--column", required=True, help="Name of the list column to use")
    p.add_argument("--type", choices=PIPELINES.keys(), required=True, help="Pipeline type")
    p.add_argument("--label_column", default="churn", help="Target/label column name")
    p.add_argument("--out", default="./graphs", help="Output directory for similarity matrices")
    # Categorical-specific hyper-parameters
    p.add_argument("--min_support", type=float, default=0.03, help="min_support for association rules")
    p.add_argument("--min_lift", type=float, default=1.2, help="min_lift for association rules")
    p.add_argument("--no-similarity-map", action="store_true", help="For categorical: skip association rule mining, use direct overlap")
    p.add_argument("--no-preprocess", action="store_true", help="Disable automatic preprocessing of the list column")
    # Required for review_count pipeline
    p.add_argument("--time_window", type=str, required=True, help="Time window from which the data is sampled")
    p.add_argument("--feature_df_path", type=str, required=True, help="Path to the parquet file with feature vectors")
    # Text-specific hyper-parameters
    p.add_argument("--model_name", type=str, default="sentence-transformers/average_word_embeddings_glove.6B.300d",
                   help="SentenceTransformer model name for text embeddings")
    p.add_argument("--device", type=str, default=None, help="Device for text model inference (cuda/cpu/auto)")
    return p.parse_args()

def main() -> None:
    args = _parse_args()

    module = importlib.import_module(PIPELINES[args.type])
    build_similarity_matrix = getattr(module, "build_similarity_matrix")

    path = Path(args.csv)
    df = pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)
    special_print(df.head(), "Loaded DF")

    # --------------------------------------------------
    # Optional preprocessing of the list column
    # --------------------------------------------------
    if not args.no_preprocess and args.column in PREPROCESSORS:
        special_print(f"Applying preprocessing for column '{args.column}'", "Info")
        df[args.column] = df[args.column].apply(PREPROCESSORS[args.column])

    # --------------------------------------------------
    # Build kwargs based on pipeline type
    # --------------------------------------------------
    build_kwargs = dict(
        df=df,
        label_column=args.label_column,
        item_list_column=args.column,
    )

    if args.type == "categorical":
        build_kwargs.update(
            min_support=args.min_support,
            min_lift=args.min_lift,
            use_similarity_map=not args.no_similarity_map
        )
    
    if args.type == "review_count":
        build_kwargs["feature_df"] = pd.read_parquet(args.feature_df_path)

    if args.type == "text":
        build_kwargs.update(
            model_name=args.model_name,
            device=args.device,
        )

    # --------------------------------------------------
    # Compute similarity matrix and labels
    # --------------------------------------------------
    similarity_matrix, y = build_similarity_matrix(**build_kwargs)

    # --------------------------------------------------
    # Save outputs
    # --------------------------------------------------
    save_similarity_matrix(
        similarity_matrix=similarity_matrix,
        labels=y,
        output_dir=args.out,
        column_name=args.column,
    )

if __name__ == "__main__":
    main()
