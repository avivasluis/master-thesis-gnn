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

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Graph constructor")
    p.add_argument("--csv", required=True, help="Path to expanded training CSV")
    p.add_argument("--column", required=True, help="Name of the list column to use")
    p.add_argument("--type", choices=PIPELINES.keys(), required=True, help="Pipeline type")
    p.add_argument("--label-column", default="churn", help="Target/label column name (optional)")
    p.add_argument("--out", default="./graphs", help="Output directory for .pt files")
    p.add_argument("--densities", default="15,10,7,4", help="Comma-separated densities")
    return p.parse_args()

def main() -> None:
    args = _parse_args()
    target_densities = [float(x) for x in args.densities.split(",")]

    module = importlib.import_module(PIPELINES[args.type])
    build_graph = getattr(module, "build_graph")

    path = Path(args.csv)
    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    special_print(df.head(), "Loaded DF")

    datas = build_graph(
        df,
        label_column=args.label_column,
        item_list_column=args.column,
        target_densities=target_densities,
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
