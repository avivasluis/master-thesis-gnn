# Graph Modeling of Relational Databases

This project explores the application of Graph Neural Networks (GNNs) to tasks typically performed on relational databases. 

## 📌 Project Overview

The core goal of this framework is to experiment with different methods of constructing graphs from tabular data and evaluating standard GNN architectures against them.

Key features include:
* **Dynamic Graph Construction**: diverse strategies to define edges between users:
    * Shared product categories (exact match, cosine similarity, or Apriori-based frequent itemsets).
    * Semantic similarity of textual data (reviews, product descriptions) using Sentence Transformers (GloVe).
* **Time-Window Data Processing**: Efficient handling of temporal relational data using Polars to aggregate user history within specific lookback periods (e.g., past 6 months).
* **GNN Implementations**: PyTorch Geometric implementations of GCN and GIN, alongside an MLP baseline.
* **Experiment Tracking**: Comprehensive logging of hyperparameters, graph metrics (density, homophily), and model performance (F1, AUC).

## 📂 Repository Structure

```text
├── data/                   # Data storage (raw, intermediate, processed graphs)
├── src/
│   ├── data/               # Scripts for ETL and intermediate data generation (Polars/Pandas)
│   ├── graph_analysis/     # Metrics for analyzing graph properties (density, homophily)
│   ├── graph_construction/ # Strategies for converting relational data to PyG graph objects
│   │   └── rel-amazon/tasks/user-churn/
│   └── models/             # PyTorch model definitions (GCN, GIN, MLP) and training loops
├── run_experiment.py       # Main entry point for training and evaluating models
├── requirements.txt        # Project dependencies
└── README.md
