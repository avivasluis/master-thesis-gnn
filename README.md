# Graph Modeling of Relational Databases

This project explores the application of Graph Neural Networks (GNNs) to tasks typically performed on relational databases. 

The core goal of this framework is to experiment with different methods of constructing graphs from tabular data and evaluating standard GNN architectures against them.

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
