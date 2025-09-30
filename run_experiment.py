import argparse
import torch
from src.models.gcn import GCN 
from src.models.train_evaluate import train_and_evaluate
from src.data.return_sub_graph import get_subgraph_first_n_nodes
import os

def main(args):
    if args.n_nodes == -1:
        data = torch.load(args.processed_graph_path, weights_only=False)
    else:
        data = torch.load(args.processed_graph_path, weights_only=False)
        data = get_subgraph_first_n_nodes(data, args.n_nodes)
    
    print(data)

    # 2. Initialize model based on arguments
    #if args.model == 'gcn':
    #    model = GCN(in_channels=data.x.shape[1], 
    #                        hidden_channels=args.hidden_channels, 
    #                        out_channels=1)

    # 3. Train and evaluate
    #print(f"Training model...")
    #val_f1, val_auc = train_and_evaluate(model, data)
    #print(f'val_f1 = {val_f1} \n val_auc = {val_auc}')
    # Here you would save the results, model, plots etc. to a unique directory.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run GNN Experiment')
    
    # Arguments for data, model, training etc.
    parser.add_argument('graph', type=str, help = 'Name of the file that stores the PyG Data object')
    parser.add_argument('--n_nodes', type=int, default = 50, help = 'Number of nodes to return from the complete graph. -1 to use the complete graph')
    # parser.add_argument('--model', type=str, default='gcn', help='GNN model to use (gcn, gat, etc.)')
    parser.add_argument('--data_path', type=str, default='data', help='Path to the data directory')
        
    args = parser.parse_args()
    args.processed_graph_path = f'{args.data_path}/3_processed/rel-amazon/tasks/user-churn/common_purchased_categories_by_string/{args.graph}'
    args.hidden_channels = 128

    main(args)