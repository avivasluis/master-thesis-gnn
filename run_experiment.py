import argparse
import torch
from src.models.gcn import GCN 
from src.models.train_evaluate import train_and_evaluate
from src.data.return_sub_graph import get_subgraph_first_n_nodes
import os

def main(args):
    print(f"Start training")
    val_f1, val_auc, _, _ = train_and_evaluate(args.model, args.data, test_data = args.test_data ,print_flag = args.print_flag, log_file=args.log_file)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run GNN Experiment')
    
    # Arguments for data, model, training etc.
    parser.add_argument('graph', type=str, help = 'Name of the file that stores the PyG Data object')
    parser.add_argument('--n_nodes', type=int, default = 50, help = 'Number of nodes to return from the complete graph. -1 to use the complete graph')
    parser.add_argument('--data_path', type=str, default='data', help='Path to the data directory')
    parser.add_argument('--print_flag', type=bool, default='False', help='')
    parser.add_argument('--test_data', type=bool, default='False', help='')
    parser.add_argument('--log_file_name', type=str, default='None', help='Name of the file to save results.')
    args = parser.parse_args()

    args.hidden_channels = 128
    args.log_file = f'results/rel-amazon/tasks/user-churn/common_purchased_categories_by_string/{args.log_file_name}'
    processed_graph_path = f'{args.data_path}/3_processed/rel-amazon/tasks/user-churn/common_purchased_categories_by_string/{args.graph}'

    if args.n_nodes == -1:
        args.data = torch.load(processed_graph_path, weights_only=False)
    else:
        data = torch.load(processed_graph_path, weights_only=False)
        args.data = get_subgraph_first_n_nodes(data, args.n_nodes)

    # If the degree one-hot-encoded vector is an integer, the GCNConv Layer raises an error
    args.data.x = args.data.x.float()

    args.model = GCN(in_channels=args.data.x.shape[1], 
                        hidden_channels=args.hidden_channels, 
                        out_channels=1)

    main(args)