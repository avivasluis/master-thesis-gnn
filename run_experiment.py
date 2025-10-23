import argparse
import torch
from src.models.gcn import GCN 
from src.models.mlp import MLP 
from src.models.gin import GIN 
from src.models.train_evaluate import train_and_evaluate
from src.data.return_sub_graph import get_subgraph_first_n_nodes
import os
import json

def main(args):
    print(f"Start training")
    if args.test_data:
        test_f1, test_auc, _, _ = train_and_evaluate(args.model, args.data, test_data = args.test_data ,print_flag = args.print_flag, log_file=args.log_file)
        if args.log_file:
            results = {
                'graph_name': args.graph,
                'test_f1': test_f1,
                'test_auc': test_auc,
            }
    
    else:
        val_f1, val_auc, _, _ = train_and_evaluate(args.model, args.data, test_data = args.test_data ,print_flag = args.print_flag, log_file = args.log_file)
        if args.log_file:
            results = {
                'graph_name': args.graph,
                'val_f1': val_f1,
                'val_auc': val_auc,
            }
        
        # Derive json path from log_file path
        json_file_path = os.path.splitext(args.log_file)[0] + '.json'
        with open(json_file_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results dictionary saved to {json_file_path}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run GNN Experiment')
    
    # Arguments for data, model, training etc.
    parser.add_argument('graph', type=str, help = 'Name of the file that stores the PyG Data object')
    parser.add_argument('--n_nodes', type=int, default = 50, help = 'Number of nodes to return from the complete graph. -1 to use the complete graph')
    parser.add_argument('--GNN_model', type=str, default='GCN', help='')
    parser.add_argument('--num_layers', type=int, default=2, help='')
    parser.add_argument('--dropout', type=float, default=0.5, help='')
    parser.add_argument('--print_flag_str', type=str, default='False', help='')
    parser.add_argument('--test_data_str', type=str, default='False', help='')
    parser.add_argument('--log_file_name', type=str, default='None', help='Name of the file to save results.')
    args = parser.parse_args()

    if args.print_flag_str == 'False' or 'false':
        args.print_flag = False
    else:
        args.print_flag = True

    if args.test_data_str == 'False' or 'false':
        args.test_data = False
    else:
        args.test_data = True


    args.hidden_channels = 128
    if args.log_file_name != 'None':
        args.log_file = f'{args.log_file_name}'
    else:
        args.log_file = None
        
    processed_graph_path = f'{args.graph}'

    if args.n_nodes == -1:
        args.data = torch.load(processed_graph_path, weights_only=False)
    else:
        data = torch.load(processed_graph_path, weights_only=False)
        args.data = get_subgraph_first_n_nodes(data, args.n_nodes)


    if args.GNN_model == 'MLP':
        args.model = MLP(in_channels=args.data.x.shape[1], 
                        hidden_channels=args.hidden_channels, 
                        out_channels=1,
                        num_layers = args.num_layers,
                        dropout = args.dropout)
    elif args.GNN_model == 'GCN':
        args.model = GCN(in_channels=args.data.x.shape[1], 
                        hidden_channels=args.hidden_channels, 
                        out_channels=1,
                        num_layers = args.num_layers,
                        dropout = args.dropout)
    elif args.GNN_model == 'GIN':
        args.model = GIN(in_channels=args.data.x.shape[1], 
                        hidden_channels=args.hidden_channels, 
                        out_channels=1,
                        num_layers = args.num_layers,
                        dropout = args.dropout)

    print(args.model)

    main(args)