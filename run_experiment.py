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
    print()
    print(f"Start training")
    print('Model to use: ->')
    print(args.model)
    print()

    log_file = None
    if args.log_file_path and args.data_column and args.log_file_name:
        os.makedirs(os.path.join(args.log_file_path, args.data_column), exist_ok=True)
        log_file = os.path.join(args.log_file_path, args.data_column, args.log_file_name)

    train_acc, train_f1, train_auc, train_true_labels, train_pred_labels, val_acc, val_f1, val_auc, val_true_labels, val_pred_labels, test_acc, test_f1, test_auc, test_true_labels, test_pred_labels = train_and_evaluate(
        args.model, args.data,
        lr=args.lr, weight_decay=args.weight_decay, pos_weight=args.pos_weight,
        n_epochs=args.n_epochs, early_stop_patience=args.early_stop_patience, log_file=log_file
    )

    results = {
        'graph_name': args.graph,
        'density': getattr(args.data, 'density', None),
        'train_acc': train_acc,
        'train_f1': train_f1,
        'train_auc': train_auc,
        'val_acc': val_acc,
        'val_f1': val_f1,
        'val_auc': val_auc,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'test_auc': test_auc,
    }
    
    if log_file:
        json_file_path = os.path.splitext(log_file)[0] + '_results.json'
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
    parser.add_argument('--hidden_channels', type=int, default=128, help='')

    parser.add_argument('--lr', type=float, default=0.001, help='')
    parser.add_argument('--weight_decay', type=float, default=0, help='')

    parser.add_argument('--dropout', type=float, default=0.5, help='')
    parser.add_argument('--early_stop_patience', type=int, default=15, help='')
    parser.add_argument('--pos_weight', type=float, default=None, help='')
    parser.add_argument('--n_epochs', type=int, default=100, help='')

    parser.add_argument('--log_file_path', type=str, default=None, help='Path of the file to save results (without data column)')
    parser.add_argument('--log_file_name', type=str, default=None, help='Name of the file to save results.')
    parser.add_argument('--data_column', type=str, default=None, help='Data column from which the graph was created')

    args = parser.parse_args()

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

    main(args)