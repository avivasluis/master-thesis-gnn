import argparse
import torch
from src.models.gcn import GCN 
from src.models.mlp import MLP 
from src.models.gin import GIN 
from src.models.train_evaluate import train_and_evaluate
from src.data.return_sub_graph import get_subgraph_first_n_nodes
from src.graph_analysis.metrics import compute_density, compute_assortativity_categorical, compute_homophily_ratio
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

    train_acc, train_f1, train_auc, val_acc, val_f1, val_auc, test_acc, test_f1, test_auc = train_and_evaluate(
        args.model, args.data,
        lr=args.lr, weight_decay=args.weight_decay, pos_weight=args.pos_weight,
        n_epochs=args.n_epochs, early_stop_patience=args.early_stop_patience, log_file=log_file
    )

    results = {
        'graph_name': args.graph,
        'experiment_number': args.log_file_name,
        'experiment_name': args.experiment_name,
        'num_nodes': getattr(args.data, 'num_nodes', None),
        'num_nodes_features': getattr(args.data, 'num_node_features', None),
        'density': getattr(args.data, 'density', compute_density(args.data.num_nodes, args.data.num_edges/2)),
        'threshold': getattr(args.data, 'threshold', None),
        'assortativity': getattr(args.data, 'homophilly', compute_assortativity_categorical(args.data.edge_index, args.data.y)),
        'homophily_ratio': getattr(args.data, 'homophily_ratio', compute_homophily_ratio(args.data.edge_index, args.data.y)),
        'column_data': getattr(args.data, 'data_column', args.data_column),
        'time_window': getattr(args.data, 'time_window', args.time_window),
        'dataset': getattr(args.data, 'dataset', args.dataset),
        'task': getattr(args.data, 'task', args.task),
        'GNN_model': args.GNN_model ,
        'num_layers': args.num_layers,
        'hidden_channels': args.hidden_channels,
        'total_params': args.total_params,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'dropout': args.dropout,
        'train_acc': train_acc,
        'train_f1': train_f1,
        'train_auc': train_auc,
        'val_acc': val_acc,
        'val_f1': val_f1,
        'val_auc': val_auc,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'test_auc': test_auc
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
    parser.add_argument('--time_window', type=str, default='-6mo', help='Time window in which the data was aggregated')
    parser.add_argument('--dataset', type=str, default='rel-amazon', help='Dataset from which the graph was constructed')
    parser.add_argument('--task', type=str, default='user-churn', help='Task from which the graph is constructed')
    parser.add_argument('--experiment_name', type=str, default='', help='Experiment from which the test is coming')

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

    total_params = sum(p.numel() for p in args.model.parameters() if p.requires_grad)
    args.total_params = total_params
    print(f'Total trainable parameters: {total_params:,}')

    main(args)