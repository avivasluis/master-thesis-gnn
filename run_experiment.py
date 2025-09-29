import argparse
import torch
from src.models.gcn import GCN 
from src.models.train_evaluate import train_and_evaluate
import os

def main(args):   
    data = torch.load(args.processed_data_path, weights_only=False)

    # 2. Initialize model based on arguments
    if args.model == 'gcn':
        model = GCN(in_channels=data.x.shape[1], 
                            hidden_channels=args.hidden_channels, 
                            out_channels=1)

    # 3. Train and evaluate
    print(f"Training model...")
    val_f1, val_auc = train_and_evaluate(model, data)
    print(f'val_f1 = {val_f1} \n val_auc = {val_auc}')
    # Here you would save the results, model, plots etc. to a unique directory.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run GNN Experiment')
    
    # Arguments for data, model, training etc.
    parser.add_argument('--model', type=str, default='gcn', help='GNN model to use (gcn, gat, etc.)')
    parser.add_argument('--data_path', type=str, default='data', help='Dataset to use')
        
    args = parser.parse_args()
    args.processed_data_path = f'{args.data_path}/3_processed/rel-amazon/tasks/user-churn/data_thres_4.pt'
    args.hidden_channels = 128

    main(args)