import torch
from sklearn.metrics import f1_score, roc_auc_score
import os
from src.graph_construction.build_graph import return_density

def train_loop(model, optimizer, criterion, data, masks):
    model.train()              # Set model to training mode
    optimizer.zero_grad()      # Clear gradients
    out = model(data.x, data.edge_index)  # Forward pass
    loss = criterion(out[masks['train_mask']].squeeze(), data.y[masks['train_mask']].type(torch.float))
    loss.backward()            # Backward pass
    optimizer.step()           # Update weights
    return loss.item()

def test(mask, model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = torch.sigmoid(out.squeeze())  # Compute probabilities
        pred = (probs > 0.5).float()  # Binary prediction for acc and F1
        acc = (pred[mask] == data.y[mask]).sum().item() / mask.sum().item()
        pred_np = pred[mask].cpu().numpy()
        true_np = data.y[mask].cpu().numpy()
        probs_np = probs[mask].cpu().numpy()
        f1 = f1_score(true_np, pred_np)
        auroc = roc_auc_score(true_np, probs_np)  # Use probabilities for AUC
    return acc, f1, auroc, true_np, pred_np

def train_and_evaluate(model, data, early_stop_patience = 15, test_data = False, print_flag = False, log_file=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    log_f = None
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        log_f = open(log_file, 'w')

    def custom_print(message):
        if print_flag:
            print(message)
            
        if log_f:
            log_f.write(str(message) + '\n')

    masks = data.masks
    data = data.to(device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_auc = 0
    epochs_no_improve = 0
    best_model_state = model.state_dict()



    custom_print("*" * 92)
    custom_print('Data object: ')
    custom_print(f'{data}')
    density = return_density(data.num_nodes, (len(data.edge_index[0])/2))
    custom_print(f'Graph density: {density:.4f}%')

    for epoch in range(1, 101):
        train_loss = train_loop(model, optimizer, criterion, data, masks)
        val_acc, val_f1, val_auc, true_labels, pred_labels = test(masks['val_mask'], model, data)

        if print_flag:
            custom_print("*" * 92)
            custom_print(f'Epoch: {epoch:03d} | Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val AUROC: {val_auc:.4f}')

        if val_auc > best_val_auc:
            if print_flag:
                custom_print(f'\tEpoch: {epoch:03d} | Validation  improved from {best_val_auc:.6f} to {val_auc:.6f} in epoch {epoch:03d}')
            best_val_auc = val_auc
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if print_flag:
                custom_print(f'\tEpoch: {epoch:03d} | No improvement for {epochs_no_improve}/{early_stop_patience} epochs')

        # Check early stopping condition
        if epochs_no_improve >= early_stop_patience:
            if print_flag:
                custom_print(f"Early stopping triggered after {epoch} epochs!")
                custom_print("*" * 92)
            break

    # Load the best model
    model.load_state_dict(best_model_state)

    if test_data:
        test_acc, test_f1, test_auc, true_labels, pred_labels = test(masks['test_mask'], model, data)
        custom_print(f'test_f1 = {test_f1:.4f} \n test_auc = {test_auc:.4f}')
        if log_f:
            log_f.close()

        return test_f1, test_auc, true_labels, pred_labels
        
    else:
        val_acc, val_f1, val_auc, true_labels, pred_labels = test(masks['val_mask'], model, data)
        custom_print(f'val_f1 = {val_f1:.4f} \n val_auc = {val_auc:.4f}')
        if log_f:
            log_f.close()

        return val_f1, val_auc, true_labels, pred_labels

        