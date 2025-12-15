import copy
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
import os
from torch_geometric.loader import NeighborLoader


def train_epoch(model, optimizer, criterion, train_loader, device, use_hierarchical_sampling=True):
    """Train for one epoch using mini-batches from NeighborLoader.
    
    Parameters
    ----------
    model : torch.nn.Module
        The GNN model to train.
    optimizer : torch.optim.Optimizer
        The optimizer.
    criterion : torch.nn.Module
        The loss function.
    train_loader : NeighborLoader
        DataLoader for training data with neighbor sampling.
    device : torch.device
        Device to run training on.
    use_hierarchical_sampling : bool
        Whether to use hierarchical sampling optimization (trim_to_layer).
    
    Returns
    -------
    float
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0
    total_examples = 0
    
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        
        # Forward pass with optional hierarchical sampling
        if use_hierarchical_sampling and hasattr(batch, 'num_sampled_nodes') and hasattr(batch, 'num_sampled_edges'):
            logits = model(
                batch.x,
                batch.edge_index,
                num_sampled_nodes_per_hop=batch.num_sampled_nodes,
                num_sampled_edges_per_hop=batch.num_sampled_edges,
            ).view(-1)
        else:
            logits = model(batch.x, batch.edge_index).view(-1)
        
        # Only use the target nodes (first batch_size nodes)
        target_logits = logits[:batch.batch_size]
        target_labels = batch.y[:batch.batch_size].float()
        
        loss = criterion(target_logits, target_labels)
        loss.backward()
        optimizer.step()
        
        total_loss += float(loss) * batch.batch_size
        total_examples += batch.batch_size
    
    return total_loss / total_examples


def test_with_loader(model, loader, device, use_hierarchical_sampling=True):
    """Evaluate model using mini-batches from NeighborLoader.
    
    Parameters
    ----------
    model : torch.nn.Module
        The GNN model to evaluate.
    loader : NeighborLoader
        DataLoader for evaluation data with neighbor sampling.
    device : torch.device
        Device to run evaluation on.
    use_hierarchical_sampling : bool
        Whether to use hierarchical sampling optimization.
    
    Returns
    -------
    tuple
        (accuracy, f1_score, auroc, true_labels_np, pred_labels_np)
    """
    model.eval()
    
    all_probs = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            # Forward pass with optional hierarchical sampling
            if use_hierarchical_sampling and hasattr(batch, 'num_sampled_nodes') and hasattr(batch, 'num_sampled_edges'):
                logits = model(
                    batch.x,
                    batch.edge_index,
                    num_sampled_nodes_per_hop=batch.num_sampled_nodes,
                    num_sampled_edges_per_hop=batch.num_sampled_edges,
                ).view(-1)
            else:
                logits = model(batch.x, batch.edge_index).view(-1)
            
            # Only use the target nodes (first batch_size nodes)
            target_logits = logits[:batch.batch_size]
            target_labels = batch.y[:batch.batch_size]
            
            probs = torch.sigmoid(target_logits)
            preds = (probs > 0.5).float()
            
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(target_labels.cpu())
    
    # Concatenate all predictions
    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # Compute metrics
    acc = (all_preds == all_labels).sum() / len(all_labels)
    f1 = f1_score(all_labels, all_preds)
    
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auroc = float('nan')
    
    return acc, f1, auroc, all_labels, all_preds


def generate_model_report(true_labels, pred_labels, name=None):
    """Generate a classification report with confusion matrix.
    
    Parameters
    ----------
    true_labels : np.ndarray
        Ground truth labels.
    pred_labels : np.ndarray
        Predicted labels.
    name : str, optional
        Name to include in the report header.
    
    Returns
    -------
    str
        Formatted report string.
    """
    report = classification_report(true_labels, pred_labels)
    cm = confusion_matrix(true_labels, pred_labels)

    if name:
        report_str = f"Classification Report {name}:\n"
    else:
        report_str = "Classification Report:\n"
    report_str += report
    report_str += "\nConfusion Matrix:\n"
    report_str += str(cm)

    return report_str


def train_and_evaluate(
    model,
    data,
    lr=0.001,
    weight_decay=0,
    pos_weight=None,
    n_epochs=100,
    early_stop_patience=15,
    log_file=None,
    batch_size=1024,
    num_neighbors=None,
    num_workers=0,
    use_hierarchical_sampling=True,
):
    """Train and evaluate a GNN model using NeighborLoader for mini-batch training.
    
    Parameters
    ----------
    model : torch.nn.Module
        The GNN model to train.
    data : torch_geometric.data.Data
        The graph data object with node features, edge_index, labels, and masks.
    lr : float, optional (default=0.001)
        Learning rate.
    weight_decay : float, optional (default=0)
        Weight decay (L2 regularization).
    pos_weight : float, optional
        Weight for positive class in BCEWithLogitsLoss (for imbalanced data).
    n_epochs : int, optional (default=100)
        Maximum number of training epochs.
    early_stop_patience : int, optional (default=15)
        Number of epochs to wait for improvement before early stopping.
    log_file : str, optional
        Path to file for logging training progress.
    batch_size : int, optional (default=1024)
        Number of target nodes per mini-batch.
    num_neighbors : list of int, optional
        Number of neighbors to sample per hop. Length should match model depth.
        Default is [25, 10] for 2-layer models.
    num_workers : int, optional (default=0)
        Number of worker processes for data loading. Use 0 for Windows compatibility.
    use_hierarchical_sampling : bool, optional (default=True)
        Whether to use hierarchical sampling optimization (trim_to_layer).
    
    Returns
    -------
    tuple
        (train_acc, train_f1, train_auc, val_acc, val_f1, val_auc, test_acc, test_f1, test_auc)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device = {device}')
    print()

    log_f = None
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        log_f = open(log_file, 'w')

    def custom_print(message):
        if log_f:
            log_f.write(str(message) + '\n')

    # Get masks from data
    masks = data.masks
    
    # Set default num_neighbors based on model depth if not provided
    if num_neighbors is None:
        num_layers = getattr(model, 'num_layers', len(model.convs))
        num_neighbors = [25, 10][:num_layers] if num_layers <= 2 else [25] + [10] + [5] * (num_layers - 2)
    
    # Convert masks to boolean tensors for NeighborLoader
    train_mask = torch.as_tensor(masks['train_mask']).bool()
    val_mask = torch.as_tensor(masks['val_mask']).bool()
    test_mask = torch.as_tensor(masks['test_mask']).bool()
    
    # Create NeighborLoaders for train, val, test
    # Note: We send x and y to device for faster access during sampling
    data_for_loader = data.to(device, 'x', 'y')
    
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'num_neighbors': num_neighbors,
    }
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True
    
    train_loader = NeighborLoader(
        data_for_loader,
        input_nodes=train_mask,
        shuffle=True,
        **loader_kwargs,
    )
    
    val_loader = NeighborLoader(
        data_for_loader,
        input_nodes=val_mask,
        shuffle=False,
        **loader_kwargs,
    )
    
    test_loader = NeighborLoader(
        data_for_loader,
        input_nodes=test_mask,
        shuffle=False,
        **loader_kwargs,
    )
    
    # Move model to device
    model.to(device)
    
    # Setup optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if pos_weight is not None:
        pos_weight_tensor = torch.tensor([pos_weight], device=device)
    else:
        pos_weight_tensor = None

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    best_val_auc = float('-inf')
    epochs_no_improve = 0
    best_model_state = copy.deepcopy(model.state_dict())

    custom_print("*" * 92)
    custom_print('Data object: ')
    custom_print(f'{data}')
    custom_print(f'\nDevice = {device}')
    custom_print(f'Batch size = {batch_size}')
    custom_print(f'Num neighbors = {num_neighbors}')
    custom_print(f'Hierarchical sampling = {use_hierarchical_sampling}\n')

    for epoch in range(1, n_epochs + 1):
        # Training
        train_loss = train_epoch(
            model, optimizer, criterion, train_loader, device,
            use_hierarchical_sampling=use_hierarchical_sampling
        )
        
        # Validation
        val_acc, val_f1, val_auc, _, _ = test_with_loader(
            model, val_loader, device,
            use_hierarchical_sampling=use_hierarchical_sampling
        )

        custom_print("*" * 92)
        custom_print(f'Epoch: {epoch:03d} | Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val AUROC: {val_auc:.4f}')

        if val_auc > best_val_auc:
            custom_print(f'\tEpoch: {epoch:03d} | Validation improved from {best_val_auc:.6f} to {val_auc:.6f}')
            best_val_auc = val_auc
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            custom_print(f'\tEpoch: {epoch:03d} | No improvement for {epochs_no_improve}/{early_stop_patience} epochs')

        if epochs_no_improve >= early_stop_patience:
            custom_print(f"Early stopping triggered after {epoch} epochs!")
            custom_print("*" * 92)
            break

    # Load best model
    model.load_state_dict(best_model_state)

    # Final evaluation on all splits
    # For training set, we need a loader that covers all training nodes
    train_eval_loader = NeighborLoader(
        data_for_loader,
        input_nodes=train_mask,
        shuffle=False,
        **loader_kwargs,
    )
    
    train_acc, train_f1, train_auc, train_true_labels, train_pred_labels = test_with_loader(
        model, train_eval_loader, device, use_hierarchical_sampling=use_hierarchical_sampling
    )
    custom_print('\nTraining Partition Results: ')
    custom_print(f'train_f1 = {train_f1:.4f} \n train_auc = {train_auc:.4f}\n')
    train_report = generate_model_report(train_true_labels, train_pred_labels)
    custom_print(train_report)

    val_acc, val_f1, val_auc, val_true_labels, val_pred_labels = test_with_loader(
        model, val_loader, device, use_hierarchical_sampling=use_hierarchical_sampling
    )
    custom_print('\nValidation Partition Results: ')
    custom_print(f'val_f1 = {val_f1:.4f} \n val_auc = {val_auc:.4f}\n')
    val_report = generate_model_report(val_true_labels, val_pred_labels)
    custom_print(val_report)

    custom_print('\nTest Partition Results: ')
    test_acc, test_f1, test_auc, test_true_labels, test_pred_labels = test_with_loader(
        model, test_loader, device, use_hierarchical_sampling=use_hierarchical_sampling
    )
    custom_print(f'test_f1 = {test_f1:.4f} \n test_auc = {test_auc:.4f}\n')
    test_report = generate_model_report(test_true_labels, test_pred_labels)
    custom_print(test_report)

    if log_f:
        log_f.close()

    return (train_acc, train_f1, train_auc, val_acc, val_f1, val_auc, test_acc, test_f1, test_auc)
