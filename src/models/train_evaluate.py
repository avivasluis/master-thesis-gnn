import copy
import torch
import numpy as np
import time
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
import os


def find_optimal_threshold(y_true, y_probs):
    """
    Find optimal classification threshold on validation data.
    
    Args:
        y_true: Ground truth binary labels
        y_probs: Predicted probabilities

    Returns:
        best_threshold: Optimal threshold value
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.5
    best_score = 0
    
    for thresh in thresholds:
        preds = (y_probs >= thresh).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    return best_threshold

def train_loop(model, optimizer, criterion, data, masks):
    model.train()
    optimizer.zero_grad()
    if data.x.is_cuda:
        torch.cuda.synchronize()
    step_start = time.perf_counter()

    logits = model(data.x, data.edge_index).view(-1)
    loss = criterion(logits[masks['train_mask']], data.y[masks['train_mask']].float())
    loss.backward()
    optimizer.step()

    if data.x.is_cuda:
        torch.cuda.synchronize()
    step_time_sec = time.perf_counter() - step_start
    return loss.item(), step_time_sec

def test(mask, model, data, threshold=0.5):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index).view(-1)
        probs = torch.sigmoid(logits)
        pred = (probs > threshold).float()

        m = mask.bool()
        acc = (pred[m] == data.y[m]).sum().item() / m.sum().item()

        pred_np = pred[m].cpu().numpy()
        true_np = data.y[m].cpu().numpy()
        probs_np = probs[m].cpu().numpy()

        f1 = f1_score(true_np, pred_np)
        try:
            auroc = roc_auc_score(true_np, probs_np)
        except ValueError:
            auroc = float('nan')
    return acc, f1, auroc, true_np, pred_np, probs_np

def generate_model_report(true_labels, pred_labels, name = None):
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

def train_and_evaluate(model, data, lr=0.001, weight_decay=0, pos_weight=None, n_epochs=100, early_stop_patience=15, log_file=None):

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

    masks = data.masks
    data = data.to(device)
    model.to(device)
    masks = {k: torch.as_tensor(v).to(device) for k, v in masks.items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    if pos_weight is not None:
        pos_weight = torch.tensor([pos_weight], device=device)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auc = float('-inf')
    epochs_no_improve = 0
    best_model_state = copy.deepcopy(model.state_dict())

    custom_print("*" * 92)
    custom_print('Data object: ')
    custom_print(f'{data}')
    custom_print(f'\nDevice = {device}\n')

    total_train_start = time.perf_counter()
    epoch_train_step_times_sec = []
    for epoch in range(1, n_epochs + 1):
        train_loss, train_step_time_sec = train_loop(model, optimizer, criterion, data, masks)
        epoch_train_step_times_sec.append(train_step_time_sec)
        val_acc, val_f1, val_auc, true_labels, pred_labels, _ = test(masks['val_mask'], model, data)

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
    if device.type == 'cuda':
        torch.cuda.synchronize()
    total_train_time_sec = time.perf_counter() - total_train_start

    model.load_state_dict(best_model_state)

    # Find optimal threshold using validation set
    _, _, _, val_true_labels, _, val_probs = test(masks['val_mask'], model, data, threshold=0.5)
    optimal_threshold = find_optimal_threshold(val_true_labels, val_probs)
    custom_print(f'\nOptimal threshold (based on validation F1): {optimal_threshold:.4f}')

    # Evaluate all partitions using optimal threshold
    train_acc, train_f1, train_auc, train_true_labels, train_pred_labels, _ = test(masks['train_mask'], model, data, threshold=optimal_threshold)
    custom_print('\nTraining Partition Results: ')
    custom_print(f'train_f1 = {train_f1:.4f} \n train_auc = {train_auc:.4f}\n')
    train_report = generate_model_report(train_true_labels, train_pred_labels)
    custom_print(train_report)

    val_acc, val_f1, val_auc, val_true_labels, val_pred_labels, _ = test(masks['val_mask'], model, data, threshold=optimal_threshold)
    custom_print('\nValidation Partition Results: ')
    custom_print(f'val_f1 = {val_f1:.4f} \n val_auc = {val_auc:.4f}\n')
    val_report = generate_model_report(val_true_labels, val_pred_labels)
    custom_print(val_report)

    custom_print('\nTest Partition Results: ')
    if device.type == 'cuda':
        torch.cuda.synchronize()
    inference_start = time.perf_counter()
    test_acc, test_f1, test_auc, test_true_labels, test_pred_labels, _ = test(masks['test_mask'], model, data, threshold=optimal_threshold)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    inference_time_sec = time.perf_counter() - inference_start
    custom_print(f'test_f1 = {test_f1:.4f} \n test_auc = {test_auc:.4f}\n')
    test_report = generate_model_report(test_true_labels, test_pred_labels)
    custom_print(test_report)

    if log_f:
        log_f.close()

    peak_gpu_memory_mb = None
    if device.type == 'cuda':
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    avg_train_time_per_epoch_sec = float('nan')
    if epoch_train_step_times_sec:
        avg_train_time_per_epoch_sec = float(sum(epoch_train_step_times_sec) / len(epoch_train_step_times_sec))

    hardware_metrics = {
        'avg_train_time_per_epoch_sec': avg_train_time_per_epoch_sec,
        'total_train_time_to_convergence_sec': total_train_time_sec,
        'inference_time_sec': inference_time_sec,
        'peak_gpu_memory_allocated_mb': peak_gpu_memory_mb,
    }

    return (train_acc, train_f1, train_auc, val_acc, val_f1, val_auc, test_acc, test_f1, test_auc, optimal_threshold, hardware_metrics)