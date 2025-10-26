import copy
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
import os

def train_loop(model, optimizer, criterion, data, masks):
    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index).view(-1)
    loss = criterion(logits[masks['train_mask']], data.y[masks['train_mask']].float())
    loss.backward()
    optimizer.step()
    return loss.item()

def test(mask, model, data):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index).view(-1)
        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).float()

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
    return acc, f1, auroc, true_np, pred_np

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
    masks = {k: v.to(device) for k, v in masks.items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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

    for epoch in range(1, n_epochs + 1):
        train_loss = train_loop(model, optimizer, criterion, data, masks)
        val_acc, val_f1, val_auc, true_labels, pred_labels = test(masks['val_mask'], model, data)

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

    model.load_state_dict(best_model_state)

    train_acc, train_f1, train_auc, train_true_labels, train_pred_labels = test(masks['train_mask'], model, data)
    custom_print('\nTraining Partition Results: ')
    custom_print(f'train_f1 = {train_f1:.4f} \n train_auc = {train_auc:.4f}\n')
    train_report = generate_model_report(train_true_labels, train_pred_labels)
    custom_print(train_report)

    val_acc, val_f1, val_auc, val_true_labels, val_pred_labels = test(masks['val_mask'], model, data)
    custom_print('\nValidation Partition Results: ')
    custom_print(f'val_f1 = {val_f1:.4f} \n val_auc = {val_auc:.4f}\n')
    val_report = generate_model_report(val_true_labels, val_pred_labels)
    custom_print(val_report)

    custom_print('\nTest Partition Results: ')
    test_acc, test_f1, test_auc, test_true_labels, test_pred_labels = test(masks['test_mask'], model, data)
    custom_print(f'test_f1 = {test_f1:.4f} \n test_auc = {test_auc:.4f}\n')
    test_report = generate_model_report(test_true_labels, test_pred_labels)
    custom_print(test_report)

    if log_f:
        log_f.close()

    return (train_acc, train_f1, train_auc, val_acc, val_f1, val_auc, test_acc, test_f1, test_auc)