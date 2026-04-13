"""Full-batch training and evaluation for multi-class node classification."""

import copy
import os
import time

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score


def train_loop(model, optimizer, criterion, data, masks):
    model.train()
    optimizer.zero_grad()
    if data.x.is_cuda:
        torch.cuda.synchronize()
    step_start = time.perf_counter()

    logits = model(data.x, data.edge_index)
    train_idx = masks["train_mask"].bool()
    y_train = data.y[train_idx].long()
    loss = criterion(logits[train_idx], y_train)
    loss.backward()
    optimizer.step()

    if data.x.is_cuda:
        torch.cuda.synchronize()
    step_time_sec = time.perf_counter() - step_start
    return loss.item(), step_time_sec


def test(mask, model, data):
    """Evaluate on a mask; returns acc, f1_micro, f1_macro, auroc, true_np, pred_np, probs_np."""
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)

        m = mask.bool()
        acc = (pred[m] == data.y[m]).sum().item() / m.sum().item()

        pred_np = pred[m].cpu().numpy()
        true_np = data.y[m].cpu().numpy().astype(np.int64)
        probs_np = probs[m].cpu().numpy()

        f1_micro = f1_score(true_np, pred_np, average="micro", zero_division=0)
        f1_macro = f1_score(true_np, pred_np, average="macro", zero_division=0)
        try:
            auroc = roc_auc_score(
                true_np, probs_np, multi_class="ovr", average="macro"
            )
        except ValueError:
            auroc = float("nan")

    return acc, f1_micro, f1_macro, auroc, true_np, pred_np, probs_np


def generate_model_report(true_labels, pred_labels, name=None):
    report = classification_report(true_labels, pred_labels, zero_division=0)
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
    class_weights=None,
    n_epochs=100,
    early_stop_patience=15,
    log_file=None,
):
    """
    Train and evaluate a multi-class node classifier (full-batch).

    Early stopping uses validation micro-F1. No threshold tuning (argmax).

    Parameters
    ----------
    class_weights : torch.Tensor or None
        Optional 1D tensor of shape [num_classes] for CrossEntropyLoss weight.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device = {device}")
    print()

    log_f = None
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        log_f = open(log_file, "w")

    def custom_print(message):
        if log_f:
            log_f.write(str(message) + "\n")

    masks = data.masks
    data = data.to(device)
    model.to(device)
    masks = {k: torch.as_tensor(v).to(device) for k, v in masks.items()}
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32, device=device)

    criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)

    best_val_f1_micro = float("-inf")
    epochs_no_improve = 0
    best_model_state = copy.deepcopy(model.state_dict())

    custom_print("*" * 92)
    custom_print("Data object: ")
    custom_print(f"{data}")
    custom_print(f"\nDevice = {device}\n")

    total_train_start = time.perf_counter()
    epoch_train_step_times_sec = []
    for epoch in range(1, n_epochs + 1):
        train_loss, train_step_time_sec = train_loop(
            model, optimizer, criterion, data, masks
        )
        epoch_train_step_times_sec.append(train_step_time_sec)
        val_acc, val_f1_micro, val_f1_macro, val_auc, _, _, _ = test(
            masks["val_mask"], model, data
        )

        custom_print("*" * 92)
        custom_print(
            f"Epoch: {epoch:03d} | Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}, "
            f"Val F1 (micro): {val_f1_micro:.4f}, Val F1 (macro): {val_f1_macro:.4f}, "
            f"Val AUROC (OvR macro): {val_auc:.4f}"
        )

        if val_f1_micro > best_val_f1_micro:
            custom_print(
                f"\tEpoch: {epoch:03d} | Validation micro-F1 improved from "
                f"{best_val_f1_micro:.6f} to {val_f1_micro:.6f}"
            )
            best_val_f1_micro = val_f1_micro
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            custom_print(
                f"\tEpoch: {epoch:03d} | No improvement for "
                f"{epochs_no_improve}/{early_stop_patience} epochs"
            )

        if epochs_no_improve >= early_stop_patience:
            custom_print(f"Early stopping triggered after {epoch} epochs!")
            custom_print("*" * 92)
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_train_time_sec = time.perf_counter() - total_train_start

    model.load_state_dict(best_model_state)

    train_acc, train_f1_micro, train_f1_macro, train_auc, train_true, train_pred, _ = test(
        masks["train_mask"], model, data
    )
    custom_print("\nTraining Partition Results: ")
    custom_print(
        f"train_f1_micro = {train_f1_micro:.4f} \n train_f1_macro = {train_f1_macro:.4f} \n "
        f"train_auc = {train_auc:.4f}\n"
    )
    custom_print(generate_model_report(train_true, train_pred))

    val_acc, val_f1_micro, val_f1_macro, val_auc, val_true, val_pred, _ = test(
        masks["val_mask"], model, data
    )
    custom_print("\nValidation Partition Results: ")
    custom_print(
        f"val_f1_micro = {val_f1_micro:.4f} \n val_f1_macro = {val_f1_macro:.4f} \n "
        f"val_auc = {val_auc:.4f}\n"
    )
    custom_print(generate_model_report(val_true, val_pred))

    custom_print("\nTest Partition Results: ")
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_start = time.perf_counter()
    test_acc, test_f1_micro, test_f1_macro, test_auc, test_true, test_pred, _ = test(
        masks["test_mask"], model, data
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_time_sec = time.perf_counter() - inference_start
    custom_print(
        f"test_f1_micro = {test_f1_micro:.4f} \n test_f1_macro = {test_f1_macro:.4f} \n "
        f"test_auc = {test_auc:.4f}\n"
    )
    custom_print(generate_model_report(test_true, test_pred))

    if log_f:
        log_f.close()

    peak_gpu_memory_mb = None
    if device.type == "cuda":
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    avg_train_time_per_epoch_sec = float("nan")
    if epoch_train_step_times_sec:
        avg_train_time_per_epoch_sec = float(
            sum(epoch_train_step_times_sec) / len(epoch_train_step_times_sec)
        )

    hardware_metrics = {
        "avg_train_time_per_epoch_sec": avg_train_time_per_epoch_sec,
        "total_train_time_to_convergence_sec": total_train_time_sec,
        "inference_time_sec": inference_time_sec,
        "peak_gpu_memory_allocated_mb": peak_gpu_memory_mb,
    }

    return (
        train_acc,
        train_f1_micro,
        train_f1_macro,
        train_auc,
        val_acc,
        val_f1_micro,
        val_f1_macro,
        val_auc,
        test_acc,
        test_f1_micro,
        test_f1_macro,
        test_auc,
        hardware_metrics,
    )
