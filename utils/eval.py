import torch
import torch.nn.functional as F
import numpy as np
import gc
from utils.model_metrics import compute_classification_metrics, compute_regression_metrics
from utils.helper import sw_loss, timed_forward
from compression_engine.ptq import apply_fp16, apply_static_ptq, apply_int4_awq
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, r2_score,
    mean_absolute_error, mean_squared_error
)


# def eval(dataloader, model, class_loss, reg1_loss, reg2_loss, quantization="base", precision="fp32", device="cuda"):
#     """Evaluates "the model with metrics: Precision, Recall, AUC, F1, R², Memory Usage, and Latency."""
#     model.to(device)
#     model.eval()

#     mean_stats = []
#     total_loss = 0
#     total_ce_loss = 0
#     total_mse_loss = 0
#     total_mae_loss = 0
#     correct = 0
#     total_samples = 0

#     all_preds = []
#     all_targets = []
#     all_probs = []
#     all_hb_targets = []
#     all_hb_preds = []

#     torch.cuda.empty_cache()
#     gc.collect()

#     if quantization == "ptq":
#         if precision == "fp16":
#             model = apply_fp16(model)

#         elif precision == "int8":
#             model = apply_static_ptq(model, dataloader)

#         elif precision == "int4":
#             model = apply_int4_awq(model, dataloader)

#     with torch.no_grad():
#         for _, (img_id, img, multiclass, hb_level) in enumerate(dataloader):
#             img = img.to(device)
#             multiclass = multiclass.to(device).long()
#             hb_level = hb_level.to(device).float()

#             if quantization == "ptq" and precision == "fp16":
#                 img, hb_level = img.half(), hb_level.half()

#             class_pred, reg_pred, stats = timed_forward(model, img)
#             mean_stats.append(stats)

#             ce_loss = class_loss(class_pred, multiclass)
#             mse_loss = reg1_loss(reg_pred, hb_level)
#             mae_loss = reg2_loss(reg_pred, hb_level)
#             loss = sw_loss(ce_loss, mae_loss, 0.7)
#             # loss = ce_loss

#             total_loss += loss.item()
#             total_ce_loss += ce_loss.item()
#             total_mse_loss += mse_loss.item()
#             total_mae_loss += mae_loss.item()

#             class_probs = F.softmax(class_pred, dim=1)
#             highest_prob_class = torch.argmax(class_probs, dim=1)

#             correct += (highest_prob_class == multiclass).sum().item()
#             total_samples += multiclass.size(0)

#             all_preds.extend(highest_prob_class.detach().cpu().numpy())
#             all_targets.extend(multiclass.detach().cpu().numpy())
#             all_probs.extend(class_probs.detach().cpu().numpy())
#             all_hb_targets.extend(hb_level.detach().cpu().numpy())
#             all_hb_preds.extend(reg_pred.squeeze().detach().cpu().numpy())

#     mean_latency = np.mean([s["latency"] for s in mean_stats])
#     mean_mem_before = np.mean([s["malloc_before"] for s in mean_stats]) / 1_048_576
#     mean_mem_after = np.mean([s["malloc_after"] for s in mean_stats]) / 1_048_576
#     mean_max_mem = np.mean([s["max_malloc"] for s in mean_stats]) / 1_048_576

#     final_mean_stats = [mean_latency, mean_mem_before, mean_mem_after, mean_max_mem]

#     clf_metrics = compute_classification_metrics(all_targets, all_preds, all_probs)
#     reg_metrics = compute_regression_metrics(all_hb_targets, all_hb_preds)

#     avg_loss = total_loss / len(dataloader)
#     avg_ce_loss = total_ce_loss / len(dataloader)
#     avg_mse_loss = total_mse_loss / len(dataloader)
#     avg_mae_loss = total_mae_loss / len(dataloader)
#     accuracy = correct / total_samples

#     final_metrics = [
#         avg_loss,
#         avg_ce_loss,
#         accuracy,
#         clf_metrics["precision"],
#         clf_metrics["recall"],
#         clf_metrics["f1"],
#         clf_metrics["auc"],
#         reg_metrics["r2"],
#         reg_metrics["mae"],
#         reg_metrics["mse"]
#     ]

#     return final_metrics, final_mean_stats

def eval(dataloader, model, class_loss, reg1_loss, reg2_loss, quantization="base", precision="fp32", device="cuda"):
    """Evaluates the model with additional metrics: Precision, Recall, AUC, F1, R², Memory Usage, and Latency."""
    model.to(device)
    model.eval()
    mean_stats = []

    total_loss = 0
    total_ce_loss = 0
    total_mse_loss = 0
    total_mae_loss = 0
    correct = 0
    total_samples = 0

    all_preds = []
    all_targets = []
    all_probs = []
    all_hb_targets = []
    all_hb_preds = []

    torch.cuda.empty_cache()
    gc.collect()

    if quantization == "ptq":
        if precision == "fp16":
            model = apply_fp16(model)

        elif precision == "int8":
            model = apply_static_ptq(model, dataloader)

        elif precision == "int4":
            model = apply_int4_awq(model, dataloader)

    with torch.no_grad():
        for _, (_, img, multiclass, hb_level) in enumerate(dataloader):
            img = img.to(device)
            multiclass = multiclass.to(device).long()
            hb_level = hb_level.to(device).unsqueeze(1).float()

            # Forward pass with latency & memory tracking
            class_pred, reg_pred, stats = timed_forward(model, img)
            mean_stats.append(stats)

            # Compute losses
            ce_loss = class_loss(class_pred, multiclass)
            mse_loss = reg1_loss(reg_pred, hb_level)
            mae_loss = reg2_loss(reg_pred, hb_level)
            loss = sw_loss(ce_loss, mse_loss, 0.7)

            # Track total losses
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_mse_loss += mse_loss.item()
            total_mae_loss += mae_loss.item()

            # Compute classification accuracy
            class_probs = F.softmax(class_pred, dim=1)
            highest_prob_class = torch.argmax(class_probs, dim=1)

            correct += (highest_prob_class == multiclass).sum().item()
            total_samples += multiclass.size(0)

            # Collect data for additional metrics
            all_preds.extend(highest_prob_class.detach().cpu().numpy())
            all_targets.extend(multiclass.detach().cpu().numpy())
            all_probs.extend(class_probs.detach().cpu().numpy())
            all_hb_targets.extend(hb_level.detach().cpu().numpy())
            all_hb_preds.extend(reg_pred.squeeze().detach().cpu().numpy())

    # Compute mean statistics
    mean_latency = np.mean([s["latency"] for s in mean_stats])
    mean_mem_before = np.mean([s["malloc_before"] for s in mean_stats]) / 1_048_576  # Convert bytes to MB
    mean_mem_after = np.mean([s["malloc_after"] for s in mean_stats]) / 1_048_576  # Convert bytes to MB
    mean_max_mem = np.mean([s["max_malloc"] for s in mean_stats]) / 1_048_576  # Convert bytes to MB

    # Store final mean statistics
    final_mean_stats = [mean_latency, mean_mem_before, mean_mem_after, mean_max_mem]

    # Compute additional evaluation metrics
    precision = precision_score(all_targets, all_preds, average="weighted")
    recall = recall_score(all_targets, all_preds, average="weighted")
    f1 = f1_score(all_targets, all_preds, average="weighted")
    auc = roc_auc_score(all_targets, all_probs, multi_class="ovr")
    r2 = r2_score(all_hb_targets, all_hb_preds)

    # Compute confusion matrix
    # cm = confusion_matrix(all_targets, all_preds)

    # Compute final average losses
    avg_loss = total_loss / len(dataloader)
    avg_ce_loss = total_ce_loss / len(dataloader)
    avg_mse_loss = total_mse_loss / len(dataloader)
    avg_mae_loss = total_mae_loss / len(dataloader)
    accuracy = correct / total_samples

    # Store metrics
    final_metrics = [avg_loss, avg_ce_loss, accuracy, precision, recall, f1, auc, r2, avg_mae_loss, avg_mse_loss]

    return final_metrics, final_mean_stats