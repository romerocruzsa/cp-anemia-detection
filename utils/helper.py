import os
import shutil
import torch
import gc
import csv
from torch.ao.quantization.quantize_fx import convert_fx
import numpy as np
from scipy.stats import gaussian_kde
from torchvision.transforms import ToPILImage
from PIL import ImageDraw, ImageFont

def input_train_config(config_line):
    args = config_line.strip().split()
    architecture = args[0]
    quantization_mode = args[2]
    pruning_mode = args[3]
    distillation_mode = args[4]
    precision = "fp32"
    signature = args[1]
    batch_size = 32
    folds = 5
    epochs = 150*folds

    # cross_entropy_loss = torch.nn.CrossEntropyLoss()
    cross_entropy_loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    mse_loss = torch.nn.MSELoss()
    mae_loss = torch.nn.L1Loss()

    return (architecture, signature, quantization_mode, precision, 
            pruning_mode, distillation_mode, batch_size, epochs, folds, 
            cross_entropy_loss, mse_loss, mae_loss)


def print_train_config(config):
    print("="*100)
    print(f"[Setup] Model Architecture:     \t\t{config[0]}")
    print(f"[Setup] Signature:              \t\t{config[1]}")
    print(f"[Setup] Quantization Mode:      \t\t{config[2]}")
    print(f"[Setup] Precision:              \t\t{config[3]}")
    print(f"[Setup] Pruning Mode:           \t\t{config[4]}")
    print(f"[Setup] Distillation Mode:      \t\t{config[5]}")
    print(f"[Setup] Batch Size:             \t\t{config[6]}")
    print(f"[Setup] Epochs:                 \t\t{config[7]}")
    print(f"[Setup] Cross-validation Folds: \t\t{config[8]}")
    print(f"[Setup] Classification Loss:    \t\t{config[9]}")
    print(f"[Setup] Regression Loss (1):    \t\t{config[10]}")
    print(f"[Setup] Regression Loss (2):    \t\t{config[11]}")
    print("="*100)

def cuda_check():
    # Default device
    global device
    device = torch.device('cpu')

    # Check for CUDA availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("[Setup] CUDA is not available, using CPU.")

    print(f"[Setup] Selected device: {device}")
    return device

def clear_folder(folder_path):
    """
    Deletes all contents inside the given folder without removing the folder itself.
    
    Args:
        folder_path (str): Path to the folder to clear.
    """
    if not os.path.exists(folder_path):
        print(f"[Warning] Folder '{folder_path}' does not exist.")
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove file or symbolic link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Recursively delete directory
        except Exception as e:
            print(f"[Error] Failed to delete {file_path}. Reason: {e}")


def get_model_size(model, model_type="pytorch", model_path="model.onnx"):
    """Returns model size in MB"""
    if model_type == "pytorch":
        torch.save(model, "tmp.pth")
        model_size = os.path.getsize("tmp.pth") / 1e6  # Convert bytes to MB
        os.remove("tmp.pth")
    return f"Model Size: {model_size:.2f} MB"

def calibrate_loop(model, calibration_loader):
    for img, _, _ in calibration_loader:
        model(img.to("cpu"))

# Function to measure inference time & memory
def timed_forward(model, nail_tensor, skin_tensor):
    """Measures inference time and memory usage for PyTorch, ONNX, and TensorRT."""
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Clear cache
    torch.cuda.empty_cache()
    gc.collect()

    # Record memory usage before inference
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    max_mem_before = torch.cuda.max_memory_allocated()

    # Start measuring latency
    start_event.record()

    class_pred, reg_pred = model(nail_tensor, skin_tensor)

    end_event.record()
    torch.cuda.synchronize()  # Ensure accurate timing
    latency = start_event.elapsed_time(end_event)  # Time in ms

    # Record memory usage after inference
    mem_after = torch.cuda.memory_allocated()
    max_mem_after = torch.cuda.max_memory_allocated()

    # Store stats
    stats = {
        "latency": latency,
        "malloc_before": mem_before,
        "malloc_after": mem_after,
        "max_malloc": max_mem_after,
    }

    return class_pred, reg_pred, stats

# Static Weighting Function. Set eta_class to desired importance (Classification > .5, Regression < .5, Equal == .5)
def sw_loss(loss_class, loss_reg, eta_class=0.5):
    eta_reg = 1 - eta_class
    total_loss = (eta_class * loss_class) + (eta_reg * loss_reg)
    return total_loss

def quantile_loss(preds, targets, quantiles=[0.25, 0.5, 0.75]):
    loss = 0
    for i, q in enumerate(quantiles):
        errors = targets - preds[:, i]
        loss += torch.max((q - 1) * errors, q * errors).mean()
    return loss

def save_model(score, model, architecture, signature, dir, distillation="base", quantization="base", pruning="base"):
    # Save best model based on validation accuracy        
    if quantization == "qat":
        qat_model = convert_fx(model.to("cpu"))
        torch.save(
            qat_model.state_dict(),
            f"{dir}/model_best_loss_{architecture}_{signature}_{distillation}_{quantization}_{pruning}.pth",
        )
    else:
        torch.save(
            model.state_dict(),
            f"{dir}/model_best_loss_{architecture}_{signature}_{distillation}_{quantization}_{pruning}.pth",
        )
    print(f"Best model saved with Loss: {score:.4f}")

def log_metrics(log_type, phase, epoch, fold, metrics, architecture, hw_metrics=None, writer=None):
    if log_type == "print":
        print(f"{phase.capitalize()}: Fold {fold} - Total Loss: {metrics[0]:.4f}, Cross Entropy: {metrics[1]:4f}, Accuracy: {metrics[2]:.4f}, "
            f"Precision: {metrics[3]:.4f}, Recall: {metrics[4]:.4f}, F1 Score: {metrics[5]:.4f}, AUC: {metrics[6]:.4f}, "
            f"R2 Score: {metrics[7]:4f}, MAE: {metrics[8]:.4f}, MSE: {metrics[9]:.4f}")
        writer.add_scalar(f"{architecture}/Fold_{fold}/{phase}/Total_Loss", metrics[0], epoch)
        writer.add_scalar(f"{architecture}/Fold_{fold}/{phase}/Accuracy", metrics[2], epoch)
        writer.add_scalar(f"{architecture}/Fold_{fold}/{phase}/F1", metrics[5], epoch)
        writer.add_scalar(f"{architecture}/Fold_{fold}/{phase}/MAE", metrics[8], epoch)
        writer.add_scalar(f"{architecture}/Fold_{fold}/{phase}/MSE", metrics[9], epoch)
        
        if phase == "validation":
            print(f"Avg Latency (ms): {hw_metrics[0]:.2f}, Avg Memory Before (MB): {hw_metrics[1]:.2f}, "
                f"Avg Memory After (MB): {hw_metrics[2]:.2f}, Avg Max Memory (MB): {hw_metrics[3]:.2f}")
        
    if log_type == "dict":
        if hw_metrics == None:
            metrics_dict = {
                        "epoch": epoch + 1,
                        "fold": fold,
                        "total_loss": metrics[0],
                        "cross_entropy_loss": metrics[1],
                        "accuracy": metrics[2],
                        "precision": metrics[3],
                        "recall": metrics[4],
                        "f1_score": metrics[5],
                        "auc": metrics[6],
                        "r2_score": metrics[7],
                        "mae_loss": metrics[8],
                        "mse_loss": metrics[9]} 
        else:
            metrics_dict = {
                            "epoch": epoch + 1,
                            "fold": fold,
                            "total_loss": metrics[0],
                            "cross_entropy_loss": metrics[1],
                            "accuracy": metrics[2],
                            "precision": metrics[3],
                            "recall": metrics[4],
                            "f1_score": metrics[5],
                            "auc": metrics[6],
                            "r2_score": metrics[7],
                            "mae_loss": metrics[8],
                            "mse_loss": metrics[9],
                            "latency": hw_metrics[0],
                            "malloc_before": hw_metrics[1],
                            "malloc_after": hw_metrics[2],
                            "max_malloc": hw_metrics[3]}          
        return metrics_dict
    
def save_metrics(metrics, dir, phase, architecture, signature, distillation="base", quantization="base", pruning="base"):
    keys = metrics[0].keys()
    with open(f"{dir}/{phase}/{phase}_metrics_{architecture}_{signature}_{distillation}_{quantization}_{pruning}.csv", 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(metrics)

def kde_undersample_subset(dataset, target_attr="hb_level", n_samples=100, bandwidth=0.5):
    """
    Perform KDE-based under-sampling on a Subset or full Dataset.

    Args:
        dataset: torch.utils.data.Subset or Dataset
        target_attr: attribute returned by dataset.__getitem__ to use for sampling (e.g., Hb value)
        n_samples: number of points to retain
        bandwidth: KDE bandwidth in same units as target

    Returns:
        A new Subset with under-sampled indices
    """
    targets = []
    for i in range(len(dataset)):
        try:
            _, _, _, _, hb_level = dataset[i]
            targets.append(float(hb_level))
        except Exception:
            continue  # fallback if sample is broken

    targets = np.array(targets)
    kde = gaussian_kde(targets, bw_method=bandwidth / np.std(targets))
    probs = 1.0 / (kde(targets) + 1e-6)
    probs /= probs.sum()
    sampled_indices = np.random.choice(len(dataset), size=n_samples, replace=False, p=probs)

    return torch.utils.data.Subset(dataset, sampled_indices)