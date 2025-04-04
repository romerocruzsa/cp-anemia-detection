import os
import torch
import gc
import time
import psutil
import csv
from torch.ao.quantization.quantize_fx import convert_fx

def params():
    architecture = "mobilenetv2"
    compression_mode = "base"
    lr = 0.001
    batch_size = 4
    epochs = 1
    folds = 2

    # Define loss functions
    cross_entropy_loss = torch.nn.CrossEntropyLoss()  # Multi-class classification loss
    mse_loss = torch.nn.MSELoss()  # Regression loss
    mae_loss = torch.nn.L1Loss()  # Regression loss

    return architecture, compression_mode, lr, batch_size, epochs, folds, cross_entropy_loss, mse_loss, mae_loss

def cuda_check():
    global device
    device = torch.device("cpu")  # Default fallback

    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    #     print("[Setup] CUDA is available. Using GPU.")
    # elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    #     device = torch.device("mps")
    #     print("[Setup] Apple MPS backend is available. Using MPS.")
    # else:
    #     print("[Setup] No GPU found. Using CPU.")

    print(f"[Setup] Selected device: {device}")
    return device


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
def timed_forward(model, img):
    """Measures inference time and memory usage across CPU, CUDA, and MPS."""
    device = next(model.parameters()).device

    gc.collect()
    process = psutil.Process(os.getpid())

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()
    else:
        mem_before = process.memory_info().rss  # in bytes

    # Start timing
    start_time = time.perf_counter()

    with torch.no_grad():
        class_pred, reg_pred = model(img)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.perf_counter()
    latency = (end_time - start_time) * 1000  # ms

    if device.type == "cuda":
        mem_after = torch.cuda.memory_allocated()
        max_mem = torch.cuda.max_memory_allocated()
    else:
        mem_after = process.memory_info().rss
        max_mem = mem_after  # no peak available natively on CPU

    stats = {
        "device": device.type,
        "latency_ms": latency,
        "mem_before_mb": mem_before / (1024 ** 2),
        "mem_after_mb": mem_after / (1024 ** 2),
        "max_mem_mb": max_mem / (1024 ** 2),
    }

    return class_pred, reg_pred, stats

# Static Weighting Function. Set eta_class to desired importance (Classification > .5, Regression < .5, Equal == .5)
def sw_loss(loss_class, loss_reg, eta_class=0.5):
    eta_reg = 1 - eta_class
    total_loss = (eta_class * loss_class) + (eta_reg * loss_reg)
    return total_loss

def save_model(score, model, architecture, signature, dir, mode="base"):
    # Save best model based on validation accuracy        
    if mode == "qat":
        qat_model = convert_fx(model.to("cpu"))
        torch.save(
            qat_model.state_dict(),
            f"{dir}/model_best_accuracy_{architecture}_{signature}_{mode.upper()}.pth",
        )
    else:
        torch.save(
            model.state_dict(),
            f"{dir}/model_best_accuracy_{architecture}_{signature}_{mode.upper()}.pth",
        )
    print(f"Best model saved with Accuracy: {score:.4f}")

def log_metrics(log_type, phase, epoch, fold, metrics, hw_metrics=None):
    if log_type == "print":
        print(f"{phase.capitalize()}: Fold {fold} - Total Loss: {metrics[0]:.4f}, Cross Entropy: {metrics[1]:4f}, Accuracy: {metrics[2]:.4f}, "
            f"Precision: {metrics[3]:.4f}, Recall: {metrics[4]:.4f}, F1 Score: {metrics[5]:.4f}, AUC: {metrics[6]:.4f}, "
            f"R2 Score: {metrics[7]:4f}, MAE: {metrics[8]:.4f}, MSE: {metrics[9]:.4f}")
        
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
    
def save_metrics(metrics, dir, phase, architecture, signature, mode="base"):
    keys = metrics[0].keys()
    with open(f"{dir}/{phase}_metrics_{architecture}_{signature}_{mode}.csv", 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(metrics)