# /jobs/manager.py
import os
import sys
import json
import torch
import subprocess
import webbrowser
from backend.ETL.extract import extract_data
from utils.model_load import MultiModel
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from utils.helper import params, cuda_check, save_model, save_metrics, log_metrics
from utils.train import train
from utils.eval import eval

import warnings
warnings.filterwarnings("ignore")

def create_if_missing(*paths, verbose=False):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            if verbose:
                print(f"[Setup] Created: {path}")

def dir_setup():    
    add_on = f"/Workspace/Projects/Research/"
    data_dir = os.path.expanduser(f"~{add_on}cp-anemia-detection/data/")
    datasets = ['cp-anemia']
    weights_dir = os.path.expanduser(f"~~{add_on}cp-anemia-detection/output/weights")
    metrics_dir = os.path.expanduser(f"~~{add_on}cp-anemia-detection/output/metrics")

    dataset_dir = os.path.join(data_dir, datasets[0])
    anemic_dir = os.path.join(dataset_dir, "Anemic")
    non_anemic_dir = os.path.join(dataset_dir, "Non-anemic")
    edge_input_path = os.path.join(data_dir, "edge-input", "sample_img1.png")

    create_if_missing(dataset_dir, weights_dir, metrics_dir, anemic_dir, non_anemic_dir, os.path.dirname(edge_input_path), verbose=True)

    return dataset_dir, weights_dir, metrics_dir, edge_input_path

def launch_backend():
    print("[1] Launching controller API...")
    return subprocess.Popen(['uvicorn', 'server:app', '--port', '8000', '--reload'], cwd="distributed_training/controller")

def launch_dashboard():
    print("[2] Launching monitoring dashboard...")
    dashboard_root = os.path.abspath(os.path.dirname(__file__))
    webbrowser.open("http://localhost:5500/ml_dashboard.html")
    return subprocess.Popen(["python3", "-m", "http.server", "5500"], cwd=dashboard_root)

def main(config):

    dataset_dir, weights_dir, metrics_dir, edge_input_path = dir_setup()
    architecture, compression_mode, lr, batch_size, epochs, folds, cross_entropy_loss, mse_loss, mae_loss = params() 
    if config:
        print(f"[Main] Running with config: {config}")
        # Apply config values if provided
        batch_size = config.get("batchSize", batch_size)
        epochs = config.get("epochs", epochs)
        lr = config.get("learningRate", 1e-4)  # default fallback
        
    device = cuda_check()
    edge_input_path = "" # Mannual toggle on/off

    # Determine mode based on whether edge input file exists
    if os.path.exists(edge_input_path):
        print("[Data] Edge input mode detected.")
        dataloader = extract_data(edge_input_path, batch_size=1)
        print("[Edge Mode] Ready to run model on edge input batch")
        # for data in dataloader:
        #     print(data.shape)
    else:
        print("[Data] Dataset mode detected.")
        dataset, dataloader = extract_data(dataset_dir, batch_size=batch_size)
        train_dataset, test_dataset = dataset
        train_loader, test_loader = dataloader
        print("[Dataset Mode] Ready for training and evaluation.")

    # Set up 5-Fold Cross Validation
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)

    print("=" * 100)
    print(f"Training Model: {architecture}")
    
    model = MultiModel(architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = -float("inf")  # Track best validation accuracy
    train_metrics_list = []
    val_metrics_list = []

    # === MAIN LOOP ===
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        fold = 1

        for train_idx, val_idx in kf.split(range(len(train_dataset))):
            train_subset = Subset(train_dataset, train_idx)
            val_subset = Subset(train_dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, pin_memory=True)

            if fold == folds:
                # === VALIDATION PHASE ===
                phase = "validation"
                val_metrics, val_stats = eval(val_loader, model, cross_entropy_loss, mse_loss, mae_loss)
                log_metrics(log_type="print", phase=phase, epoch=epoch, fold=fold, metrics=val_metrics, hw_metrics=val_stats)

                if val_metrics[2] > best_val_acc:
                    best_val_acc = val_metrics[2]
                    save_model(score=best_val_acc, model=model, architecture=architecture, signature="", dir=weights_dir)

                # Store validation metrics
                val_metrics_dict = log_metrics(log_type="dict", phase=phase, epoch=epoch, fold=fold, metrics=val_metrics, hw_metrics=val_stats)
                val_metrics_list.append(val_metrics_dict)

            else:
                # === TRAINING PHASE ===
                phase = "training"
                model, train_metrics = train(train_loader, model, cross_entropy_loss, mse_loss, mae_loss, optimizer)
                log_metrics(log_type="print", phase=phase, epoch=epoch, fold=fold, metrics=train_metrics)

                # Store training metrics
                train_metrics_dict = log_metrics(log_type="dict", phase=phase, epoch=epoch, fold=fold, metrics=train_metrics)
                train_metrics_list.append(train_metrics_dict)

            fold += 1  # Move to next fold

        save_metrics(train_metrics_list, phase="training", dir=metrics_dir, architecture=architecture, signature="")
        save_metrics(val_metrics_list, phase="validation", dir=metrics_dir, architecture=architecture, signature="")
    
    # print(f"\nFine-tuned {get_model_size(model)}")
    print("=" * 100)

    return {
    "status": "complete",
    "epochs": epochs,
    "best_val_acc": best_val_acc,
    "train_folds": len(train_metrics_list),
    "val_folds": len(val_metrics_list),
    "metrics": {
        "training": train_metrics_list[-1] if train_metrics_list else {},
        "validation": val_metrics_list[-1] if val_metrics_list else {}}
    }

if __name__ == "__main__":
    config = {}
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config = json.load(f)

    main(config)
