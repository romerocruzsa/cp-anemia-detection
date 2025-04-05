import os
import torch
from backend.ETL.extract import extract_data
from compression_engine.qat import apply_qat_fx
from compression_engine.prune import PruningScheduler
from compression_engine.kd import EnsembleSelfDistillation
from utils.model_load import MultiModel
from utils.compression_load import compression_config
from utils.helper import input_train_config, print_train_config, cuda_check, save_model, save_metrics, log_metrics, get_model_size, clear_folder
from utils.train import train
from utils.eval import eval
from torch.utils.data import DataLoader, Subset
# from torch.quantization.quantize import convert
from sklearn.model_selection import KFold
from utils.early_stopping import EarlyStopping

import warnings
warnings.filterwarnings("ignore")

def create_if_missing(*paths, verbose=False):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            if verbose:
                print(f"[Setup] Created: {path}")

def dir_config():    
    data_dir = os.path.expanduser("~/cp-anemia-detection/data/")
    datasets = ['cp-anemia']
    weights_dir = os.path.expanduser("~/cp-anemia-detection/output/weights")
    metrics_dir = os.path.expanduser("~/cp-anemia-detection/output/metrics")
    checkpoints_dir = os.path.expanduser("~/cp-anemia-detection/output/checkpoints")

    dataset_dir = os.path.join(data_dir, datasets[0])
    anemic_dir = os.path.join(dataset_dir, "Anemic")
    non_anemic_dir = os.path.join(dataset_dir, "Non-anemic")
    edge_input_path = os.path.join(data_dir, "edge-input", "sample_img1.png")

    create_if_missing(dataset_dir, weights_dir, metrics_dir, anemic_dir, non_anemic_dir, os.path.dirname(edge_input_path), verbose=True)

    return dataset_dir, weights_dir, metrics_dir, checkpoints_dir, edge_input_path

def main():
    dataset_dir, weights_dir, metrics_dir, checkpoints_dir, edge_input_path = dir_config()
    # architecture, signature, quantization_mode, precision, pruning_mode, distillation_mode, batch_size, epochs, folds, cross_entropy_loss, mse_loss, mae_loss = train_config()

    # Load from sweeps.sh
    with open("main.sh", "r") as f:
        sweep_lines = [
            line.strip() for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    # Run each config
    for sweep_line in sweep_lines:
        (architecture, signature, quantization_mode, precision, 
        pruning_mode, distillation_mode, batch_size, epochs, folds, 
        cross_entropy_loss, mse_loss, mae_loss) = input_train_config(sweep_line)

        print_train_config(input_train_config(sweep_line))

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
    
        train_metrics_list = []
        val_metrics_list = []

        epochs_per_fold = epochs // folds

        for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(train_dataset))), 1):
            print(f"\n===== \tFold {fold}/{folds} \t=====")

            best_val_acc = -float("inf")  # Track best validation accuracy

            model = MultiModel(architecture).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            checkpoint_path = os.path.join(checkpoints_dir, f"{architecture}_{signature}_{quantization_mode}_{pruning_mode}_{distillation_mode}_fold{fold}.pt")
            if fold > 1 and best_val_acc > 0.75:
                prev_path = os.path.join(checkpoints_dir, f"{architecture}_{signature}_{quantization_mode}_{pruning_mode}_{distillation_mode}_fold{fold-1}.pt")
                if os.path.exists(prev_path):
                    print(f"[Checkpoint] Loading previous best model from fold {fold-1}")
                    model.load_state_dict(torch.load(prev_path))

            train_subset = Subset(train_dataset, train_idx)
            val_subset = Subset(train_dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, pin_memory=True)

            model, prune_scheduler, distiller = compression_config(model,
                                                                   architecture,
                                                                   quantization_mode,
                                                                   pruning_mode, 
                                                                   distillation_mode, 
                                                                   epochs,
                                                                   train_loader)
            
            early_stopper = EarlyStopping(patience=12, delta=0.002, mode="max")

            # === MAIN LOOP ===
            for epoch in range(epochs_per_fold):
                print(f"\n===== \tEpoch {epoch+1}/{epochs_per_fold} ===== Total Epochs: {epochs} \t=====")

                # === TRAINING PHASE ===
                phase = "training"
                model, train_metrics = train(train_loader, model, cross_entropy_loss, mse_loss, mae_loss, optimizer, distiller=distiller, quantization=quantization_mode, device=device)
                log_metrics(log_type="print", phase=phase, epoch=epoch, fold=fold, metrics=train_metrics)

                # Store training metrics
                train_metrics_dict = log_metrics(log_type="dict", phase=phase, epoch=epoch, fold=fold, metrics=train_metrics, hw_metrics=None)
                train_metrics_list.append(train_metrics_dict)

                # === VALIDATION PHASE ===
                phase = "validation"
                val_metrics, val_stats = eval(val_loader, model, cross_entropy_loss, mse_loss, mae_loss, quantization=quantization_mode, device="cpu")
                log_metrics(log_type="print", phase=phase, epoch=epoch, fold=fold, metrics=val_metrics, hw_metrics=val_stats)

                if val_metrics[2] > best_val_acc:
                    best_val_acc = val_metrics[2]

                    if distillation_mode == "self-distil":
                        distiller.update_top_models(model, best_val_acc, epoch)
                    save_model(score=best_val_acc, model=model, architecture=architecture, signature=signature, dir=weights_dir, distillation=distillation_mode, quantization=quantization_mode, pruning=pruning_mode)
                    torch.save(model.state_dict(), checkpoint_path)

                early_stopper(val_metrics[2], train_f1=train_metrics[5], val_f1=val_metrics[5], model=model)
                if early_stopper.early_stop:
                    print(f"[EarlyStopping] Triggered at epoch {epoch+1} due to validation stagnation or F1 gap.")
                    break

                # Store validation metrics
                val_metrics_dict = log_metrics(log_type="dict", phase=phase, epoch=epoch, fold=fold, metrics=val_metrics, hw_metrics=val_stats)
                val_metrics_list.append(val_metrics_dict)

            save_metrics(train_metrics_list, phase="training", dir=metrics_dir, architecture=architecture, signature=signature, distillation=distillation_mode, quantization=quantization_mode, pruning=pruning_mode)
            save_metrics(val_metrics_list, phase="validation", dir=metrics_dir, architecture=architecture, signature=signature, distillation=distillation_mode, quantization=quantization_mode, pruning=pruning_mode)
            
        if quantization_mode == "qat":
            print(f"\n[{quantization_mode.upper()}] Converting to FX Graph Mode")
            save_model(score=best_val_acc, model=model, architecture=architecture, signature=signature, dir=weights_dir, distillation=distillation_mode, quantization=quantization_mode, pruning=pruning_mode)
        else:
            print(f"\nFine-tuned {get_model_size(model)}")
        print("=" * 100)

if __name__ == "__main__":
    main()