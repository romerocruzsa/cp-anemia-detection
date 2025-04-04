import os
import torch
from backend.ETL.extract import extract_data
from compression_engine.qat import apply_qat_fx
from compression_engine.prune import PruningScheduler
from compression_engine.kd import EnsembleSelfDistillation
from utils.model_load import MultiModel
from utils.helper import train_config, input_train_config, print_train_config, cuda_check, save_model, save_metrics, log_metrics, get_model_size, clear_folder
from utils.train import train
from utils.eval import eval
from torch.utils.data import DataLoader, Subset
# from torch.quantization.quantize import convert
from sklearn.model_selection import KFold

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

    dataset_dir = os.path.join(data_dir, datasets[0])
    anemic_dir = os.path.join(dataset_dir, "Anemic")
    non_anemic_dir = os.path.join(dataset_dir, "Non-anemic")
    edge_input_path = os.path.join(data_dir, "edge-input", "sample_img1.png")

    create_if_missing(dataset_dir, weights_dir, metrics_dir, anemic_dir, non_anemic_dir, os.path.dirname(edge_input_path), verbose=True)

    return dataset_dir, weights_dir, metrics_dir, edge_input_path

def main():
    dataset_dir, weights_dir, metrics_dir, edge_input_path = dir_config()
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
    
        best_val_acc = -float("inf")  # Track best validation accuracy
        train_metrics_list = []
        val_metrics_list = []


        # === MAIN LOOP ===
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            fold = 1

            for train_idx, val_idx in kf.split(range(len(train_dataset))):

                model = MultiModel(architecture).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                if quantization_mode == "qat":
                    model = apply_qat_fx(model, train_loader)

                if pruning_mode != "base":
                    if pruning_mode == "unstructured":
                        prune_scheduler = PruningScheduler(
                            model=model,
                            start_epoch=2,
                            end_epoch=int(epochs * 0.8),
                            interval=2,
                            amount=0.2,
                            mode="unstructured")
                        if epoch >= 2:
                            prune_scheduler.step(epoch)
                    
                    elif pruning_mode == "structured":
                        prune_scheduler = PruningScheduler(
                            model=model,
                            start_epoch=2,
                            end_epoch=int(epochs * 0.8),
                            interval=2,
                            amount=0.2,
                            mode="structured")
                        if epoch >= 2:
                            prune_scheduler.step(epoch)
                    
                if distillation_mode == "self-distil":
                    clear_folder(os.path.expanduser("~/cp-anemia-detection/compression_engine/kd_checkpoints"))
                    distiller = EnsembleSelfDistillation(
                        model_fn=lambda: MultiModel(architecture),
                        save_dir='compression_engine/kd_checkpoints/',
                        max_teachers=3,
                        device=device
                    )
                else:
                    distiller = None

                train_subset = Subset(train_dataset, train_idx)
                val_subset = Subset(train_dataset, val_idx)

                train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=True)
                val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, pin_memory=True)

                if fold == folds:
                    # === VALIDATION PHASE ===
                    phase = "validation"
                    val_metrics, val_stats = eval(val_loader, model, cross_entropy_loss, mse_loss, mae_loss, quantization=quantization_mode, device="cpu")
                    log_metrics(log_type="print", phase=phase, epoch=epoch, fold=fold, metrics=val_metrics, hw_metrics=val_stats)

                    if val_metrics[2] > best_val_acc:
                        best_val_acc = val_metrics[2]
                        save_model(score=best_val_acc, model=model, architecture=architecture, signature=signature, dir=weights_dir, distillation=distillation_mode, quantization=quantization_mode, pruning=pruning_mode)

                        if distillation_mode == "self-distil":
                            distiller.update_top_models(model, best_val_acc, epoch)

                    # Store validation metrics
                    val_metrics_dict = log_metrics(log_type="dict", phase=phase, epoch=epoch, fold=fold, metrics=val_metrics, hw_metrics=val_stats)
                    val_metrics_list.append(val_metrics_dict)

                else:
                    # === TRAINING PHASE ===
                    phase = "training"
                    model, train_metrics = train(train_loader, model, cross_entropy_loss, mse_loss, mae_loss, optimizer, distiller=distiller, quantization=quantization_mode, device=device)
                    log_metrics(log_type="print", phase=phase, epoch=epoch, fold=fold, metrics=train_metrics)

                    # Store training metrics
                    train_metrics_dict = log_metrics(log_type="dict", phase=phase, epoch=epoch, fold=fold, metrics=train_metrics, hw_metrics=None)
                    train_metrics_list.append(train_metrics_dict)

                fold += 1  # Move to next fold

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