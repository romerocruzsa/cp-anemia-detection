import os
import torch
from backend.ETL.extract import extract_data
from utils.cnn_load import MultiModel
from utils.compression_load import compression_config
from utils.helper import (input_train_config, print_train_config, cuda_check,
                           save_model, save_metrics, log_metrics, get_model_size, 
                           clear_folder, kde_undersample_subset, kde_balance_by_severity)
from utils.train import train
from utils.eval import eval
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, StratifiedKFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.early_stopping import EarlyStopping
from torch.utils.tensorboard import SummaryWriter

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
    datasets = "fingernail-anemia"
    weights_dir = os.path.expanduser("~/cp-anemia-detection/output/weights")
    metrics_dir = os.path.expanduser("~/cp-anemia-detection/output/metrics")
    checkpoints_dir = os.path.expanduser("~/cp-anemia-detection/output/checkpoints")

    dataset_dir = os.path.join(data_dir, datasets)
    anemic_dir = os.path.join(dataset_dir, "Anemic")
    non_anemic_dir = os.path.join(dataset_dir, "Non-anemic")
    edge_input_path = os.path.join(data_dir, "edge-input", "sample_img1.png")

    create_if_missing(dataset_dir, weights_dir, metrics_dir, anemic_dir, non_anemic_dir, os.path.dirname(edge_input_path), verbose=True)

    return dataset_dir, weights_dir, metrics_dir, checkpoints_dir, edge_input_path

def train_config():
    dataset_dir, weights_dir, metrics_dir, checkpoints_dir, edge_input_path = dir_config()
    # edge_input_path = "/Users/romerocruzsa/Workspace/Projects/Research/cp-anemia-detection/data/edge-input/sample_img2.png" # Mannual toggle on/off
    edge_input_path = "" # Mannual toggle on/off

    # Determine mode based on whether edge input file exists
    if os.path.exists(edge_input_path):
        print("[Data] Edge input mode detected.")
        dataloader = extract_data("edge-input", dataset_dir=edge_input_path, batch_size=1)
        print("[Edge Mode] Ready to run model on edge input batch")
        return dataloader

    else:
        print(f"[Data] Dataset mode detected.")
        if dataset_dir.endswith("fingernail-anemia"):
            dataset, dataloader = extract_data(dataset_type="fingernail-anemia",
                                                dataset_dir=dataset_dir,
                                                batch_size=32)
            train_dataset, test_dataset = dataset
            train_loader, test_loader = dataloader
            print("[Dataset: Fingernail-Anemia] Ready for training and evaluation.")
            return train_dataset, train_loader, test_dataset, test_loader


def main():
    dataset_dir, weights_dir, metrics_dir, checkpoints_dir, edge_input_path = dir_config()
    print(os.path.exists(edge_input_path))
    if edge_input_path != "":
        dataloader = train_config()
        print(dataloader)
        import pdb;pdb.set_trace()

    else:
        train_dataset, train_loader, test_dataset, test_loader = train_config()
    log_dir = os.path.expanduser("~/cp-anemia-detection/output/logs")

    writer = SummaryWriter(log_dir=log_dir)
    device = cuda_check()

    script = os.path.expanduser("~/cp-anemia-detection/scripts/base_testing.sh")
    with open(script, "r") as f:
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

        print("=" * 100)
        print(f"Training Model: {architecture}")
        torch.cuda.empty_cache()
    
        train_metrics_list = []
        val_metrics_list = []

        epochs_per_fold = epochs // folds

        df = train_dataset.df.copy()

        # Apply KDE balancing before CV
        df_balanced = kde_balance_by_severity(
            df,
            severity_column="RemarkClass",
            target_column="HB_LEVEL_GperL",
            n_per_class=50,
            seed=42
        )
        
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(skf.split(df_balanced, df_balanced["RemarkClass"]), 1):

            print(f"\n===== \tFold {fold}/{folds} \t=====")

            train_df = df_balanced.iloc[train_idx].reset_index(drop=True)
            val_df = df_balanced.iloc[val_idx].reset_index(drop=True)

            train_subset = train_dataset.__class__(train_dataset.base_dir, train_df, transform=train_dataset.transform)
            val_subset = train_dataset.__class__(train_dataset.base_dir, val_df, transform=train_dataset.transform)

            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, pin_memory=True)

            print(f"Dataset loaded — Balanced Train: {len(train_subset)}, Test: {len(val_subset)}")

            best_val_loss = float("inf")  # Track best validation accuracy

            # model = MultiModel(architecture).to(device)
            model = MultiModel().to(device)
            # import pdb; pdb.set_trace()
            optimizer = torch.optim.Adam([
                {'params': model.encoder_nail.parameters(), 'lr': 1e-4},
                # {'params': model.encoder_skin.parameters(), 'lr': 1e-4},
                {'params': model.classifier_head.parameters(), 'lr': 1e-4},
                {'params': model.quantile_head.parameters(), 'lr': 1e-4}
            ])
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.1,       # big drop from 1e-4 → 1e-5
                patience=5,       # decay quickly if val loss stagnates
                cooldown=1,       # wait 1 epoch before monitoring again
                min_lr=1e-6,
                verbose=True
            )

            model, pruning_scheduler, distiller = compression_config(model,
                                                                   architecture,
                                                                   quantization_mode,
                                                                   pruning_mode, 
                                                                   distillation_mode, 
                                                                   epochs,
                                                                   train_loader)
            
            early_stopper = EarlyStopping(patience=30, delta=0.002, mode="min")

            # === MAIN LOOP ===
            for epoch in range(epochs_per_fold):
                print(f"\n===== \tEpoch {epoch+1}/{epochs_per_fold} ===== Total Epochs: {epochs} \t=====")

                if pruning_scheduler:
                    pruning_scheduler.step(epoch)

                # === TRAINING PHASE ===
                phase = "training"
                model, train_metrics = train(train_loader,
                                             model,
                                             cross_entropy_loss,
                                             mse_loss, mae_loss,
                                             optimizer,
                                             distiller=distiller,
                                             quantization=quantization_mode,
                                             device="cuda:0")
                log_metrics(log_type="print", phase=phase, epoch=epoch, fold=fold, metrics=train_metrics, architecture=architecture, writer=writer)

                # Store training metrics
                train_metrics_dict = log_metrics(log_type="dict", phase=phase, epoch=epoch, fold=fold, metrics=train_metrics, architecture=architecture, hw_metrics=None)
                train_metrics_list.append(train_metrics_dict)

                # === VALIDATION PHASE ===
                phase = "validation"
                val_metrics, val_stats = eval(val_loader,
                                              model,
                                              cross_entropy_loss,
                                              mse_loss,
                                              mae_loss,
                                              quantization=quantization_mode,
                                              device="cpu")
                log_metrics(log_type="print", phase=phase, epoch=epoch, fold=fold, metrics=val_metrics, architecture=architecture, hw_metrics=val_stats, writer=writer)

                if val_metrics[0] < best_val_loss:
                    best_val_loss = val_metrics[0]

                    if distillation_mode == "self-distil":
                        distiller.update_top_models(model, best_val_loss, epoch)
                    save_model(score=best_val_loss, model=model, architecture=architecture, signature=signature, dir=weights_dir, distillation=distillation_mode, quantization=quantization_mode, pruning=pruning_mode)
                
                scheduler.step(val_metrics[0])
                early_stopper(val_metrics[0], train_f1=train_metrics[5], val_f1=val_metrics[5], model=model)

                if early_stopper.early_stop:
                    print(f"[EarlyStopping] Triggered at epoch {epoch+1} due to validation stagnation or F1 gap.")
                    break
                
                # Store validation metrics
                val_metrics_dict = log_metrics(log_type="dict", phase=phase, epoch=epoch, fold=fold, metrics=val_metrics, architecture=architecture, hw_metrics=val_stats)
                val_metrics_list.append(val_metrics_dict)

            save_metrics(train_metrics_list,
                         phase="training",
                         dir=metrics_dir,
                         architecture=architecture,
                         signature=signature,
                         distillation=distillation_mode,
                         quantization=quantization_mode,
                         pruning=pruning_mode)
            save_metrics(val_metrics_list,
                         phase="validation",
                         dir=metrics_dir,
                         architecture=architecture,
                         signature=signature,
                         distillation=distillation_mode,
                         quantization=quantization_mode,
                         pruning=pruning_mode)
        
        phase = "testing"
        test_metrics, test_stats = eval(test_loader,
                                              model,
                                              cross_entropy_loss,
                                              mse_loss,
                                              mae_loss,
                                              quantization=quantization_mode,
                                              device="cpu") 
        log_metrics(log_type="print", phase=phase, epoch=epoch, fold=fold, metrics=test_metrics, architecture=architecture, hw_metrics=test_stats, writer=writer)
        log_metrics(log_type="dict", phase=phase, epoch=epoch, fold=fold, metrics=test_metrics, architecture=architecture, hw_metrics=test_stats)
        save_metrics(val_metrics_list,
                         phase=phase,
                         dir=metrics_dir,
                         architecture=architecture,
                         signature=signature,
                         distillation=distillation_mode,
                         quantization=quantization_mode,
                         pruning=pruning_mode)

if __name__ == "__main__":
    torch.manual_seed(42)
    main()