import sys
import os
import argparse
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

# Import custom modules (assuming these contain your functions and classes)
from preprocessing import load_data
from train import train, eval
from metrics import CumulativeMetrics
from model import load_mobilenet

def parse_arguments():
    parser = argparse.ArgumentParser(description="Training and Evaluation Pipeline for Anemia Detection")
    parser.add_argument("--task_type", type=str, default="binary", choices=["binary", "regression"],
                        help="Specify the task type for the model. **Regression not yet implemented")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--weights_dir", type=str, default="./weights", help="Directory to save model weights")
    parser.add_argument("--signature", type=str, default="anemia_model", help="Model signature for saved files")

    # Display help if an incorrect argument is given
    try:
        args = parser.parse_args()
    except argparse.ArgumentError:
        parser.print_help()
        sys.exit(1)

    return args

def main():
    args = parse_arguments()
    root_dir = os.getcwd()
    data_dir = f"{root_dir}/data/cp-anemia"
    weights_dir = f"{root_dir}/weights"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    # Check for CUDA availability

    # Load datasets
    train_dataset, test_dataset = load_data(data_dir)  # Assumes load_data() returns train and test datasets
    cumulative_metrics = CumulativeMetrics(model_type=args.task_type, device=device)

    # Initialize model, loss, and optimizer
    model = load_mobilenet().to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss() if args.task_type == "binary" else torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Cross-validation setup
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    best_val_score = -float("inf") if args.task_type in ["binary", "multiclass"] else float("inf")
    early_stop_count = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        if early_stop_count >= 10:
            print("Early stopping triggered.")
            break

        for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset), 1):
            train_subset = Subset(train_dataset, train_idx)
            val_subset = Subset(train_dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)

            # Training phase
            avg_train_loss, train_metrics = train(train_loader, model, loss_fn, optimizer, cumulative_metrics, device, task_type=args.task_type)
            print(f"Training Fold {fold}: Loss: {avg_train_loss:.4f}, Metrics: {train_metrics}")

            # Validation phase
            avg_val_loss, val_metrics = eval(val_loader, model, loss_fn, "Validation", cumulative_metrics, device, task_type=args.task_type)
            print(f"Validation Fold {fold}: Loss: {avg_val_loss:.4f}, Metrics: {val_metrics}")

            # Check for improvement in validation performance and save best model
            if (args.task_type == "binary" and val_metrics["f1"] > best_val_score) or (args.task_type == "regression" and val_metrics["r2"] > best_val_score):
                best_val_score = val_metrics["f1"] if args.task_type == "binary" else val_metrics["r2"]
                torch.save(model.state_dict(), os.path.join(args.weights_dir, f"model_best_{args.task_type}_{args.signature}.pth"))
                print(f"New best model saved with score: {best_val_score:.4f}")

            # Update early stopping counter
            if fold == args.folds and avg_val_loss >= best_val_score:
                early_stop_count += 1
                print(f"Early stopping count incremented: {early_stop_count}")

            cumulative_metrics.reset()  # Reset metrics for next fold

if __name__ == "__main__":
    main()