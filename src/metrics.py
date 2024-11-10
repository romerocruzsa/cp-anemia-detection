import torch
import torchmetrics
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, r2_score

class CumulativeMetrics:
    def __init__(self, model_type="binary", num_classes=1, device="cpu"):
        self.model_type = model_type
        self.num_classes = num_classes
        self.device = device
        self.reset()

        # Classification metrics initialization
        if model_type == "binary":
            self.accuracy = torchmetrics.Accuracy(task="binary").to(device)
            self.precision = torchmetrics.Precision(task="binary").to(device)
            self.recall = torchmetrics.Recall(task="binary").to(device)
            self.f1_score = torchmetrics.F1Score(task="binary").to(device)

    def reset(self):
        self.num_batches = 0
        # Cumulative sums for each metric
        self.cum_acc = 0.0
        self.cum_prec = 0.0
        self.cum_rec = 0.0
        self.cum_f1 = 0.0
        self.cum_mse = 0.0
        self.cum_mae = 0.0
        self.cum_r2 = 0.0
        # Storage for AUC
        self.y_true_all = []
        self.y_score_all = []

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        if self.model_type == "binary":
            y_pred_class = torch.round(torch.sigmoid(y_pred)).int()
            self.y_true_all.extend(y_true.detach().cpu().numpy().tolist())
            self.y_score_all.extend(torch.sigmoid(y_pred).detach().cpu().numpy().tolist())

            # Update binary classification metrics
            self.cum_acc += self.accuracy(y_pred_class, y_true).item()
            self.cum_prec += self.precision(y_pred_class, y_true).item()
            self.cum_rec += self.recall(y_pred_class, y_true).item()
            self.cum_f1 += self.f1_score(y_pred_class, y_true).item()

        elif self.model_type == "regression":
            y_true_np = y_true.detach().cpu().numpy()
            y_pred_np = y_pred.detach().cpu().numpy()
            self.y_true_all.extend(y_true_np.tolist())
            self.y_score_all.extend(y_pred_np.tolist())

            # Compute per-batch regression metrics
            self.cum_mse += mean_squared_error(y_true_np, y_pred_np)
            self.cum_mae += mean_absolute_error(y_true_np, y_pred_np)
            self.cum_r2 += r2_score(y_true_np, y_pred_np)

        self.num_batches += 1

    def compute(self):
        if self.model_type == "binary":
            return {
                "Accuracy": self.cum_acc / self.num_batches,
                "Precision": self.cum_prec / self.num_batches,
                "Recall": self.cum_rec / self.num_batches,
                "F1 Score": self.cum_f1 / self.num_batches,
                "AUC Score": roc_auc_score(self.y_true_all, self.y_score_all)
            }

        elif self.model_type == "regression":
            return {
                "Mean Squared Error": self.cum_mse / self.num_batches,
                "Mean Absolute Error": self.cum_mae / self.num_batches,
                "R2 Score": self.cum_r2 / self.num_batches,
            }