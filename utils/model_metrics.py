from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, r2_score,
    mean_absolute_error, mean_squared_error
)

def compute_classification_metrics(targets, preds, probs, average="weighted"):
    precision = precision_score(targets, preds, average=average)
    recall = recall_score(targets, preds, average=average)
    f1 = f1_score(targets, preds, average=average)
    accuracy = accuracy_score(targets, preds)

    try:
        auc = roc_auc_score(targets, probs, multi_class="ovr")
    except ValueError:
        auc = float('nan')

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }

def compute_regression_metrics(y_true, y_pred):
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred)
    }