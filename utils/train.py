import torch
import torch.nn.functional as F
from utils.model_metrics import compute_classification_metrics, compute_regression_metrics
from utils.helper import cuda_check, sw_loss

def train(dataloader, model, class_loss, reg1_loss, reg2_loss, optimizer, distiller=None, quantization="base", device="cuda"):
    """Trains the model and logs additional metrics."""
    model.to(device)
    model.train()

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

    for _, (img_id, img, multiclass, hb_level) in enumerate(dataloader):
        img = img.to(device)
        multiclass = multiclass.to(device).long()
        hb_level = hb_level.to(device).unsqueeze(1).float()

        optimizer.zero_grad()

        # Forward pass
        class_pred, reg_pred = model(img)

        # Compute losses
        ce_loss = class_loss(class_pred, multiclass)
        mse_loss = reg1_loss(reg_pred, hb_level)
        mae_loss = reg2_loss(reg_pred, hb_level)
        loss = sw_loss(ce_loss, mse_loss, 0.7)

        # Distillation loss (if self-distillation is enabled)
        if distiller is not None:
            kd_loss = distiller.compute_loss(class_pred, reg_pred, img, multiclass, hb_level) # Knowledge-distillation Loss
            ce_loss = class_loss(class_pred, multiclass)
            mse_loss = reg1_loss(reg_pred, hb_level)
            mae_loss = reg2_loss(reg_pred, hb_level)
            if kd_loss is not None:
                loss = kd_loss
            else:
                ce_loss = class_loss(class_pred, multiclass)
                mse_loss = reg1_loss(reg_pred, hb_level)
                mae_loss = reg2_loss(reg_pred, hb_level)
                loss = sw_loss(ce_loss, mse_loss, eta_class=distiller.eta_class)

        # Backpropagation
        loss.backward()
        optimizer.step()

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
        all_hb_preds.extend(reg_pred.squeeze().cpu().detach().numpy())

    # Compute metrics
    clf_metrics = compute_classification_metrics(all_targets, all_preds, all_probs)
    reg_metrics = compute_regression_metrics(all_hb_targets, all_hb_preds)

    avg_loss = total_loss / len(dataloader)
    avg_ce_loss = total_ce_loss / len(dataloader)
    avg_mse_loss = total_mse_loss / len(dataloader)
    avg_mae_loss = total_mae_loss / len(dataloader)
    accuracy = correct / total_samples

    final_metrics = [
        avg_loss,
        avg_ce_loss,
        accuracy,
        clf_metrics["precision"],
        clf_metrics["recall"],
        clf_metrics["f1"],
        clf_metrics["auc"],
        reg_metrics["r2"],
        reg_metrics["mae"],
        reg_metrics["mse"]
    ]

    return model, final_metrics