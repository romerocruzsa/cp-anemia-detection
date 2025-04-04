import torch
import torch.nn.functional as F
from torch import nn
from collections import deque
import os
import glob

class EnsembleSelfDistillation:
    def __init__(
        self,
        model_fn,
        save_dir,
        max_teachers=3,
        device='cuda',
        temperature=3.0,
        alpha=0.7,
        epsilon=0.7,
        eta_class=0.7,
        multi_teacher=True
    ):
        """
        Hybrid KD for classification + regression using top-K model ensemble.

        Args:
            model_fn: Callable that returns a new model instance
            save_dir: Where to save best checkpoint weights
            max_teachers: How many top-performing teachers to store
            device: Device for inference
            temperature: Softmax temperature
            alpha: KD vs. ground truth loss weight
            epsilon: Classification vs. regression loss weight
            eta_class: Static weighting for classification vs. regression
            multi_teacher: Whether to ensemble multiple teachers
        """
        self.model_fn = model_fn
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.max_teachers = max_teachers
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        self.epsilon = epsilon
        self.eta_class = eta_class
        self.multi_teacher = multi_teacher

        self.top_models = deque(maxlen=max_teachers)  # FIFO storage [(path, val_acc)]

    def update_top_models(self, model, val_acc, epoch):
        # Save new checkpoint
        path = os.path.join(self.save_dir, f"teacher_epoch{epoch}_acc{val_acc:.4f}.pth")
        torch.save(model.state_dict(), path)
        self.top_models.append((path, val_acc))

        # --- File cleanup to enforce max_teachers on disk ---
        all_ckpts = sorted(
            glob.glob(os.path.join(self.save_dir, "teacher_epoch*.pth")),
            key=os.path.getmtime  # sort by creation time
        )

        # If we have too many checkpoints, remove oldest
        if len(all_ckpts) > self.max_teachers:
            excess_ckpts = all_ckpts[:len(all_ckpts) - self.max_teachers]
            for ckpt_path in excess_ckpts:
                try:
                    os.remove(ckpt_path)
                    print(f"[Distiller] Removed old teacher checkpoint: {os.path.basename(ckpt_path)}")
                except Exception as e:
                    print(f"[Distiller] Warning: Failed to remove {ckpt_path} -> {e}")

    def get_teacher_models(self):
        models = []
        for path, _ in self.top_models:
            model = self.model_fn().to(self.device)
            model.load_state_dict(torch.load(path, map_location=self.device))
            model.eval()
            models.append(model)
        return models

    def get_best_teacher_path(self):
        if not self.top_models:
            return None
        return max(self.top_models, key=lambda x: x[1])[0]

    def compute_loss(self, student_class_logits, student_regression_logits, x_input, true_labels, true_levels):
        """
        Computes distillation loss from teachers.
        """
        if not self.top_models:
            return None  # No teachers yet

        # Collect teacher predictions
        if self.multi_teacher and len(self.top_models) > 1:
            teachers = self.get_teacher_models()
            class_preds, reg_preds = [], []

            for teacher in teachers:
                with torch.no_grad():
                    c, r = teacher(x_input)
                class_preds.append(c)
                reg_preds.append(r)

            t_class_logits = torch.mean(torch.stack(class_preds), dim=0)
            t_reg_preds = torch.mean(torch.stack(reg_preds), dim=0)
        else:
            teacher = self.model_fn().to(self.device)
            best_path = self.get_best_teacher_path()
            teacher.load_state_dict(torch.load(best_path, map_location=self.device))
            teacher.eval()
            with torch.no_grad():
                t_class_logits, t_reg_preds = teacher(x_input)

        # Compute hybrid KD loss
        tau = self.temperature

        # Classification KD
        teacher_probs = F.softmax(t_class_logits / tau, dim=1)
        student_log_probs = F.log_softmax(student_class_logits / tau, dim=1)
        kd_ce_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (tau ** 2)

        # Ground-truth CE + MSE
        ce_loss = F.cross_entropy(student_class_logits, true_labels)
        mse_loss = F.mse_loss(student_regression_logits.squeeze(), true_levels)

        task_loss = (self.epsilon * ce_loss) + ((1 - self.epsilon) * mse_loss)
        hybrid_loss = self.alpha * kd_ce_loss + (1 - self.alpha) * task_loss

        return hybrid_loss
