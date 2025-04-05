import torch

class EarlyStopping:
    def __init__(self, patience=10, delta=0.001, mode="max", f1_gap_threshold=.25):
        """
        Args:
            patience (int): How long to wait after last improvement.
            delta (float): Minimum change to qualify as improvement.
            mode (str): "max" for accuracy/F1, "min" for loss.
            f1_gap_threshold (float): Optional max allowed gap between training and validation F1.
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.mode = mode
        self.f1_gap_threshold = f1_gap_threshold

    def __call__(self, val_score, train_f1=None, val_f1=None, model=None):
        # Check if F1 gap triggers early stopping
        if self.f1_gap_threshold is not None and train_f1 is not None and val_f1 is not None:
            gap = abs(train_f1 - val_f1)
            if gap > self.f1_gap_threshold:
                print(f"[EarlyStopping] Triggered due to F1 gap: {gap:.4f} > {self.f1_gap_threshold}")
                self.early_stop = True
                return

        # Normal early stopping logic
        if self.best_score is None:
            self.best_score = val_score
        elif (self.mode == "max" and val_score < self.best_score + self.delta) or \
             (self.mode == "min" and val_score > self.best_score - self.delta):
            self.counter += 1
            if self.counter >= self.patience:
                print(f"[EarlyStopping] No improvement for {self.patience} steps.")
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
