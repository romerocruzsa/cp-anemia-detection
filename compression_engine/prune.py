import torch
import torch.nn.utils.prune as prune

class PruningScheduler:
    def __init__(self, model, start_epoch, end_epoch, interval, amount, mode="structured"):
        """
        Args:
            model: The model to be pruned.
            start_epoch: Epoch to start pruning.
            end_epoch: Epoch to stop pruning.
            interval: Prune every N epochs.
            amount: Amount to prune (fraction of weights or channels).
            mode: 'structured' (channel-wise) or 'unstructured' (weight-wise).
        """
        self.model = model
        self.start = start_epoch
        self.end = end_epoch
        self.interval = interval
        self.amount = amount
        self.mode = mode
        self.prunable_layers = [m for m in self.model.modules() if isinstance(m, torch.nn.Conv2d)]

    def step(self, epoch):
        if self.start <= epoch < self.end and epoch % self.interval == 0:
            print(f"[Pruning] {self.mode.capitalize()} pruning at epoch {epoch}")
            if self.mode == "structured":
                self.apply_structured_pruning()
            elif self.mode == "unstructured":
                self.apply_unstructured_pruning()
            else:
                raise ValueError(f"Unknown pruning mode: {self.mode}")
            self.report_sparsity()

    def apply_structured_pruning(self):
        for layer in self.prunable_layers:
            if hasattr(layer, 'weight'):
                prune.ln_structured(layer, name='weight', amount=self.amount, n=2, dim=0)
                prune.remove(layer, 'weight')  # Make pruning permanent

    def apply_unstructured_pruning(self):
        for layer in self.prunable_layers:
            if hasattr(layer, 'weight'):
                prune.l1_unstructured(layer, name='weight', amount=self.amount)
                prune.remove(layer, 'weight')  # Make pruning permanent

    def report_sparsity(self):
        total_weights = 0
        zero_weights = 0
        for layer in self.prunable_layers:
            if hasattr(layer, 'weight'):
                weight = layer.weight.detach().cpu().numpy()
                total_weights += weight.size
                zero_weights += (weight == 0).sum()
        
        sparsity = 100.0 * zero_weights / total_weights if total_weights > 0 else 0.0
        print(f"[Pruning] Total Sparsity across pruned layers: {sparsity:.2f}%")
