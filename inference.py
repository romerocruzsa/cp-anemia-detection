import os
import torch
from backend.ETL.extract import extract_data
from utils.model_load import MultiModel, load_checkpoint
from utils.eval import eval
from utils.helper import input_train_config, cuda_check
from utils.hw_metrics import benchmark_single_model 


def scan_weight_paths(weights_root: str = "~/cp-anemia-detection/output/weights"):
    weights_dir = os.path.expanduser(weights_root)
    all_weights = []
    for dirpath, _, filenames in os.walk(weights_dir):
        for fname in filenames:
            if fname.endswith(".pth"):
                all_weights.append(os.path.join(dirpath, fname))
    return all_weights

def parse_metadata_from_path(path: str):
    # Example filename: mobilenetv2_qat_pruning_ce.pth
    base = os.path.basename(path).replace(".pth", "")
    parts = base.split("_")
    architecture = parts[0]
    config = "_".join(parts[1:])  # useful for tracking or later mapping
    return architecture, config

def infer_and_benchmark(weights_dir=None):
    print("[Infer] Scanning weight files...")
    model_paths = scan_weight_paths(weights_dir)
    print(f"[Infer] Found {len(model_paths)} models to evaluate\n")

    device = cuda_check()
    dataset_dir = os.path.expanduser("~/cp-anemia-detection/data/cp-anemia")
    _, dataloaders = extract_data(dataset_dir, batch_size=16)
    _, test_loader = dataloaders

    for model_path in model_paths:
        architecture, signature = parse_metadata_from_path(model_path)

        print(f"\n{'='*80}")
        print(f"[Model] Architecture: {architecture} | Signature: {signature}")
        print(f"[Path] {model_path}")

        model = MultiModel(architecture)
        load_checkpoint(model, model_path)
        model.to(device)

        # Evaluation metrics
        metrics, _ = eval(test_loader, model, None, None, None, quantization="none", device=device)
        print("\n[Eval] Classification Accuracy: {:.2f}%".format(metrics[2]))

        # Benchmarking (inference speed, memory, power)
        print("\n[Benchmarking] Hardware Performance Metrics")
        benchmark_single_model(model, test_loader, device)
        print("="*80)

if __name__ == "__main__":
    infer_and_benchmark()
