import torch
from torch.ao.quantization import get_default_qat_qconfig, QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_qat_fx

def apply_qat_fx(model, dataloader, backend="x86"):
    """
    Applies Quantization-Aware Training (QAT) to the model using PyTorch FX.
    """
    torch.backends.quantized.engine = backend

    qconfig = get_default_qat_qconfig(backend)
    qconfig_mapping = QConfigMapping().set_global(qconfig)

    example_input = next(iter(dataloader))[1], next(iter(dataloader))[2]
    qat_model = prepare_qat_fx(model, qconfig_mapping, example_input)

    return qat_model