import torch
from torch.ao.quantization import QConfig, QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization.observer import MinMaxObserver
# import modelopt.torch.quantization as mtq
from utils.helper import calibrate_loop

def apply_static_ptq(model, dataloader, num_calib_batches=10):
    """
    Applies static post-training quantization using FX graph mode.
    """
    qconfig = QConfig(
        activation=MinMaxObserver.with_args(
            quant_min=0, quant_max=255, dtype=torch.quint8, qscheme=torch.per_tensor_affine
        ),
        weight=MinMaxObserver.with_args(
            quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
        )
    )
    qconfig_mapping = QConfigMapping().set_global(qconfig)

    example_input = next(iter(dataloader))[0][:1].to(next(model.parameters()).device)
    prepared_model = prepare_fx(model, qconfig_mapping, example_input)

    with torch.no_grad():
        for i, (images, _, _) in enumerate(dataloader):
            prepared_model(images.to(next(model.parameters()).device))
            if i >= num_calib_batches:
                break

    int8_model = convert_fx(prepared_model)
    return int8_model

def apply_fp16(model):
    """
    Converts model weights to FP16 for inference.
    """
    fp16_model = model.half()
    return fp16_model

# INT4 can be added here if you're using an external library like MIT-HAN Lab's MTQ
# def apply_int4_awq(model, dataloader):
#     """
#     Converts model weights to INT4 using AWQ config for inference.
#     """
#     quant_cfg = mtq.INT4_AWQ_REAL_QUANT_CFG
#     int4_model = mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop(dataloader))
#     return int4_model
