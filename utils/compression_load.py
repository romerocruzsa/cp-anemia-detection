import os
from compression_engine.prune import PruningScheduler
from compression_engine.qat import apply_qat_fx
from compression_engine.kd import EnsembleSelfDistillation
from utils.helper import clear_folder
from utils.model_load import MultiModel

def compression_config(model, architecture, quantization_mode, pruning_mode, distillation_mode, epochs, dataloader, device="cuda"):
    if quantization_mode == "qat":
                model = apply_qat_fx(model, dataloader)

    if pruning_mode != "base":
        prune_scheduler = PruningScheduler(
            model=model,
            start_epoch=2,
            end_epoch=int(epochs * 0.8),
            interval=2,
            amount=0.2,
            mode=pruning_mode)
    else:
          prune_scheduler = None
        
    if distillation_mode == "self-distil":
        clear_folder(os.path.expanduser("~/cp-anemia-detection/compression_engine/kd_checkpoints"))
        distiller = EnsembleSelfDistillation(
            model_fn=lambda: MultiModel(architecture),
            save_dir='compression_engine/kd_checkpoints/',
            max_teachers=3,
            device=device
        )
    else:
         distiller = None
            
    return model, prune_scheduler, distiller