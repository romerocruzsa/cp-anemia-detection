import torch
import torch.nn as nn
import torchvision.models as models
import os

def load_mobilenet():
    model = models.mobilenet_v2(pretrained=True)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(num_ftrs, 1)
    )
    return model

def get_model_size(model):
    torch.save(model.state_dict(), "tmp.pt")
    model_size = os.path.getsize("tmp.pt") / 1e6
    os.remove('tmp.pt')
    return f"Model Size: {model_size:.2f} MB"