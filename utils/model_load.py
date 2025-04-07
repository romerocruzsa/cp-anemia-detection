import torch
import torch.nn as nn
from torchvision import models
from timm import create_model

class MultiModel(nn.Module):
    MODEL_MAPPING = {
        "mobilenetv2": lambda: models.mobilenet_v2(weights=None),
        "resnet18": lambda: models.resnet18(weights=None),
        "densenet121": lambda: models.densenet121(weights=None),
        "vgg16": lambda: models.vgg16(weights=None),
        "vit-tiny": lambda: create_model("vit_tiny_patch16_224", pretrained=False),
        "convnext-tiny": lambda: models.convnext_tiny(weights=None),
        "efficientnet-b0": lambda: models.efficientnet_b0(weights=None),
        "shufflenetv2-0.5x": lambda: models.shufflenet_v2_x0_5(weights=None),
        "regnety-400mf": lambda: models.regnet_y_400mf(weights=None),
        "mnasnet0_5": lambda: models.mnasnet0_5(weights=None),
        "ghostnetv2": lambda: create_model("ghostnetv2_100.in1k", pretrained=False),
        "tinynet-a": lambda: create_model("tinynet_a.in1k", pretrained=False)
    }

    FEATURE_LAYER_MAPPING = {
        "fc": ["resnet", "shufflenet", "regnet"],
        "classifier": ["densenet", "vgg", "mobilenet", "efficientnet",
                       "mnasnet","convnext", "ghostnet", "tinynet"],
        "head": ["vit"]
    }

    def __init__(self, model_name, dropout_p=0.2, num_classes=4):
        super().__init__()
        self.model_name = model_name
        self.dropout_p = dropout_p

        if model_name not in self.MODEL_MAPPING:
            raise ValueError(f"Model {model_name} not supported")

        self.backbone = self.MODEL_MAPPING[model_name]()
        num_ftrs = self._get_feature_size()

        # Replace head with identity for feature extraction
        self._strip_classifier()

        # Classification head
        self.classifier_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, num_classes),
        )

        # Regression head
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 1),
            nn.ReLU()
        )

    def _strip_classifier(self):
        if "vgg" in self.model_name:
            self.backbone.classifier = nn.Identity()
        elif "resnet" in self.model_name or "shufflenet" in self.model_name or "regnet" in self.model_name:
            self.backbone.fc = nn.Identity()
        elif "vit" in self.model_name:
            self.backbone.head = nn.Identity()
        elif hasattr(self.backbone, "classifier"):
            self.backbone.classifier = nn.Identity()

    def _get_feature_size(self):
        if "vgg" in self.model_name:
            return 25088  # VGG-specific
        if hasattr(self.backbone, "fc") and isinstance(self.backbone.fc, nn.Linear):
            return self.backbone.fc.in_features
        if hasattr(self.backbone, "classifier") and isinstance(self.backbone.classifier, nn.Sequential):
            return self.backbone.classifier[-1].in_features
        if hasattr(self.backbone, "head") and isinstance(self.backbone.head, nn.Linear):
            return self.backbone.head.in_features
        return getattr(self.backbone, "num_features", 1024)  # fallback default

    def forward(self, x):
        features = self.backbone(x)
        class_output = self.classifier_head(features)
        reg_output = self.regression_head(features).squeeze(1)
        return class_output, reg_output