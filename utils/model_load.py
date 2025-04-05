import torch.nn as nn
import torchvision.models as models
from timm import create_model

class MultiModel(nn.Module):
    MODEL_MAPPING = {
        "mobilenetv2": lambda: models.mobilenet_v2(pretrained=False),
        "resnet18": lambda: models.resnet18(pretrained=False),
        "densenet121": lambda: models.densenet121(pretrained=False),
        "vgg16": lambda: models.vgg16(pretrained=False),
        "vit-tiny": lambda: create_model("vit_tiny_patch16_224", pretrained=False),
        "convnext-tiny": lambda: models.convnext_tiny(pretrained=False),
        "efficientnet-b0": lambda: models.efficientnet_b0(pretrained=False),
        "shufflenetv2-0.5x": lambda: models.shufflenet_v2_x0_5(pretrained=False),
        "regnety-400mf": lambda: models.regnet_y_400mf(pretrained=False),
        "mnasnet0_5": lambda: models.mnasnet0_5(pretrained=False),
        "ghostnetv2": lambda: create_model('ghostnetv2_100.in1k', pretrained=False),
        "tinynet-a": lambda: create_model("tinynet_a.in1k", pretrained=False)
    }

    FEATURE_LAYER_MAPPING = {
        "fc": ["resnet", "shufflenet", "regnet"],
        "classifier": ["densenet", "vgg", "mobilenet", "efficientnet",
                       "mnasnet","convnext", "ghostnet", "tinynet"],
        "head": ["vit"]
    }

    def __init__(self, model_name, dropout_p=0.2):
        super().__init__()

        # self.quant = QuantStub() # Start quantization
        
        self.model_name = model_name
        self.dropout_p = dropout_p

        if self.model_name not in self.MODEL_MAPPING:
            raise ValueError(f"Model {model_name} not supported")

        self.model = self.MODEL_MAPPING[self.model_name]()
        num_ftrs = self._get_feature_size()

        # print(f"Initial Backbone {get_model_size(self.model)}")

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(128, 5)
        )

        self._assign_classifier()
        # self.dequant = DeQuantStub() # Convert back to fp32
        # print(f"Modified Backbone {get_model_size(self.model)}\n")

    def _get_feature_size(self):
        """Retrieve the number of input features for the last layer."""
        # Special case for VGG16 since its features need flattening
        if "vgg" in self.model_name:
            return 25088  # VGG16 outputs (batch, 512, 7, 7) -> flattened to 25088

        feature_layers = {
            "fc": getattr(self.model, "fc", None),
            "classifier": getattr(self.model, "classifier", None),
            "head": getattr(self.model, "head", None)
        }

        for key, layer in feature_layers.items():
            if layer:
                return layer[-1].in_features if isinstance(layer, nn.Sequential) else layer.in_features

        return getattr(self.model, "num_features", None)

    def _assign_classifier(self):
        """Assigns the appropriate classifier to the model based on its architecture."""
        if "vgg" in self.model_name:
            self.model.classifier = self.classifier
        else:
          for attr, models in self.FEATURE_LAYER_MAPPING.items():
            if any(m in self.model_name for m in models):
                setattr(self.model, attr, self.classifier)
                return

    def forward(self, x):
        output = self.model(x)
        return output[:, :4], output[:, 4]  # Class probabilities and Hb level estimate