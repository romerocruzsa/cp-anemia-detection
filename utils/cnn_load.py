import torch
import torch.nn as nn
from torchvision import models
from timm import create_model
class MultiModel(nn.Module):
    def __init__(self, model_name="mobilenetv2", dropout_p=0.3, num_classes=2):
        super().__init__()
        self.model_name = model_name
        self.dropout_p = dropout_p

        self.encoder_nail = self._create_encoder(model_name)
        # self.encoder_skin = self._create_encoder(model_name)

        feature_dim = self._get_feature_size(model_name)
        # fused_dim = self.feature_dim * 2

        # Quantization stubs
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        # Shared Bottleneck
        self.shared_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )

        # Classification Head
        self.classifier_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, num_classes)
        )

        # Quantile Regression Head (3 outputs)
        self.quantile_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 1),
            nn.Sigmoid()  # ensures output in [0, 1] for scaling
        )

    def _create_encoder(self, model_name):
        MODEL_MAPPING = {
            "mobilenetv2": lambda: models.mobilenet_v2(weights=True),
            "resnet18": lambda: models.resnet18(weights=True),
            "densenet121": lambda: models.densenet121(weights=True),
            "vgg16": lambda: models.vgg16(weights=True),
            "vit-tiny": lambda: create_model("vit_tiny_patch16_224", pretrained=True),
            "convnext-tiny": lambda: models.convnext_tiny(weights=True),
            "efficientnet-b0": lambda: models.efficientnet_b0(weights=True),
            "shufflenetv2-0.5x": lambda: models.shufflenet_v2_x0_5(weights=True),
            "regnety-400mf": lambda: models.regnet_y_400mf(weights=True),
            "mnasnet0_5": lambda: models.mnasnet0_5(weights=True),
            "ghostnetv2": lambda: create_model("ghostnetv2_100.in1k", pretrained=True),
            "tinynet-a": lambda: create_model("tinynet_a.in1k", pretrained=True),
        }

        model = MODEL_MAPPING[model_name]()
        if "vgg" in model_name:
            model.classifier = nn.Identity()
        elif "resnet" in model_name or "shufflenet" in model_name or "regnet" in model_name:
            model.fc = nn.Identity()
        elif "vit" in model_name:
            model.head = nn.Identity()
        elif hasattr(model, "classifier"):
            model.classifier = nn.Identity()
        return model

    def _get_feature_size(self, model_name):
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            model = self._create_encoder(model_name)
            features = model(dummy_input)
            return features.shape[1] if len(features.shape) > 1 else features.shape[0]

    def forward(self, nail_tensor):
        B, C, H, W = nail_tensor.shape

        # Reshape and quantize
        nail_flat = nail_tensor.view(B, C, H, W)
        # skin_flat = skin_tensor.view(B * N, C, H, W)

        nail_flat = self.quant(nail_flat)
        # skin_flat = self.quant(skin_flat)

        nail_feats = self.encoder_nail(nail_flat)
        # skin_feats = self.encoder_skin(skin_flat).view(B, N, -1).mean(dim=1)

        # fused = torch.cat([nail_feats, skin_feats], dim=1)
        # fused = self.dequant(fused)

        nail_feats = self.dequant(nail_feats)

        shared = self.shared_head(nail_feats)
        class_output = self.classifier_head(shared)
        quantiles = self.quantile_head(shared)  # [0, 1] range

        return class_output, quantiles

# class MultiModel(nn.Module):
#     def __init__(self, model_name="mobilenetv2", dropout_p=0.3, num_classes=4, use_transformer=True):
#         super().__init__()
#         self.model_name = model_name
#         self.dropout_p = dropout_p
#         self.use_transformer = use_transformer

#         self.encoder_nail = self._create_encoder(model_name)
#         self.encoder_skin = self._create_encoder(model_name)

#         self.feature_dim = self._get_feature_size(model_name)
#         fused_dim = self.feature_dim * 2

#         if self.use_transformer:
#             self.transformer = nn.TransformerEncoder(
#                 encoder_layer=nn.TransformerEncoderLayer(d_model=fused_dim, nhead=4, dim_feedforward=512, dropout=dropout_p, batch_first=True),
#                 num_layers=1
#             )

#         # Quantization stubs
#         self.quant = torch.quantization.QuantStub()
#         self.dequant = torch.quantization.DeQuantStub()

#         # Shared Bottleneck
#         self.shared_head = nn.Sequential(
#             nn.Linear(fused_dim, 256),
#             nn.ReLU(),
#             nn.Dropout(dropout_p),
#         )

#         # Classification Head
#         self.classifier_head = nn.Sequential(
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(dropout_p),
#             nn.Linear(128, num_classes)
#         )

#         # Quantile Regression Head
#         self.quantile_head = nn.Sequential(
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(dropout_p),
#             nn.Linear(128, 1),
#             nn.Sigmoid()
#         )

#     def _create_encoder(self, model_name):
#         model = {
#             "mobilenetv2": models.quantization.mobilenet_v2,
#             "resnet18": models.quantization.resnet18,
#         }.get(model_name, models.quantization.mobilenet_v2)(weights=None, quantize=False)

#         if hasattr(model, "classifier"):
#             model.classifier = nn.Identity()
#         if hasattr(model, "fc"):
#             model.fc = nn.Identity()
#         return model

#     def _get_feature_size(self, model_name):
#         dummy_input = torch.randn(1, 3, 224, 224)
#         with torch.no_grad():
#             model = self._create_encoder(model_name)
#             features = model(dummy_input)
#             return features.shape[1] if len(features.shape) > 1 else features.shape[0]

#     def forward(self, nail_tensor, skin_tensor):
#         B, N, C, H, W = nail_tensor.shape

#         nail_flat = nail_tensor.view(B * N, C, H, W)
#         skin_flat = skin_tensor.view(B * N, C, H, W)

#         nail_flat = self.quant(nail_flat)
#         skin_flat = self.quant(skin_flat)

#         nail_feats = self.encoder_nail(nail_flat).view(B, N, -1).mean(dim=1)
#         skin_feats = self.encoder_skin(skin_flat).view(B, N, -1).mean(dim=1)

#         fused = torch.cat([nail_feats, skin_feats], dim=1).unsqueeze(1)
#         fused = self.dequant(fused)

#         if self.use_transformer:
#             fused = self.transformer(fused).squeeze(1)
#         else:
#             fused = fused.squeeze(1)

#         shared = self.shared_head(fused)
#         class_output = self.classifier_head(shared)
#         quantiles = self.quantile_head(shared)

#         return class_output, quantiles
