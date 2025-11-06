import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SupConMobileNet(nn.Module):
    def __init__(self, num_classes=100, pretrained=True):
        super().__init__()
        backbone = models.mobilenet_v2(pretrained=pretrained)
        # bỏ classifier (fc)
        self.feature_extractor = backbone.features  # (B, 1280, 7, 7)
        feature_dim = backbone.last_channel  # 1280 cho mobilenet_v2

        # projection head cho contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 128)
        )

        # classification head
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x, return_features=False):
        feats = self.feature_extractor(x)
        feats = F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)  # (B, 1280)
        proj = F.normalize(self.projection_head(feats), dim=1)
        logits = self.classifier(feats)
        if return_features:
            return logits, proj
        return logits