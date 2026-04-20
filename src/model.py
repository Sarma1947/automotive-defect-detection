import torch
import torch.nn as nn
from torchvision import models

class DefectDetector(nn.Module):
    def __init__(self, model_name='resnet50', num_classes=2, pretrained=True):
        super(DefectDetector, self).__init__()
        
        if model_name == 'resnet50':
            self.backbone = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, num_classes)
            )
            
        elif model_name == 'efficientnet':
            self.backbone = models.efficientnet_b4(weights='IMAGENET1K_V1' if pretrained else None)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, num_classes)
            )
            
        elif model_name == 'densenet121':
            self.backbone = models.densenet121(weights='IMAGENET1K_V1' if pretrained else None)
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, num_classes)
            )

    def forward(self, x):
        return self.backbone(x)


def get_model(model_name='resnet50', num_classes=2, pretrained=True):
    model = DefectDetector(model_name, num_classes, pretrained)
    return model


if __name__ == '__main__':
    # Quick test
    model = get_model('resnet50')
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("Model working correctly!")