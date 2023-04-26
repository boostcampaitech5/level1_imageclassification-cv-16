import torch.nn as nn
import torch.nn.functional as F

import timm
    
class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model("resnet18", pretrained=True, num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)

class ResNet34(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model("resnet34", pretrained=True, num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)
    
class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model("resnet50", pretrained=True, num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)
    
class ResNet101(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model("resnet101", pretrained=True, num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)
