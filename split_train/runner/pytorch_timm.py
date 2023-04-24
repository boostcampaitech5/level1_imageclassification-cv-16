import torch
import torch.nn as nn
import torchvision.models as models
import timm

class TimmModel(nn.Module):
    def __init__(self, args, num_classes, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        
        self.model = timm.create_model(args.model, pretrained=self.pretrained)
        self.fc = nn.Sequential(nn.Dropout(p=0.2, inplace=True),
                               nn.Linear(1000, 512),
                               nn.Dropout(p=0.2, inplace=True),
                               nn.Linear(512, self.num_classes),
                               )

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x