import torch
import torch.nn as nn
import ResNet

class DSAN(nn.Module):

    def __init__(self, num_classes, channel, bottle_neck=True):
        super(DSAN, self).__init__()
        self.feature_layers = ResNet.resnet50(channel)
        self.bottle_neck = bottle_neck
        if bottle_neck:
            self.bottle = nn.Linear(2048, 256)
            self.cls_fc = nn.Linear(256, num_classes)
        else:
            self.cls_fc = nn.Linear(2048, num_classes)


    def forward(self, x):
        x = self.feature_layers(x)
        if self.bottle_neck:
            x = self.bottle(x)
        y = self.cls_fc(x)
        return y
