import torch
from torch import nn
from torch.nn import functional as F


class ddNet(nn.Module):

    def __init__(self, classes, channel, img_size):
        super(ddNet, self).__init__()

        # xw+b
        self.channel = channel
        self.img_size = img_size
        self.fc0 = nn.Linear(channel*img_size*img_size, 256, bias=False)
        self.dd = nn.Linear(256, 256, bias=False)
        self.dd2 = nn.Linear(256, 256, bias=False)

        self.fc2 = nn.Linear(256, classes, bias=False)

    def forward(self, x):
        # x: [b, 1, 28, 28]
        batch_size = x.size(0)
        x = x.view(batch_size, self.channel*self.img_size*self.img_size)
        x = self.fc0(x)
        c = x
        # h1 = x@w1*x
        
        x=self.dd(x)*c
        x=self.dd2(x)*c

        x = self.fc2(x)
        return x
