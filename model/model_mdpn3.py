import torch
from torch import nn
import time
class Synapse(nn.Module):
    def __init__(self, classes):
        super(Synapse, self).__init__()
        self.fc = nn.Linear(128, classes)

    def forward(self, x):
        y = self.fc(x)
        return y

class Axon(nn.Module):
    def __init__(self, m, imgsize):
        super(Axon, self).__init__()
        self.fc = nn.Linear(m*3*imgsize*imgsize, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1000, 360)
        self.m = m
    def forward(self, x):
        batch_size = x.size(0)
        img_size = x.size(2)
        x = x.view(batch_size, self.m*3*img_size*img_size)
        x = self.fc(x)
        x = self.relu(x)
        return x

class Soma(nn.Module):
    def __init__(self):
        super(Soma, self).__init__()
        self.mp = nn.MaxPool3d((4, 4, 1))
    def forward(self, x):
        x = self.mp(x)
        return x


class Dendrite(nn.Module):

    def __init__(self, w, q):
        super(Dendrite, self).__init__()
        self.params = nn.ParameterDict({'w': nn.Parameter(w)})
        self.params.update({'q': nn.Parameter(q)})
    def forward(self, x):
        _, _, _, num, side, _ = self.params['w'].shape
        batch_size = x.size(0)
        img_size = x.size(2)
        x = x.unfold(2,side,1)
        x = x.unfold(3,side,1)
        x = x.view(batch_size,3,img_size-side+1,img_size-side+1,1,side,side)
        x = x.repeat((1,1,1,1,num,1,1))
        y = (torch.pi + 2*torch.atan(torch.mul(10, (torch.mul(x, self.params['w']) - self.params['q'])))) / (2 * torch.pi)
        y = torch.log(torch.prod(y+0.6, 6))
        y = torch.sum(y,5)
        return y


class mdpn3_Model(nn.Module):
    def __init__(self, m, classes, imgsize):
        super(mdpn3_Model, self).__init__()
        self.weight = torch.randn(3, 1, 1, m, 5, 5)
        self.theta = torch.randn(3, 1, 1, m, 5, 5)
        self.imgsize = int((imgsize - 4) / 4)
        self.model = nn.Sequential(
            Dendrite(self.weight, self.theta),
            Soma(),
            Axon(m, self.imgsize),
            Synapse(classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x
