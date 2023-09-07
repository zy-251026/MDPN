import torch
from torch import nn
import time
import pandas
import datetime
import os
import numpy
import matplotlib.pyplot as plt
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
        fig = plt.figure()
        min2 = torch.clone(torch.min(x[0,:,:,:,0]))
        max2 = torch.clone(torch.max(x[0,:,:,:,0]))
        xx1 = torch.ceil(torch.mul(torch.div(x[0,:,:,:,0] - min2, max2 - min2),255).view(3,24,24))
        xxx1 = xx1[0,:,:].view(24,24)
        xxx2 = xx1[1,:,:].view(24,24)
        xxx3 = xx1[2,:,:].view(24,24)
        xx1 = torch.stack((xxx1,xxx2,xxx3),2)
        image1 = xx1.detach().numpy()
        # print(image1)
        plt.imshow(image1.astype(numpy.uint8), vmin=0, vmax=255)
        plt.show()
        x = self.mp(x)
        return x


class Dendrite(nn.Module):

    def __init__(self, w, q):
        super(Dendrite, self).__init__()
        self.params = nn.ParameterDict({'w': nn.Parameter(w)})
        self.params.update({'q': nn.Parameter(q)})
    def forward(self, x):
        _, _, _, num, side, _ = self.params['w'].shape
        # print(x.shape)
        batch_size = x.size(0)
        img_size = x.size(2)
        x = x.unfold(2,side,1)
        x = x.unfold(3,side,1)
        x = x.view(batch_size,3,img_size-side+1,img_size-side+1,1,side,side)
        x = x.repeat((1,1,1,1,num,1,1))
        w = self.params['w']
        q = self.params['q']
        w = w[:, :, :, :, 4, :].view(3,1,1,num,5,1)
        q = q[:, :, :, :, 4, :].view(3,1,1,num,5,1)
        print(w.shape, q.shape)
        y = (torch.pi + 2*torch.atan(torch.mul(10, (torch.mul(x, self.params['w']) - self.params['q'])))) / (2 * torch.pi)
        # print(yy.shape)
        # print(yy.shape)
        # y = torch.sum(y,6)
        y = torch.log(torch.prod(y+0.6, 6))
        y = torch.sum(y,5)
        return y


class mdpn3_Model(nn.Module):
    def __init__(self, m, classes, imgsize):
        super(mdpn3_Model, self).__init__()
        self.weight = torch.rand(3, 1, 1, m, 1, 1)
        self.theta = torch.rand(3, 1, 1, m, 1, 1)
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
