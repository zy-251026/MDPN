import datetime

import torch
from torch import nn
import time
import os
import pandas
import datetime
class Synapse(nn.Module):
    def __init__(self, classes):
        super(Synapse, self).__init__()
        self.fc = nn.Linear(1000, classes)

    def forward(self, x):
        y = self.fc(x)
        return y

class Axon(nn.Module):
    def __init__(self, m, imgsize):
        super(Axon, self).__init__()
        self.fc = nn.Linear(m*imgsize*imgsize, 1000)
        self.relu = nn.ReLU()
        self.m = m
    def forward(self, x):
        batch_size = x.size(0)
        img_size = x.size(1)
        x = x.view(batch_size, self.m*img_size*img_size)
        x = self.fc(x)
        x = self.relu(x)
        return x

class Soma(nn.Module):
    def __init__(self):
        super(Soma, self).__init__()
        self.mp = nn.MaxPool3d((4, 4, 1))
    def forward(self, x):
        #path = os.getcwd()
        #t = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        #path2 = path + '/' + t + '1.csv'
        #xx = torch.clone(x[0, :, :, 0].cpu())
        #xx = pandas.DataFrame(xx.detach().numpy())
        #xx.to_csv(path2, header=False, index=False)
        #path3 = path + '/' + t + '2.csv'
        x = self.mp(x)
        #xxx = torch.clone(x[0,:,:,0].cpu())
        #xxx = pandas.DataFrame(xxx.detach().numpy())
        #xxx.to_csv(path3, header=False, index=False)
        return x


class Dendrite(nn.Module):

    def __init__(self, w, q):
        super(Dendrite, self).__init__()
        self.params = nn.ParameterDict({'w': nn.Parameter(w)})
        self.params.update({'q': nn.Parameter(q)})
    def forward(self, x):
        num, side, _ = self.params['w'].shape
        batch_size = x.size(0)
        img_size = x.size(1)
        x = x.unfold(1,side,1)
        x = x.unfold(2,side,1)
        x = x.view(batch_size,img_size-side+1,img_size-side+1,1,side,side)
        x = x.repeat((1,1,1,num,1,1))
        y = (torch.pi + 2*torch.atan(torch.mul(10, (torch.mul(x, self.params['w']) - self.params['q'])))) / (2 * torch.pi)
        y = torch.log(torch.prod(y+0.6, 5))
        y = torch.sum(y,4)
        return y


class mdpn_Model(nn.Module):
    def __init__(self, m, classes, imgsize):
        super(mdpn_Model, self).__init__()
        self.weight = torch.randn(m, 5, 5)
        self.theta = torch.randn(m, 5, 5)
        self.imgsize = int((imgsize-5+1)/4)
        self.model = nn.Sequential(
            Dendrite(self.weight, self.theta),
            Soma(),
            Axon(m, self.imgsize),
            Synapse(classes)
        )

    def forward(self, x):
        x = self.model(x)  
        # print(x.shape)
        return x
