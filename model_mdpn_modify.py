import torch
from torch import nn
import time
import matplotlib.pyplot as plt
import numpy
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
        self.fc = nn.Linear(m*imgsize*imgsize, 128)
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
        N = x.size(1)
        # N = 28 - 1 + 1
        NN = int(N / 4)
        # fig = plt.figure()
        min2 = torch.clone(torch.min(x[0, :, :, 0]))
        max2 = torch.clone(torch.max(x[0, :, :, 0]))
        xx1 = torch.ceil(torch.mul(torch.div(x[0, :, :, 0] - min2, max2 - min2), 255).view(N, N))
        # xxx1 = xx1[0, :, :].view(N, N)
        # xxx2 = xx1[1, :, :].view(N, N)
        # xxx3 = xx1[2, :, :].view(N, N)
        xx1 = torch.stack((xx1, xx1, xx1), 2)
        image1 = xx1.detach().numpy()
        # print(image1)
        # plt.imshow(image1.astype(numpy.uint8), vmin=0, vmax=255)
        # plt.xticks([])
        # plt.yticks([])
        # plt.axis('off')
        x = self.mp(x)
        # plt.show()
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
        #y = torch.abs(torch.mul(x, self.params['w']) - self.params['q'])
        y = torch.log(torch.prod(y+0.6, 5))
        #y = torch.sum(y,5)
        y = torch.sum(y,4)
        return y


class mdpn_Model_modify(nn.Module):
    def __init__(self, m, classes, imgsize, N):
        super(mdpn_Model_modify, self).__init__()
        self.weight = torch.randn(m, N, N)
        self.theta = torch.randn(m, N, N)
        self.imgsize = int((imgsize-N+1)/4)
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
