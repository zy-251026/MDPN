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
        self.fc = nn.Linear(m*3*imgsize*imgsize, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1000, 360)
        self.m = m
    def forward(self, x):
        batch_size = 1
        img_size = x.size(2)
        x = x.view(batch_size, self.m*3*img_size*img_size)
        x = self.fc(x)
        x = self.relu(x)
        return x

class Soma(nn.Module):
    def __init__(self):
        super(Soma, self).__init__()
    def forward(self, x):
        N = x.size(2)
        NN = int(N / 4)
        # fig = plt.figure()
        min2 = torch.clone(torch.min(x[0, :, :, :, 0]))
        max2 = torch.clone(torch.max(x[0, :, :, :, 0]))
        xx1 = torch.ceil(torch.mul(torch.div(x[0, :, :, :, 0] - min2, max2 - min2), 255).view(3, N, N))
        xxx1 = xx1[0, :, :].view(N, N)
        xxx2 = xx1[1, :, :].view(N, N)
        xxx3 = xx1[2, :, :].view(N, N)
        xx1 = torch.stack((xxx1, xxx2, xxx3), 2)
        image1 = xx1.detach().numpy()
        # print(image1)
        # plt.imshow(image1.astype(numpy.uint8), vmin=0, vmax=255)
        x = self.mp(x)
        # plt.xticks([])
        # plt.yticks([])
        # plt.axis('off')
        # plt.show()
        return x


class Dendrite(nn.Module):

    def __init__(self, w, q):
        super(Dendrite, self).__init__()
        self.params = nn.ParameterDict({'w': nn.Parameter(w)})
        self.params.update({'q': nn.Parameter(q)})
    def forward(self, x):
        _, _, _, num, side, _ = self.params['w'].shape
        batch_size = 1
        img_size = x.size(2)
        x = x.unfold(2,side,1)
        x = x.unfold(3,side,1)
        x = x.view(batch_size,3,img_size-side+1,img_size-side+1,1,side,side)
        x = x.repeat((1,1,1,1,num,1,1))
        y = (torch.pi + 2*torch.atan(torch.mul(10, (torch.mul(x, self.params['w']) - self.params['q'])))) / (2 * torch.pi)
        #y = torch.abs(torch.mul(x, self.params['w']) - self.params['q'])
        #y = torch.sum(y,6)
        y = torch.log(torch.prod(y+0.6, 6))
        y = torch.sum(y,5)

        return y


class mdpn3_Model_modify(nn.Module):
    def __init__(self, m, classes, imgsize, N):
        super(mdpn3_Model_modify, self).__init__()
        self.weight = torch.randn(3, 1, 1, m, N, N)
        self.theta = torch.randn(3, 1, 1, m, N, N)
        self.imgsize = int((imgsize - N + 1) / 4)
        self.model = nn.Sequential(
            Dendrite(self.weight, self.theta),
            Soma(N),
            Axon(m, self.imgsize),
            Synapse(classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x
