import torch
from torch import nn

class Membrane(nn.Module):
    def __init__(self):
        super(Membrane, self).__init__()

    def forward(self, x):
        x = torch.sum(x, 1)
        return x

class Dendritic(nn.Module):
    def __init__(self):
        super(Dendritic, self).__init__()

    def forward(self, x):
        x = torch.sum(x, 2)
        return x


class Synapse_old(nn.Module):

    def __init__(self, w, q, k):
        super(Synapse_old, self).__init__()
        self.params = nn.ParameterDict({'w': nn.Parameter(w)})
        self.params.update({'q': nn.Parameter(q)})
        self.k = k

    def forward(self, x):
        num, _ = self.params['w'].shape
        x = torch.unsqueeze(x, 1)
        x = x.repeat((1, num, 1))
        y = 1 / (1 + torch.exp(
            torch.mul(-self.k, (torch.mul(x, self.params['w']) - self.params['q']))))
        return y


class DNM_Model(nn.Module):
    def __init__(self, w, q, k, qs):
        super(DNM_Model, self).__init__()
        self.model = nn.Sequential(
            Synapse_old(w, q, k),
            Dendritic(),
            Membrane()
        )

    def forward(self, x):
        x = self.model(x)
        return x
        
        
class layer1(nn.Module):
    def __init__(self, classes, channel, img_size):
        super(layer1, self).__init__()
        self.input = int(channel*(img_size**2))
        self.output = classes
        self.fcs = []
        for i in range(self.output):
            weight1 = torch.randn(30, self.input)
            theta1 = torch.randn(30, self.input)
            fc = DNM_Model(weight1, theta1, 10, 0.5)
            setattr(self, 'fc%i'%i, fc)
            self.fcs.append(fc)
    def forward(self, x):
        xx = {}
        for i in range(self.output):
            xx[str(i)] = self.fcs[i](x).view(-1,1)
            if i==0:
                y = xx[str(i)]
            else:
                y = torch.cat((xx[str(i)], y), 1)
        return y

class MDNM_Model(nn.Module):
    def __init__(self, classes, channel, img_size):
        super(MDNM_Model, self).__init__()
        self.dnm1 = layer1(classes, channel, img_size)
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        out = self.dnm1(x)
        return out
