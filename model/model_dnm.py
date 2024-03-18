import torch
from torch import nn

class Soma_old(nn.Module):
    def __init__(self, k, qs):
        super(Soma_old, self).__init__()
        self.k = k
        self.qs = qs

    def forward(self, x):
        #y = 1 / (1 + torch.exp(-self.k * (x - self.qs)))
        y = x
        return y

class Membrane(nn.Module):
    def __init__(self):
        super(Membrane, self).__init__()

    def forward(self, x):
        x = torch.mean(x, 1)
        return x

class Dendritic(nn.Module):
    def __init__(self):
        super(Dendritic, self).__init__()

    def forward(self, x):
        x = torch.mean(x, 2)
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
            Membrane(),
            Soma_old(k, qs)
        )

    def forward(self, x):
        x = self.model(x)
        return x
