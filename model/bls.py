import torch
import torchvision 
import torch.nn as nn

class blsNet(torch.nn.Module):
    def __init__(self, feature_nodes, enhancement_nodes, num_classes, channel, img_size):
        super(blsNet, self).__init__()
        linear_len = int(channel*(img_size**2))
        self.fc1 = nn.Linear(linear_len, feature_nodes)
        self.fc2 = nn.Linear(linear_len, feature_nodes)
        self.fc3 = nn.Linear(linear_len, feature_nodes)
        self.fc4 = nn.Linear(linear_len, feature_nodes)
        self.fc5 = nn.Linear(linear_len, feature_nodes)
        self.fc6 = nn.Linear(linear_len, feature_nodes)
        self.fc7 = nn.Linear(linear_len, feature_nodes)
        self.fc8 = nn.Linear(linear_len, feature_nodes)
        self.fc9 = nn.Linear(linear_len, feature_nodes)
        self.fc10 = nn.Linear(linear_len, feature_nodes)

        self.fc31 = nn.Linear(feature_nodes*10, enhancement_nodes)
        self.fc32 = nn.Linear(feature_nodes*10+enhancement_nodes, num_classes)
        # self.fc33 = nn.Linear(6140, 200)

    def forward(self, x):
        B, C, W, H = x.shape
        x = x.squeeze().view(B, -1)
        # print(x.shape)
        feature_nodes1 = torch.sigmoid(self.fc1(x))
        feature_nodes2 = torch.sigmoid(self.fc2(x))
        feature_nodes3 = torch.sigmoid(self.fc3(x))
        feature_nodes4 = torch.sigmoid(self.fc4(x))
        feature_nodes5 = torch.sigmoid(self.fc5(x))
        feature_nodes6 = torch.sigmoid(self.fc6(x))
        feature_nodes7 = torch.sigmoid(self.fc7(x))
        feature_nodes8 = torch.sigmoid(self.fc8(x))
        feature_nodes9 = torch.sigmoid(self.fc9(x))
        feature_nodes10 = torch.sigmoid(self.fc10(x))

        feature_nodes = torch.cat(
            [feature_nodes1, feature_nodes2, feature_nodes3, feature_nodes4, feature_nodes5, feature_nodes6,
             feature_nodes7, feature_nodes8, feature_nodes9, feature_nodes10], 1)
        enhancement_nodes = torch.sigmoid(self.fc31(feature_nodes))
        FeaAndEnhance = torch.cat([feature_nodes, enhancement_nodes], 1)
        outs = self.fc32(FeaAndEnhance)
        # o4 = self.fc32(o4)
        return outs
