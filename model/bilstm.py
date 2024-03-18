import torch
from torch import nn


class BiLSTM(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, dropout: float,
                 n_class: int):
        super(BiLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=1,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True,
                           dropout=dropout,
                           bidirectional=True)
        self.out = nn.Linear(in_features=2 * hidden_size, out_features=n_class)
        # weight init
        for name, param in self.rnn.named_parameters():
            if name.startswith("weight"):
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)
        nn.init.xavier_normal_(self.out.weight)
        nn.init.constant_(self.out.bias, 0)

    def forward(self, x):
        _, (h_n, _) = self.rnn(x)
        backward_out = h_n[-1, :, :]
        forward_out = h_n[-2, :, :]
        features = torch.cat((forward_out, backward_out), 1)
        x = self.out(features)

        return x


class SpatialAttention(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def __init__(self, input_c: int, patch_size: int):
        super(SpatialAttention, self).__init__()
        self.patch_size = patch_size
        self.conv1 = nn.Conv2d(input_c,
                               2 * input_c,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=(2 * input_c))
        self.pooling1 = nn.MaxPool2d(2, stride=2)
        self.P1 = nn.PReLU()
        self.conv2 = nn.Conv2d(2 * input_c, input_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=(input_c))
        self.pooling2 = nn.MaxPool2d(2, stride=2)
        self.P2 = nn.PReLU()

        self.apply(self.weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dx = self.patch_size // 2
        identity = x[:, :, dx, dx].unsqueeze(2).unsqueeze(3)
        out = self.P2(
            self.pooling2(
                self.bn2(
                    self.conv2(self.P1(self.pooling1(self.bn1(
                        self.conv1(x))))))))
        out = identity * out + identity

        return out.view(out.shape[0],1,-1)


class SpectralAttention(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def __init__(self):
        super(SpectralAttention, self).__init__()
        self.conv1 = nn.Conv1d(1, 1, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(1)
        self.ac1 = nn.PReLU()
        self.conv2 = nn.Conv1d(1, 1, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(1)
        self.ac2 = nn.PReLU()

        self.apply(self.weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.ac1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ac2(out)

        out = out * x + x

        return out.transpose(1, 2)
