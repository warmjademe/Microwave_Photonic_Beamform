"""ResCNN：四个残差卷积块；残差相加保留输入信息。"""
from torch import nn
from deep_common.layers import Readout, split_input


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv1d(128, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv1d(128, 128, 3, padding=1))

    def forward(self, x):
        return x + self.net(x)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(64, 128, 1), *[ResidualBlock() for _ in range(4)])
        self.readout = Readout()

    def forward(self, x):
        sequence, aux = split_input(x)
        return self.readout(self.encoder(sequence.transpose(1, 2)).transpose(1, 2), aux)
