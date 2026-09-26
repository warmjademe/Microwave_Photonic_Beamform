"""CNN：沿31个子载波提取局部频率变化，不把合路观测冒充64路天线。"""
from torch import nn
from deep_common.layers import Readout, split_input


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(64, 128, 3, padding=1), nn.GELU(),
            nn.Conv1d(128, 128, 3, padding=1), nn.GELU(),
            nn.Conv1d(128, 128, 3, padding=1), nn.GELU())
        self.readout = Readout()

    def forward(self, x):
        sequence, aux = split_input(x)
        return self.readout(self.encoder(sequence.transpose(1, 2)).transpose(1, 2), aux)
