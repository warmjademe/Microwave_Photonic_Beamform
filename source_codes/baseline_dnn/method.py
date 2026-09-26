"""DNN：展开所有导频，经三层全连接提取特征，再输出共同控制。"""
import torch
from torch import nn
from deep_common.layers import Readout, split_input


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(1984, 512), nn.GELU(),
            nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 256), nn.GELU())
        self.readout = Readout()
        self.readout.signal = nn.Identity()

    def forward(self, x):
        _, aux = split_input(x)
        return self.readout(self.encoder(x[:, :1984]), aux)
