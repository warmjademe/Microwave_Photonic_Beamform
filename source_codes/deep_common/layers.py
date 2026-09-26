"""公共网络层。输入为同一2513维观测；输出为128个归一化控制数。"""
import torch
from torch import nn

METHODS = ['dnn', 'cnn', 'rescnn', 'transformer', 'complex_cnn', 'jct']


def split_input(x):
    # 31个位置是调制子载波；16次探测×2块导频×I/Q构成64个特征通道。
    iq = x[:, :1984].reshape(-1, 16, 31, 2, 2)
    sequence = iq.permute(0, 2, 1, 3, 4).reshape(-1, 31, 64)
    return sequence, x[:, 1984:]


class Readout(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.signal = nn.Sequential(nn.Flatten(), nn.Linear(31 * width, 256), nn.GELU())
        # 529维辅助输入包含载频、探测质量、噪声方差、直流电流；所有方法相同。
        self.aux = nn.Sequential(nn.Linear(529, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU())
        self.head = nn.Sequential(nn.Linear(320, 256), nn.GELU(), nn.Linear(256, 128), nn.Sigmoid())

    def forward(self, signal, aux):
        return self.head(torch.cat([self.signal(signal), self.aux(aux)], dim=-1))


class AttentionBlock(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, 4, dropout=0., batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))

    def forward(self, x):
        z = self.norm1(x)
        x = x + self.attention(z, z, z, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


def build(method):
    import importlib
    if method not in METHODS:
        raise ValueError(method)
    return importlib.import_module('baseline_' + method + '.method').Model()
