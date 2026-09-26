"""真正复数卷积：复权重A+jB与I+jQ相乘，而非普通双通道CNN。"""
import torch
from torch import nn
from torch.nn import functional as F
from deep_common.layers import Readout, split_input


class ComplexConv(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        # 不使用四个独立实权重：同一A、B在实部和虚部间共享并带正确符号。
        self.a = nn.Conv1d(incoming, outgoing, 3, padding=1, bias=False)
        self.b = nn.Conv1d(incoming, outgoing, 3, padding=1, bias=False)
        self.bias = nn.Parameter(torch.zeros(2, outgoing, 1))

    def forward(self, z):
        real, imag = z
        return (self.a(real) - self.b(imag) + self.bias[0],
                self.b(real) + self.a(imag) + self.bias[1])


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([ComplexConv(32, 64), ComplexConv(64, 64), ComplexConv(64, 64)])
        self.readout = Readout()

    def forward(self, x):
        sequence, aux = split_input(x)
        paired = sequence.reshape(-1, 31, 32, 2)
        z = (paired[..., 0].transpose(1, 2), paired[..., 1].transpose(1, 2))
        for layer in self.convs:
            r, i = layer(z)
            # split-GELU是明确的复数网络非线性选择；不声称满足解析函数条件。
            z = (F.gelu(r), F.gelu(i))
        features = torch.cat(z, dim=1).transpose(1, 2)
        return self.readout(features, aux)
