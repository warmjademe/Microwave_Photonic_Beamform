"""8×8×31复响应残差网络；复线性卷积和模长激活保持公共相位等变。"""
import torch
from torch import nn


class ComplexConv(nn.Module):
    def __init__(self, inputs, outputs, kernel=3):
        super().__init__()
        self.real = nn.Conv3d(inputs, outputs, kernel, padding=kernel//2, bias=False)
        self.imag = nn.Conv3d(inputs, outputs, kernel, padding=kernel//2, bias=False)
    def forward(self, z):
        return torch.complex(self.real(z.real)-self.imag(z.imag),
                             self.real(z.imag)+self.imag(z.real))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([ComplexConv(1, 12), ComplexConv(12, 12), ComplexConv(12, 12)])
        self.condition = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, 36))
        self.output = ComplexConv(12, 1, 1)
        nn.init.zeros_(self.output.real.weight); nn.init.zeros_(self.output.imag.weight)

    def forward(self, initial, condition):
        scale = initial.abs().square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1e-12)
        z = (initial/scale).reshape(-1, 1, 8, 8, 31)
        bias = self.condition(condition).reshape(-1, 3, 12, 1, 1, 1)
        for i, layer in enumerate(self.layers):
            z = layer(z)
            amplitude = z.abs()
            z = z*torch.relu(amplitude+bias[:, i])/amplitude.clamp_min(1e-8)
        return initial+scale*self.output(z).reshape(-1, 64, 31)


def conditions(x, initial, pilots):
    """仅依赖公开X与已知导频；维持幅度尺度和噪声水平的可用信息。"""
    import numpy as np
    from our_method_response_control.physics import pilot_gain
    x = np.asarray(x)
    gain = pilot_gain(x, pilots)
    scale = np.sqrt(np.mean(abs(initial)**2, axis=(-2, -1)))
    ratio = np.mean(x[:, 2001:2497], axis=-1)/np.maximum(np.mean(abs(gain)**2, axis=(-2, -1)), 1e-30)
    return np.stack([(x[:, 1984]-12)/8, np.log10(np.maximum(scale, 1e-12))/5,
        np.clip(np.log10(np.maximum(ratio, 1e-15)), -10, 5)/5], axis=-1).astype(np.float32)
