"""将I/Q作为两个实数通道；与复数CNN保持相同网格、感受野和近似参数量。"""
import torch
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Conv3d(2,16,3,padding=1,bias=False),
            nn.Conv3d(16,16,3,padding=1,bias=False), nn.Conv3d(16,16,3,padding=1,bias=False)])
        # 条件层宽53使总参数17,524接近复数版本17,540，差异小于0.1%。
        self.condition = nn.Sequential(nn.Linear(3,53),nn.GELU(),nn.Linear(53,48))
        self.output = nn.Conv3d(16,2,1,bias=False)
        self.activation = nn.GELU()
        nn.init.zeros_(self.output.weight)

    def forward(self, initial, condition):
        scale = initial.abs().square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-12)
        normalized = initial/scale
        z = torch.stack([normalized.real,normalized.imag],dim=1).reshape(-1,2,8,8,31)
        bias = self.condition(condition).reshape(-1,3,16,1,1,1)
        for i,layer in enumerate(self.layers):
            z = self.activation(layer(z)+bias[:,i])
        correction = self.output(z).reshape(-1,2,64,31)
        return initial+scale*torch.complex(correction[:,0],correction[:,1])
