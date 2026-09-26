"""JCT适配：CNN与Transformer并行，GAP+MLP+softmax学习两支权重。

对应Wang等TWC 2025的联合模块思想及式(14)。使用固定两块结构；
不实现原文多级退出、Ghost压缩、对比预训练或波束类别分类。
"""
import torch
from torch import nn
from deep_common.layers import AttentionBlock, Readout, split_input


class JCTBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv1d(128, 128, 3, padding=1), nn.GELU(),
                                 nn.Conv1d(128, 128, 3, padding=1))
        self.transformer = AttentionBlock()
        self.gate = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2), nn.Softmax(dim=-1))

    def forward(self, x):
        weight = self.gate(x.mean(dim=1))
        oc = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        ot = self.transformer(x)
        return weight[:, 0, None, None] * oc + weight[:, 1, None, None] * ot


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(64, 128)
        self.position = nn.Parameter(torch.zeros(1, 31, 128))
        self.encoder = nn.Sequential(JCTBlock(), JCTBlock())
        self.readout = Readout()

    def forward(self, x):
        sequence, aux = split_input(x)
        return self.readout(self.encoder(self.embedding(sequence) + self.position), aux)
