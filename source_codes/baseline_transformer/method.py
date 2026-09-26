"""Transformer：31个频率token，两个四头自注意力块。"""
import torch
from torch import nn
from deep_common.layers import AttentionBlock, Readout, split_input


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(64, 128)
        self.position = nn.Parameter(torch.zeros(1, 31, 128))
        self.encoder = nn.Sequential(AttentionBlock(), AttentionBlock())
        self.readout = Readout()

    def forward(self, x):
        sequence, aux = split_input(x)
        return self.readout(self.encoder(self.embedding(sequence) + self.position), aux)
