from __future__ import annotations
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: list[int], output_dim: int):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)