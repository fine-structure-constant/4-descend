"""Spatial residual policy/value network for the 6x7 board."""
import torch
from torch import nn


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return torch.relu(x + self.conv2(torch.relu(self.conv1(x))))


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 32, 3, padding=1)
        self.blocks = nn.Sequential(*[Residual(32) for _ in range(3)])
        self.policy_conv = nn.Conv2d(32, 4, 1)
        self.policy = nn.Linear(4 * 6 * 7, 7)
        self.value_conv = nn.Conv2d(32, 2, 1)
        self.value_hidden = nn.Linear(2 * 6 * 7, 64)
        self.value = nn.Linear(64, 1)

    def forward(self, x):
        x = self.blocks(torch.relu(self.stem(x)))
        policy = self.policy(torch.relu(self.policy_conv(x)).flatten(1))
        hidden = torch.relu(self.value_hidden(torch.relu(self.value_conv(x)).flatten(1)))
        return policy, torch.tanh(self.value(hidden)).squeeze(1)
