"""Inference-only pure CNN with BatchNorm folded into the convolutions."""

import torch
from torch import nn


class DeployBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)

    def forward(self, x):
        return torch.relu(x + self.conv2(torch.relu(self.conv1(x))))


class DeployNet(nn.Module):
    def __init__(self, channels=192, blocks=12):
        super().__init__()
        self.channels = channels
        self.blocks_count = blocks
        self.stem_conv = nn.Conv2d(6, channels, 3, padding=1, bias=True)
        self.blocks = nn.Sequential(*[DeployBlock(channels) for _ in range(blocks)])
        self.policy_conv = nn.Conv2d(channels, 2, 1, bias=True)
        self.policy_fc = nn.Linear(2 * 6 * 7, 7)
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=True)
        self.value_fc1 = nn.Linear(6 * 7, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        h = torch.relu(self.stem_conv(x))
        h = self.blocks(h)
        policy = self.policy_fc(torch.relu(self.policy_conv(h)).flatten(1))
        value = torch.relu(self.value_conv(h)).flatten(1)
        value = torch.tanh(self.value_fc2(torch.relu(self.value_fc1(value)))).squeeze(1)
        return policy, value
