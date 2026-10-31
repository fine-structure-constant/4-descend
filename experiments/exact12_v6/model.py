"""Residual CNN with gravity and parity feature planes and policy/value heads."""
import torch
from torch import nn

from support.movebook import COUNT as MOVEBOOK_COUNT


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return torch.relu(x + self.conv2(torch.relu(self.conv1(x))))


class Network(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.channels=channels
        self.register_buffer('opening12', torch.zeros(6943780, dtype=torch.uint8))
        self.register_buffer('movebook_keys', torch.zeros(4 * MOVEBOOK_COUNT, dtype=torch.uint8))
        self.register_buffer('movebook_data', torch.zeros(2 * MOVEBOOK_COUNT, dtype=torch.uint8))
        self.register_buffer('deepbook', torch.zeros(33554524, dtype=torch.uint8))
        self.register_buffer('odd_rows', torch.tensor([0.,1.,0.,1.,0.,1.]).reshape(1,1,6,1))
        self.stem = nn.Conv2d(6, channels, 3, padding=1)
        self.blocks = nn.Sequential(*[Residual(channels) for _ in range(4)])
        self.policy_conv = nn.Conv2d(channels, 4, 1)
        self.policy = nn.Linear(4 * 6 * 7, 7)
        self.value_conv = nn.Conv2d(channels, 2, 1)
        self.value_hidden = nn.Linear(2 * 6 * 7, 128)
        self.value = nn.Linear(128, 1)
        # Auxiliary exact action values are trained offline; the deployed
        # forward pass need not compute this additional head.
        self.action_values = nn.Linear(channels * 6 * 7, 7)

    def features(self, state):
        occupied = state[:, :2].sum(1, keepdim=True)
        supported = torch.cat((occupied[:, :, 1:], torch.ones_like(occupied[:, :, :1])), dim=2)
        playable = state[:, 2:3] * supported
        first = (state[:, 0:1].sum((2,3), keepdim=True) == state[:, 1:2].sum((2,3), keepdim=True)).to(state.dtype)
        favoured = (self.odd_rows * first + (1-self.odd_rows) * (1-first)).expand(-1,-1,6,7)
        side = first.expand(-1,-1,6,7)
        x = torch.cat((state, playable, favoured, side), dim=1)
        x = self.blocks(torch.relu(self.stem(x)))
        return x

    def heads(self, x):
        policy = self.policy(torch.relu(self.policy_conv(x)).flatten(1))
        hidden = torch.relu(self.value_hidden(torch.relu(self.value_conv(x)).flatten(1)))
        return policy, torch.tanh(self.value(hidden)).squeeze(1)

    def forward(self, state):
        return self.heads(self.features(state))

    def training_outputs(self, state):
        x = self.features(state)
        policy, value = self.heads(x)
        return policy, value, torch.tanh(self.action_values(x.flatten(1)))
