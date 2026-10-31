"""Residual MLP policy/value network; trained from random initialization."""
import torch
from torch import nn


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3 * 6 * 7, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.policy = nn.Linear(256, 7)
        self.value_hidden = nn.Linear(256, 64)
        self.value = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x.reshape(-1, 3 * 6 * 7)))
        x = torch.relu(x + self.fc3(torch.relu(self.fc2(x))))
        logits = self.policy(x)
        value = torch.tanh(self.value(torch.relu(self.value_hidden(x)))).squeeze(1)
        return logits, value
