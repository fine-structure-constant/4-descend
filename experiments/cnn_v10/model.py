"""Training-time pure residual CNN for V10.

No move books, solver payloads, or auxiliary lookup buffers are part of this
model: the policy and value heads are the entire learned evaluator.
"""

import torch
from torch import nn

from deploy.net import DeployNet


class Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return torch.relu(x + y)


class Net(nn.Module):
    def __init__(self, channels=192, blocks=12):
        super().__init__()
        self.channels = channels
        self.blocks_count = blocks
        self.stem_conv = nn.Conv2d(6, channels, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.blocks = nn.Sequential(*[Block(channels) for _ in range(blocks)])
        self.policy_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 6 * 7, 7)
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(6 * 7, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def features(self, x):
        h = torch.relu(self.stem_bn(self.stem_conv(x)))
        return self.blocks(h)

    def heads(self, h):
        policy = self.policy_fc(torch.relu(self.policy_bn(self.policy_conv(h))).flatten(1))
        value = torch.relu(self.value_bn(self.value_conv(h))).flatten(1)
        value = torch.tanh(self.value_fc2(torch.relu(self.value_fc1(value)))).squeeze(1)
        return policy, value

    def forward(self, x):
        return self.heads(self.features(x))


class _Flat(nn.Module):
    def forward(self, x):
        return x.flatten(1)


def _fold_conv_bn(conv, bn):
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    weight = conv.weight * scale.reshape(-1, 1, 1, 1)
    bias = bn.bias - bn.running_mean * scale
    return weight.detach(), bias.detach()


@torch.no_grad()
def to_deploy(model):
    """Fold every BatchNorm into its convolution for 1-thread CPU play."""
    model.eval()
    deploy = DeployNet(model.channels, model.blocks_count)
    w, b = _fold_conv_bn(model.stem_conv, model.stem_bn)
    deploy.stem_conv.weight.copy_(w)
    deploy.stem_conv.bias.copy_(b)
    for src, dst in zip(model.blocks, deploy.blocks):
        w, b = _fold_conv_bn(src.conv1, src.bn1)
        dst.conv1.weight.copy_(w)
        dst.conv1.bias.copy_(b)
        w, b = _fold_conv_bn(src.conv2, src.bn2)
        dst.conv2.weight.copy_(w)
        dst.conv2.bias.copy_(b)
    w, b = _fold_conv_bn(model.policy_conv, model.policy_bn)
    deploy.policy_conv.weight.copy_(w)
    deploy.policy_conv.bias.copy_(b)
    w, b = _fold_conv_bn(model.value_conv, model.value_bn)
    deploy.value_conv.weight.copy_(w)
    deploy.value_conv.bias.copy_(b)
    deploy.policy_fc.weight.copy_(model.policy_fc.weight)
    deploy.policy_fc.bias.copy_(model.policy_fc.bias)
    deploy.value_fc1.weight.copy_(model.value_fc1.weight)
    deploy.value_fc1.bias.copy_(model.value_fc1.bias)
    deploy.value_fc2.weight.copy_(model.value_fc2.weight)
    deploy.value_fc2.bias.copy_(model.value_fc2.bias)
    deploy.eval()
    return deploy


def load_checkpoint(path, map_location="cpu"):
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)
    config = checkpoint.get("config", {})
    model = Net(int(config.get("channels", 192)), int(config.get("blocks", 12)))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, config
