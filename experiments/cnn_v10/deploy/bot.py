"""Platform adapter for the pure CNN V10 bot (CPU, single thread)."""

from pathlib import Path

import torch

from deploy.net import DeployNet
from game import from_board, initial_position, legal_columns
from mcts import search, visit_counts


def _load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class Bot:
    def __init__(self, context=None):
        torch.set_num_threads(1)
        torch.set_grad_enabled(False)
        root = Path(__file__).resolve().parent.parent
        checkpoint = _load(root / "models" / "model.pt")
        config = checkpoint.get("config", {})
        self.simulations = int(checkpoint.get("simulations", 64))
        self.model = DeployNet(int(config.get("channels", 192)),
                               int(config.get("blocks", 12)))
        state = checkpoint.get("state_dict", checkpoint)
        self.model.load_state_dict(state)
        self.model.eval()
        self.device = torch.device("cpu")
        search(self.model, initial_position(), 1, self.device)

    def act(self, observation):
        if observation.get("game", "connect4") != "connect4":
            raise ValueError("V10 only supports connect4")
        player = int(observation["player"])
        position = from_board(observation["board"], player)
        legal = sorted(int(c) for c in observation["legal_actions"])
        if legal != legal_columns(position[2]):
            raise ValueError("legal_actions disagrees with the board")
        root = search(self.model, position, self.simulations, self.device,
                      rng=None, add_noise=False)
        counts = visit_counts(root)
        visited = [column for column in legal if counts[column] > 0]
        if not visited:
            raise ValueError("search did not visit any legal column")
        return int(max(visited, key=lambda column: counts[column]))
