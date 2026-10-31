"""从随机初始化开始进行自我对弈与策略/价值训练。"""

import time
import numpy as np
import torch

from model import Network
from mcts import search
from support.game import new_board, play_move, next_player
from support.game import game_result, result_value, encode
from support.runtime import sample_action, sample_batch, set_seed
from support.runtime import prepare_run, save_model, append_log


def self_play(model, simulations):
    """让模型下完一盘棋，返回用于训练的 (局面, 落子分布, 胜负) 列表。"""
    board = new_board()
    player = 1
    history = []

    while True:
        visits = search(board, player, model, simulations, add_noise=True)

        # 落子：将访问次数归一化为 pi，调用 sample_action(pi) 选 action。
        # 在落子前记录 (encode(board, player), pi, player) 到 history。
        pi = visits / visits.sum()
        action = sample_action(pi)
        history.append((encode(board, player), pi, player))

        board = play_move(board, action, player)
        done, winner = game_result(board)
        if done:
            break
        player = next_player(player)

    examples = []
    # 标签：根据 winner，为 history 中每个局面生成对应视角的胜负 z。
    # 用 result_value 完成转换，将 (state, pi, z) 放入 examples。
    for state, pi, side in history:
        examples.append((state, pi, result_value(winner, side)))
    return examples


def compute_loss(model, states, target_pi, target_z):
    """策略头学习搜索分布，价值头学习对局结果。"""
    logits, values = model(states)
    # 损失：按教程计算策略损失和价值损失，返回两者之和。
    # target_pi 的形状为 [批大小, 7]；target_z 为 [批大小]。
    # 提示：torch.log_softmax、sum、mean。不要把 pi 变成单个列号。
    policy_loss = -(target_pi * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
    value_loss = ((values - target_z) ** 2).mean()
    return policy_loss + value_loss


if __name__ == "__main__":
    ITERATIONS = 12
    GAMES_PER_ITERATION = 20
    SIMULATIONS = 64
    BATCH_SIZE = 128
    TRAIN_STEPS = 60
    MEMORY_SIZE = 20000
    LEARNING_RATE = 0.001
    DEVICE = "cpu"
    RUN_DIR = "runs/format_v1"

    set_seed(2026)
    prepare_run(RUN_DIR)
    model = Network().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    save_model(model, RUN_DIR + "/initial.pt")
    replay = []
    start = time.time()

    for iteration in range(1, ITERATIONS + 1):
        model.eval()
        for game in range(GAMES_PER_ITERATION):
            replay.extend(self_play(model, SIMULATIONS))
        replay = replay[-MEMORY_SIZE:]

        model.train()
        for step in range(TRAIN_STEPS):
            states, target_pi, target_z = sample_batch(replay, BATCH_SIZE, DEVICE)
            loss = compute_loss(model, states, target_pi, target_z)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        save_model(model, RUN_DIR + "/model.pt")
        row = {"iteration": iteration, "positions": len(replay),
               "loss": loss.item(), "seconds": round(time.time() - start, 1)}
        append_log(RUN_DIR + "/log.jsonl", row)
        print(row, flush=True)
