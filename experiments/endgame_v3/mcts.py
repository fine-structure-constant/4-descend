"""PUCT 蒙特卡洛树搜索。接口与算法说明见 教程/01-搜索与程序接口.md。"""

import math
import time
import numpy as np

from support.game import legal_moves, play_move, next_player
from support.game import game_result, result_value, convert_value
from support.runtime import Node, evaluate_position, root_noise, visit_counts
from support.tactics import position, safe_moves, possible, COLUMNS, alignment, BOARD, root_guidance, popcount

from support.endgame import SOLVER
from support.opening import lookup as opening_lookup

LAST_SEARCH = {}


def select_child(node, c_puct):
    """选择 PUCT 得分最高的子节点，返回该 Node。"""
    # 选择：遍历 node.children.values()，比较每个子节点的得分。
    # child.mean_value(node.player) 已把价值换成父节点行棋方的视角。
    # PUCT 公式见教程；平分时选择其中一个即可。
    children = [child for action, child in node.children.items()
                if action in getattr(node, "allowed", node.children)]
    wins = [child for child in children if getattr(child, "solved", None) == -1]
    if wins:
        return max(wins, key=lambda child: child.prior)
    viable = [child for child in children if getattr(child, "solved", None) != 1]
    if viable:
        children = viable
    def score(child):
        solved = getattr(child, "solved", None)
        q = -solved if solved is not None else child.mean_value(node.player)
        if solved is None and hasattr(child, "guide_value"):
            q = .15 * q + .85 * child.guide_value
        return q + c_puct * child.prior * math.sqrt(node.visits + 1) / (1 + child.visits)
    return max(children, key=score)


def expand(node, priors):
    """为合法动作建立子节点，保存到 node.children[列号]。"""
    # 扩展：调用合法动作、落子、换手函数，为每个动作创建 Node。
    # 子节点的 prior 使用 priors[列号]。play_move 不会改变原棋盘。
    own, mask = node.bits if hasattr(node, "bits") else position(node.board, node.player)
    node.bits = own, mask
    safe, status = safe_moves(own, mask)
    allowed = safe if safe else possible(mask)
    node.allowed = {col for col in range(7) if allowed & COLUMNS[col]}
    if status:
        node.solved = status
    total = sum(float(priors[col]) for col in node.allowed)
    for action in legal_moves(node.board):
        prior = float(priors[action]) / total if action in node.allowed and total > 0 else 0.
        child = Node(play_move(node.board, action, node.player),
                     next_player(node.player), prior)
        bit = possible(mask) & COLUMNS[action]
        child.bits = own ^ mask, mask | bit
        child.solved = -1 if alignment(own | bit) else 0 if (mask | bit) == BOARD else None
        node.children[action] = child


def backpropagate(path, value):
    """更新路径上每个节点的 visits 与 value_sum。"""
    # 回传：value 是 path[-1].player 的价值。
    # 用 convert_value(value, 原视角, 新视角) 换成各节点自己的视角。
    leaf_player = path[-1].player
    for node in reversed(path):
        node.visits += 1
        node.value_sum += convert_value(value, leaf_player, node.player)
        if node.children:
            outcomes = [getattr(child, "solved", None) for child in node.children.values()]
            if -1 in outcomes:
                node.solved = 1
            elif all(outcome is not None for outcome in outcomes):
                node.solved = max(-outcome for outcome in outcomes)


def search(board, player, model, simulations=32, c_puct=1.0, add_noise=False):
    """返回长度为 7 的数组，表示各列在本次搜索中被访问的次数。"""
    started_cpu = time.process_time()
    if simulations < 1:
        raise ValueError("搜索次数至少为 1")
    if game_result(board)[0]:
        raise ValueError("已经结束的对局不能继续搜索")

    # 建立根节点。训练时加入少量噪声，让自我对弈探索不同走法。
    root = Node(board, player)
    priors, value = evaluate_position(model, board, player)
    if add_noise:
        priors = root_noise(priors, board)
    expand(root, priors)
    initial_allowed = sorted(root.allowed)
    opening_bounds = None
    endgame_bounds = None
    distance_scores = None
    depth = 0
    occupied = popcount(root.bits[1])
    if not add_noise and simulations >= 32 and occupied < 8:
        own, mask = root.bits
        opening_bounds = [None] * 7
        for col, child in root.children.items():
            outcome = opening_lookup(*child.bits)
            if outcome is not None:
                root_outcome = -outcome
                opening_bounds[col] = (root_outcome, root_outcome)
                child.solved = outcome
            else:
                opening_bounds[col] = (-1, 1)
        wins = {col for col in root.allowed if opening_bounds[col] == (1, 1)}
        if wins:
            root.allowed = wins
        else:
            lower = max(opening_bounds[col][0] for col in root.allowed)
            root.allowed = {col for col in root.allowed if opening_bounds[col][1] >= lower}
    # Exact proofs become useful before visibly forced endgames. Allocate budget
    # to outcomes first, rather than spending it on mate distance in one column.
    if not add_noise and simulations >= 32 and occupied >= 12 and len(root.allowed) > 1:
        exact_budget = .38 if occupied < 20 else .58
        preferred = max(root.allowed, key=lambda col: root.children[col].prior)
        endgame_bounds = SOLVER.outcome_bounds(*root.bits, seconds=exact_budget, preferred=preferred)
        for col, bounds in enumerate(endgame_bounds):
            if bounds is not None and bounds[0] == bounds[1]:
                root.children[col].solved = -bounds[0]
        wins = {col for col in root.allowed if endgame_bounds[col] == (1, 1)}
        if wins:
            root.allowed = wins
        else:
            lower = max(endgame_bounds[col][0] for col in root.allowed)
            root.allowed = {col for col in root.allowed if endgame_bounds[col][1] >= lower}
        # In completely solved late positions use exact distance only as a
        # tiebreak: delay a forced loss or finish a proven win promptly.
        exact_complete = all(endgame_bounds[col][0] == endgame_bounds[col][1]
                             for col in initial_allowed)
        remaining = .60 - (time.process_time() - started_cpu)
        if exact_complete and remaining > .015 and occupied >= 24:
            distance_scores = SOLVER.score_moves(*root.bits, seconds=remaining, preferred=preferred)
            if all(distance_scores[col] is not None for col in root.allowed):
                best_distance = max(distance_scores[col] for col in root.allowed)
                root.allowed = {col for col in root.allowed if distance_scores[col] == best_distance}
    if not add_noise and simulations >= 32 and len(root.allowed) > 1:
        allowance = min(.50, .61 - (time.process_time() - started_cpu))
        scores, depth = root_guidance(*root.bits, seconds=allowance) if allowance > .015 else (None, 0)
        if scores is not None:
            best = max(scores[col] for col in root.allowed)
            # Only proved wins/losses can remove moves. A small difference in a
            # depth-limited heuristic is insufficient evidence to prune a column.
            if best > 10000:
                root.allowed = {col for col in root.allowed if scores[col] > 10000}
            elif best > -10000:
                root.allowed = {col for col in root.allowed if scores[col] > -10000}
            raw = np.array([scores[col] for col in sorted(root.allowed)])
            weights = np.exp(np.maximum((raw - raw.max()) / 6., -80))
            weights /= weights.sum()
            for col, weight in zip(sorted(root.allowed), weights):
                child = root.children[col]
                child.prior = .85 * float(weight) + .15 * float(priors[col])
                child.guide_value = float(np.tanh(scores[col] / 70.))

    # 模拟：重复 simulations 次模拟，每次从 root 开始。
    # 选择：沿已有子节点向下走，记录经过的 path。
    # 评估：终局用实际胜负；非终局调用网络，并扩展叶节点。
    # 回传：把叶节点视角的 value 沿 path 更新。
    for _ in range(simulations):
        # Leave margin below the platform's one-second CPU limit. Tiny interface
        # checks and training still run the requested full simulation count.
        if not add_noise and simulations >= 32 and _ >= 8:
            if time.process_time() - started_cpu >= .80:
                break
        node = root
        path = [node]
        while node.children:
            node = select_child(node, c_puct)
            path.append(node)
            if getattr(node, "solved", None) is not None:
                break
        done, winner = game_result(node.board)
        if done:
            value = result_value(winner, node.player)
        elif getattr(node, "solved", None) is not None:
            value = node.solved
        else:
            priors, value = evaluate_position(model, node.board, node.player)
            expand(node, priors)
            if popcount(node.bits[1]) <= 8:
                outcome = opening_lookup(*node.bits)
                if outcome is not None:
                    node.solved = outcome
            if getattr(node, "solved", None) is not None:
                value = node.solved
        backpropagate(path, value)

    counts = visit_counts(root)
    LAST_SEARCH.clear()
    LAST_SEARCH.update({"occupied": occupied, "initial_allowed": initial_allowed,
                        "allowed": sorted(root.allowed), "opening_bounds": opening_bounds, "outcome_bounds": endgame_bounds,
                        "distance_scores": distance_scores, "guidance_depth": depth,
                        "simulations": int(counts.sum()),
                        "cpu_seconds": round(time.process_time() - started_cpu, 6)})
    return counts
