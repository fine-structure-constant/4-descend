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
from support.deep_opening import configure as configure_opening, safe_lookup as deep_lookup, set_deadline as opening_deadline
from support.movebook import configure as configure_movebook, lookup as movebook_lookup

LAST_SEARCH = {}
KNOWN_OUTCOMES = {}
KNOWN_FLOORS = {}


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
            weight = getattr(child, "guide_weight", .85)
            q = (1 - weight) * q + weight * child.guide_value
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
    has_deep_opening = configure_opening(model)
    has_movebook = configure_movebook(model)
    opening_deadline(started_cpu + .20)
    def book_value(bits, occupied):
        if has_movebook and occupied <= 12:
            entry = movebook_lookup(*bits)
            if entry is not None:
                return int(np.sign(entry[1]))
        value = opening_lookup(*bits) if occupied <= 8 else None
        if value is None and has_deep_opening and occupied <= 12:
            value = deep_lookup(*bits, occupied)
        return value
    if simulations < 1:
        raise ValueError("搜索次数至少为 1")
    if game_result(board)[0]:
        raise ValueError("已经结束的对局不能继续搜索")

    root_bits = position(board, player)
    # Every position with <= 12 discs has an exactly solved optimal move; use it
    # directly instead of spending the move's CPU budget on a search.
    if has_movebook and not add_noise and simulations >= 32 and popcount(root_bits[1]) <= 12:
        entry = movebook_lookup(*root_bits)
        if entry is not None:
            move_mask, score = entry
            options = [col for col in range(7)
                       if (move_mask >> col) & 1 and board[0, col] == 0]
            if options:
                move = min(options, key=lambda col: (abs(col - 3), col))
                if len(options) > 1:
                    # Every option is game-theoretically optimal; break ties
                    # towards the continuation the network likes most.
                    priors, _ = evaluate_position(model, board, player)
                    move = max(options, key=lambda col: (float(priors[col]), -abs(col - 3)))
                counts = np.zeros(7, dtype=np.float64)
                counts[move] = float(simulations)
                # An exact winning score stays winning after every legal reply;
                # carry that floor to the 13+ disc search that follows.
                own, mask = root_bits
                bit = possible(mask) & COLUMNS[move]
                after_own, after_mask = own ^ mask, mask | bit
                if score > 0 and bit and not alignment(own | bit):
                    for col in range(7):
                        reply = possible(after_mask) & COLUMNS[col]
                        if reply and not alignment(after_own | reply):
                            key = (after_own ^ after_mask) + (after_mask | reply)
                            KNOWN_FLOORS[key] = max(KNOWN_FLOORS.get(key, -1), 1)
                            KNOWN_OUTCOMES[key] = 1
                LAST_SEARCH.clear()
                LAST_SEARCH.update({"occupied": popcount(root_bits[1]), "book_move": move,
                                    "book_options": options, "book_score": score,
                                    "simulations": int(simulations),
                                    "cpu_seconds": round(time.process_time() - started_cpu, 6)})
                return counts

    # 建立根节点。训练时加入少量噪声，让自我对弈探索不同走法。
    root = Node(board, player)
    root.bits = root_bits
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
    root_minimum = KNOWN_FLOORS.get(sum(root.bits))
    root_known = KNOWN_OUTCOMES.get(sum(root.bits))
    if root_known is None and occupied <= 12:
        root_known = book_value(root.bits, occupied)
    if root_known is not None and len(root.allowed) == 1:
        root.children[next(iter(root.allowed))].solved = -root_known
    if not add_noise and simulations >= 32 and occupied < 12:
        own, mask = root.bits
        opening_bounds = [None] * 7
        for col, child in root.children.items():
            child_occupied = occupied + 1
            outcome = book_value(child.bits, child_occupied)
            if outcome is not None:
                root_outcome = -outcome
                opening_bounds[col] = (root_outcome, root_outcome)
                child.solved = outcome
                child.root_lower = root_outcome
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
        endgame_bounds = SOLVER.outcome_bounds(*root.bits, seconds=exact_budget, preferred=preferred, known=root_known, minimum=root_minimum)
        for col, bounds in enumerate(endgame_bounds):
            if bounds is not None:
                root.children[col].root_lower = bounds[0]
                if bounds[0] == bounds[1]:
                    root.children[col].solved = -bounds[0]
        wins = {col for col in root.allowed if endgame_bounds[col] == (1, 1)}
        if wins:
            root.allowed = wins
        else:
            protected = {col for col in root.allowed if endgame_bounds[col][0] >= 0}
            if protected and root_known != 1:
                root.allowed = protected
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
                guide_weight = .35 if 8 <= occupied <= 26 else .85
                child.prior = guide_weight * float(weight) + (1 - guide_weight) * float(priors[col])
                child.guide_weight = guide_weight
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
            child_occupied = popcount(node.bits[1])
            if child_occupied <= 12:
                outcome = book_value(node.bits, child_occupied)
                if outcome is not None:
                    node.solved = outcome
            if getattr(node, "solved", None) is not None:
                value = node.solved
        backpropagate(path, value)

    counts = visit_counts(root)
    # A move proved to defeat the opponent stays winning after every legal
    # opponent reply. Carry those exact outcomes to our next decision, so the
    # proof is still available after leaving the opening database frontier.
    chosen = int(np.argmax(counts))
    child = root.children[chosen]
    solved = getattr(child, "solved", None)
    floor = -solved if solved is not None else getattr(child, "root_lower", -1)
    if floor >= 0 and not game_result(child.board)[0]:
        opponent, mask = child.bits
        for col in range(7):
            bit = possible(mask) & COLUMNS[col]
            if bit and not alignment(opponent | bit):
                key = (opponent ^ mask) + (mask | bit)
                KNOWN_FLOORS[key] = max(KNOWN_FLOORS.get(key, -1), floor)
                if floor == 1:
                    KNOWN_OUTCOMES[key] = 1
    LAST_SEARCH.clear()
    LAST_SEARCH.update({"occupied": occupied, "initial_allowed": initial_allowed,
                        "allowed": sorted(root.allowed), "root_known": root_known, "root_minimum": root_minimum, "opening_bounds": opening_bounds, "outcome_bounds": endgame_bounds,
                        "distance_scores": distance_scores, "guidance_depth": depth,
                        "simulations": int(counts.sum()),
                        "cpu_seconds": round(time.process_time() - started_cpu, 6)})
    return counts
