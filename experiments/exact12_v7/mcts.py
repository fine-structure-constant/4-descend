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
from support import pons_book
from support.constrained import SOLVER as CLOSED_SOLVER, applicable as closed_applicable
from support.corrections import lookup as correction_lookup
from support.midbook import lookup as midbook_lookup, configure as configure_midbook
from support import native

LAST_SEARCH = {}
KNOWN_OUTCOMES = {}
KNOWN_FLOORS = {}
GUIDE_WEIGHT = .10
CPU_LIMIT = .80
WALL_LIMIT = .92


def remember_floor(bits, move, floor):
    own,mask=bits;bit=possible(mask)&COLUMNS[move]
    if floor<0 or alignment(own|bit):return
    opponent,after=own^mask,mask|bit
    for col in range(7):
        reply=possible(after)&COLUMNS[col]
        if reply and not alignment(opponent|reply):
            key=(opponent^after,after|reply)
            KNOWN_FLOORS[key]=max(KNOWN_FLOORS.get(key,-1),floor)
            if floor==1:KNOWN_OUTCOMES[key]=1


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
    started_wall = time.monotonic()
    native_cpu = 0.
    def cpu_used():
        return time.process_time() - started_cpu + native_cpu
    has_deep_opening = configure_opening(model)
    has_movebook = configure_movebook(model)
    pons_book.configure(model)
    configure_midbook(model)
    opening_deadline(started_cpu + .20)
    def book_value(bits, occupied):
        if occupied <= 14:
            exact = pons_book.score(*bits)
            if exact is not None:
                return (exact > 0) - (exact < 0)
        if has_movebook and occupied <= 12:
            entry = movebook_lookup(*bits)
            if entry is not None:
                return int(np.sign(entry[1]))
        value = opening_lookup(*bits) if occupied <= 8 else None
        if value is None and has_deep_opening and occupied <= 12:
            value = deep_lookup(*bits, occupied)
        if value is None and occupied <= 21:
            mid = midbook_lookup(*bits)
            if mid is not None:
                resolved = [v for v in mid if v is not None]
                if resolved:
                    value = max(resolved)
        return value
    if simulations < 1:
        raise ValueError("搜索次数至少为 1")
    if game_result(board)[0]:
        raise ValueError("已经结束的对局不能继续搜索")

    root_bits = position(board, player)
    # Every position with <= 12 discs has an exactly solved optimal move; use it
    # directly instead of spending the move's CPU budget on a search.
    if has_movebook and simulations >= 32 and popcount(root_bits[1]) <= 12:
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
                    if add_noise:
                        weights = np.asarray([max(float(priors[col]), .02) for col in options])
                        weights = .75 * weights / weights.sum() + .25 / len(options)
                        move = int(np.random.choice(options, p=weights))
                counts = np.zeros(7, dtype=np.float64)
                counts[move] = float(simulations)
                # An exact winning score stays winning after every legal reply;
                # carry that floor to the 13+ disc search that follows.
                own, mask = root_bits
                bit = possible(mask) & COLUMNS[move]
                after_own, after_mask = own ^ mask, mask | bit
                if score >= 0 and bit and not alignment(own | bit):
                    for col in range(7):
                        reply = possible(after_mask) & COLUMNS[col]
                        if reply and not alignment(after_own | reply):
                            key = (after_own ^ after_mask, after_mask | reply)
                            floor = int(score > 0)
                            KNOWN_FLOORS[key] = max(KNOWN_FLOORS.get(key, -1), floor)
                            if floor: KNOWN_OUTCOMES[key] = 1
                LAST_SEARCH.clear()
                LAST_SEARCH.update({"occupied": popcount(root_bits[1]), "book_move": move,
                                    "book_options": options, "book_score": score,
                                    "simulations": int(simulations),
                                    "cpu_seconds": round(time.process_time() - started_cpu, 6)})
                return counts

    # Retain proved corrections from previous real games. These are exact
    # non-losing actions, not preferences inferred from the network.
    correction=correction_lookup(*root_bits) if simulations>=32 else None
    if correction is not None:
        mask,score=correction
        options=[c for c in range(7) if mask&(1<<c) and board[0,c]==0]
        if not options:raise ValueError('Invalid exact correction entry')
        move=min(options,key=lambda c:(abs(c-3),c))
        if len(options)>1:
            policy,_=evaluate_position(model,board,player)
            weights=np.array([max(float(policy[c]),.001) for c in options])
            move=int(np.random.choice(options,p=weights/weights.sum())) if add_noise else max(options,key=lambda c:float(policy[c]))
        counts=np.zeros(7,dtype=np.float64);counts[move]=float(simulations)
        remember_floor(root_bits,move,score)
        LAST_SEARCH.clear();LAST_SEARCH.update({'phase':'refutation_book','occupied':popcount(root_bits[1]),
            'action':move,'root_lower':score,'optimal_options':options,'simulations':simulations,
            'cpu_seconds':time.process_time()-started_cpu})
        return counts

    # Keep all top-cell spare turns. These narrow endgames are small enough
    # for a full-game proof; no heuristic bonus for a merely floating three.
    closed_bounds=None
    if simulations>=32 and closed_applicable(root_bits[1]):
        known=KNOWN_OUTCOMES.get(root_bits)
        minimum=KNOWN_FLOORS.get(root_bits)
        seconds=max(.01,min(.70-cpu_used(),WALL_LIMIT-(time.monotonic()-started_wall)-.12))
        closed_bounds=CLOSED_SOLVER.analyze(*root_bits,seconds=seconds,known=known,minimum=minimum)
        wins=[c for c,b in enumerate(closed_bounds) if b==(1,1)]
        complete=all(b is None or b[0]==b[1] for b in closed_bounds)
        nonlosing=[c for c,b in enumerate(closed_bounds) if b is not None and b[0]>=0]
        options=wins
        if not options and complete:
            best=max(b[0] for b in closed_bounds if b is not None)
            options=[c for c,b in enumerate(closed_bounds) if b is not None and b[0]==best]
        if not options and nonlosing and known!=1 and minimum!=1:options=nonlosing
        if options:
            move=min(options,key=lambda c:(abs(c-3),c))
            if len(options)>1:
                policy,_=evaluate_position(model,board,player)
                weights=np.array([max(float(policy[c]),.001) for c in options])
                move=int(np.random.choice(options,p=weights/weights.sum())) if add_noise else max(options,key=lambda c:float(policy[c]))
            counts=np.zeros(7,dtype=np.float64);counts[move]=float(simulations)
            remember_floor(root_bits,move,closed_bounds[move][0])
            LAST_SEARCH.clear();LAST_SEARCH.update(CLOSED_SOLVER.last)
            LAST_SEARCH.update({'phase':'constrained','occupied':popcount(root_bits[1]),
                                'action':move,'optimal_options':options,'root_lower':closed_bounds[move][0],
                                'simulations':simulations,'cpu_seconds':time.process_time()-started_cpu})
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
    root_minimum = KNOWN_FLOORS.get(root.bits)
    root_known = KNOWN_OUTCOMES.get(root.bits)
    if root_known is None and occupied <= 14:
        root_known = book_value(root.bits, occupied)
    if root_known is not None and len(root.allowed) == 1:
        root.children[next(iter(root.allowed))].solved = -root_known
    if simulations >= 32 and occupied <= 13:
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
    mid_bounds = None
    if simulations >= 32 and 13 < occupied <= 21 and len(root.allowed) > 1:
        mid = midbook_lookup(*root.bits)
        if mid is not None:
            mid_bounds = [None] * 7
            for col in root.allowed:
                outcome = mid[col]
                if outcome is None:
                    continue
                mid_bounds[col] = (outcome, outcome)
                root.children[col].solved = -outcome
                root.children[col].root_lower = outcome
            resolved = [mid_bounds[col][0] for col in root.allowed if mid_bounds[col] is not None]
            if resolved:
                if root_known is None:
                    root_known = max(resolved)
                if root_minimum is None:
                    root_minimum = max(resolved)
    # Deep positions are settled by the native exact searcher when possible.
    # Early plies stay on the stored books, and any failure falls back to the
    # pure-Python pipeline below without raising.
    native_used = False
    native_bounds = None
    if (simulations >= 32 and not add_noise and occupied >= 15 and len(root.allowed) > 1
            and not (mid_bounds is not None
                     and all(mid_bounds[col] is not None for col in initial_allowed))):
        wall_used = time.monotonic() - started_wall
        budget_ms = int(min(.45, max(.06, min(
            WALL_LIMIT - wall_used - .28, CPU_LIMIT - cpu_used() - .20))) * 1000)
        if budget_ms >= 60 and native.available():
            call_started = time.monotonic()
            result = native.analyze(*root.bits, budget_ms)
            native_cpu += time.monotonic() - call_started
            if result is not None:
                col, score, solved, wdl = result
                if solved:
                    options = [c for c in root.allowed if wdl[c] == score]
                    if options:
                        move = options[0] if len(options) == 1 else max(
                            options, key=lambda c: (float(priors[c]), -abs(c - 3)))
                        if score >= 0:
                            remember_floor(root.bits, move, score)
                        counts = np.zeros(7, dtype=np.float64)
                        counts[move] = float(simulations)
                        LAST_SEARCH.clear()
                        LAST_SEARCH.update({
                            "phase": "native_exact", "occupied": occupied,
                            "action": move, "optimal_options": options,
                            "root_value": score, "native_budget_ms": budget_ms,
                            "simulations": int(simulations),
                            "cpu_seconds": round(cpu_used(), 6),
                            "wall_seconds": round(time.monotonic() - started_wall, 6)})
                        return counts
                else:
                    if score >= 0:
                        root.allowed = {c for c in root.allowed
                                        if wdl[c] == 9 or wdl[c] >= score}
                    native_bounds = [None] * 7
                    for c in range(7):
                        if wdl[c] != 9:
                            native_bounds[c] = (wdl[c], wdl[c])
                            root.children[c].solved = -wdl[c]
                            root.children[c].root_lower = wdl[c]
                    native_used = True
    if simulations >= 32 and occupied >= 12 and len(root.allowed) > 1 and not native_used:
        exact_budget = min(.60 if occupied < 20 else .70, max(0., .72 - cpu_used()),
                           max(0., WALL_LIMIT - (time.monotonic() - started_wall) - .12))
        preferred = max(root.allowed, key=lambda col: root.children[col].prior)
        if mid_bounds is None and closed_bounds is None:
            # Entries left by earlier midgame solves can dominate the budget with
            # shallow bounds; cold tables resolve far more within the same time.
            if occupied >= 20:
                SOLVER.table.clear()
            endgame_bounds = SOLVER.outcome_bounds(*root.bits, seconds=exact_budget, preferred=preferred, known=root_known, minimum=root_minimum)
        else:
            endgame_bounds = mid_bounds if mid_bounds is not None else closed_bounds
        for col, bounds in enumerate(endgame_bounds):
            if bounds is not None:
                root.children[col].root_lower = bounds[0]
                if bounds[0] == bounds[1]:
                    root.children[col].solved = -bounds[0]
        wins = {col for col in root.allowed if endgame_bounds[col] == (1, 1)}
        if wins:
            root.allowed = wins
        elif root_known == 1:
            # A known win must lie outside the refuted draws (-1, 0).
            unresolved = {col for col in root.allowed
                          if endgame_bounds[col] not in ((-1, 0), (-1, -1))}
            if unresolved:
                root.allowed = unresolved
            else:
                lower = max(endgame_bounds[col][0] for col in root.allowed)
                root.allowed = {col for col in root.allowed if endgame_bounds[col][1] >= lower}
        else:
            protected = {col for col in root.allowed if endgame_bounds[col][0] >= 0}
            if protected:
                root.allowed = protected
            else:
                lower = max(endgame_bounds[col][0] for col in root.allowed)
                root.allowed = {col for col in root.allowed if endgame_bounds[col][1] >= lower}
        # In completely solved late positions use exact distance only as a
        # tiebreak: delay a forced loss or finish a proven win promptly.
        exact_complete = all(endgame_bounds[col][0] == endgame_bounds[col][1]
                             for col in initial_allowed)
        remaining = min(.30, .88 - cpu_used(),
                        WALL_LIMIT - (time.monotonic() - started_wall) - .08)
        if exact_complete and remaining > .015 and occupied >= 24:
            distance_scores = SOLVER.score_moves(*root.bits, seconds=remaining, preferred=preferred)
            if all(distance_scores[col] is not None for col in root.allowed):
                best_distance = max(distance_scores[col] for col in root.allowed)
                root.allowed = {col for col in root.allowed if distance_scores[col] == best_distance}
    if simulations >= 32 and len(root.allowed) > 1:
        allowance = min(.24, .74 - cpu_used(),
                        WALL_LIMIT - (time.monotonic() - started_wall) - .06)
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
                guide_weight = GUIDE_WEIGHT if GUIDE_WEIGHT>=0 else (.35 if 8 <= occupied <= 26 else .85)
                child.prior = guide_weight * float(weight) + (1 - guide_weight) * float(priors[col])
                child.guide_weight = guide_weight
                child.guide_value = float(np.tanh(scores[col] / 70.))

    # 模拟：重复 simulations 次模拟，每次从 root 开始。
    # 选择：沿已有子节点向下走，记录经过的 path。
    # 评估：终局用实际胜负；非终局调用网络，并扩展叶节点。
    # 回传：把叶节点视角的 value 沿 path 更新。
    for _ in range(simulations):
        # Leave margin below the platform's one-second limit on both CPU and
        # wall clock; the native search is a separate process.
        if simulations >= 32 and _ >= 8:
            if cpu_used() >= CPU_LIMIT or time.monotonic() - started_wall >= WALL_LIMIT:
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
            if child_occupied <= 21:
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
                key = (opponent ^ mask, mask | bit)
                KNOWN_FLOORS[key] = max(KNOWN_FLOORS.get(key, -1), floor)
                if floor == 1:
                    KNOWN_OUTCOMES[key] = 1
    LAST_SEARCH.clear()
    LAST_SEARCH.update({"occupied": occupied, "initial_allowed": initial_allowed,
                        "allowed": sorted(root.allowed), "root_known": root_known, "root_minimum": root_minimum, "opening_bounds": opening_bounds, "outcome_bounds": endgame_bounds,
                        "native_bounds": native_bounds, "native_used": native_used,
                        "distance_scores": distance_scores, "guidance_depth": depth,
                        "simulations": int(counts.sum()),
                        "cpu_seconds": round(cpu_used(), 6),
                        "wall_seconds": round(time.monotonic() - started_wall, 6)})
    return counts
