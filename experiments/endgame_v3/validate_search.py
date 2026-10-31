"""Independent tactical and endgame checks against the supplied array rules."""
import functools
import random
import numpy as np
from support.endgame import ExactSolver
from support.game import new_board, legal_moves, play_move, game_result
from support.tactics import position, possible, COLUMNS, winning_cells, safe_moves, Teacher


@functools.lru_cache(maxsize=200000)
def brute(encoded, player):
    board=np.frombuffer(encoded,dtype=np.int8).reshape(6,7)
    done,winner=game_result(board)
    if done:return 0 if not winner else 1 if winner==player else -1
    best=-1
    for action in legal_moves(board):
        best=max(best,-brute(play_move(board,action,player).tobytes(),-player))
        if best==1:break
    return best


def main():
    rng=random.Random(302610)
    positions=endgames=0
    for game in range(1200):
        b=new_board();p=1
        for step in range(42):
            own,mask=position(b,p)
            for side,bits in [(p,own),(-p,own^mask)]:
                predicted=winning_cells(bits,mask)&possible(mask)
                for c in legal_moves(b):
                    win=game_result(play_move(b,c,side))==(True,side)
                    assert bool(predicted&COLUMNS[c])==win
            safe,status=safe_moves(own,mask)
            for c in legal_moves(b):
                after=play_move(b,c,p)
                if game_result(after)[0]:continue
                lost=any(game_result(play_move(after,d,-p))==(True,-p) for d in legal_moves(after))
                if status!=1:assert bool(safe&COLUMNS[c])==(not lost)
            positions+=1
            if step>=34 and endgames<30:
                expected=brute(b.tobytes(),p)
                scores=Teacher(depth=42-step,node_limit=1000000).scores(own,mask)
                actual=int(np.sign(max(scores)))
                assert expected==actual,(expected,actual,scores,b)
                exact=ExactSolver().score_moves(own,mask,seconds=2.)
                for c in legal_moves(b):
                    child=play_move(b,c,p)
                    outcome=-brute(child.tobytes(),-p)
                    assert int(np.sign(exact[c]))==outcome,(c,exact,outcome,b)
                bounds=ExactSolver().outcome_bounds(own,mask,seconds=2.)
                for c in legal_moves(b):
                    outcome=-brute(play_move(b,c,p).tobytes(),-p)
                    assert bounds[c][0]<=outcome<=bounds[c][1],(c,bounds,outcome,b)
                    assert bounds[c][0]==bounds[c][1] or any(v==(1,1) for v in bounds)
                exact_best=max(v for v in exact if v is not None)
                assert expected==int(np.sign(exact_best)),(expected,exact,b)
                assert all(exact[c] is not None for c in legal_moves(b))
                endgames+=1
                if endgames==30:break
            # Bias towards nonterminal play to obtain substantial endgames.
            choices=[c for c in legal_moves(b) if not game_result(play_move(b,c,p))[0]]
            action=rng.choice(choices if choices and rng.random()<.9 else legal_moves(b))
            b=play_move(b,action,p)
            if game_result(b)[0]:break
            p=-p
        if endgames>=30 and positions>=3000:break
    assert endgames==30
    print({'tactical_positions_checked':positions,'exact_endgames_checked':endgames})


if __name__=='__main__':main()
