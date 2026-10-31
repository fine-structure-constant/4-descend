"""Generate proof-labelled positions from teacher/replay games and diverse finishes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import json
import multiprocessing
from pathlib import Path
import random
import time
import numpy as np
from support.endgame import ExactSolver
from support.game import new_board, play_move, game_result, legal_moves
from support.tactics import position, possible, COLUMNS, safe_moves, popcount

SOLVER=ExactSolver(capacity=200000)


def label(job):
    encoded,player,seconds=job
    board=np.frombuffer(encoded,dtype=np.int8).reshape(6,7)
    own,mask=position(board,player)
    bounds=SOLVER.outcome_bounds(own,mask,seconds=seconds)
    legal=legal_moves(board)
    wins=[col for col in legal if bounds[col]==(1,1)]
    if wins:
        choices=wins;z=1.
    elif all(bounds[col][0]==bounds[col][1] for col in legal):
        z=max(bounds[col][0] for col in legal)
        choices=[col for col in legal if bounds[col][0]==z]
    else:
        return None
    pi=np.zeros(7,dtype=np.float32);pi[choices]=1/len(choices)
    state=np.stack([board==player,board==-player,board==0]).astype(np.uint8)
    return state,pi,z,popcount(mask)


def build(a):
    rng=random.Random(a.seed)
    positions={}
    def add_history(history):
        b=new_board();p=1
        for action in history:
            if 10<=np.count_nonzero(b)<=37:
                positions[(b.tobytes(),p)]=None
            b=play_move(b,action,p)
            if game_result(b)[0]:break
            p=-p
    for line in Path(a.strong).read_text().splitlines():
        add_history(json.loads(line)['history'])
    for path in a.reports:
        for game in json.loads(Path(path).read_text()).get('records',[]):
            add_history(game['history'])
    for game in range(a.random_games):
        b=new_board();p=1;history=[]
        for step in range(42):
            own,mask=position(b,p);safe,status=safe_moves(own,mask)
            choices=[c for c in legal_moves(b) if safe&COLUMNS[c]]
            if not choices:choices=legal_moves(b)
            # Delay terminal finishes most of the time to reach constrained boards.
            nonterminal=[c for c in choices if not game_result(play_move(b,c,p))[0]]
            c=rng.choice(nonterminal if nonterminal and rng.random()<.85 else choices)
            history.append(c);b=play_move(b,c,p)
            if game_result(b)[0]:break
            p=-p
        add_history(history)
    items=list(positions)
    rng.shuffle(items)
    # Avoid spending the dataset budget entirely on early unsolved states.
    items.sort(key=lambda item: np.count_nonzero(np.frombuffer(item[0],dtype=np.int8)),reverse=True)
    items=items[:a.max_positions]
    states=[];policies=[];values=[];moves=[];count=Counter();start=time.time()
    jobs=((b,p,a.seconds) for b,p in items)
    context=multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=context) as pool:
        for index,result in enumerate(pool.map(label,jobs,chunksize=1),1):
            if result is not None:
                state,pi,z,occupied=result
                states.append(state);policies.append(pi);values.append(z);moves.append(occupied);count[str(z)]+=1
            if index%200==0 or index==len(items):
                print(json.dumps({'attempted':index,'total':len(items),'proved':len(states),
                                  'outcomes':dict(count),'seconds':round(time.time()-start,1)}),flush=True)
                Path(a.output).parent.mkdir(parents=True,exist_ok=True)
                # An atomic snapshot preserves useful completed labels on interruption.
                temporary=Path(a.output+'.tmp.npz')
                np.savez_compressed(temporary,states=np.asarray(states),policies=np.asarray(policies),
                                    values=np.asarray(values,dtype=np.float32),occupied=np.asarray(moves))
                temporary.replace(a.output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--strong',default='../standard_v2/data/strong.jsonl')
    p.add_argument('--reports',nargs='*',default=['../standard_v2/runs/hybrid_strong.json',
                    '../standard_v2/runs/candidate5_standard_holdout.json'])
    p.add_argument('--random-games',type=int,default=200)
    p.add_argument('--max-positions',type=int,default=8000)
    p.add_argument('--seconds',type=float,default=.8)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--seed',type=int,default=202610301)
    p.add_argument('--output',default='data/exact_endgames.npz')
    build(p.parse_args())
