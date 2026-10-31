"""Paired local matches isolate learned weights under the same Python search."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import random
import numpy as np
from support.runtime import load_model,set_seed
from support.game import new_board,play_move,legal_moves,game_result


def play(job):
    candidate,baseline,side,opening,seed=job
    from mcts import search,LAST_SEARCH,KNOWN_FLOORS,KNOWN_OUTCOMES
    set_seed(seed);a=load_model(candidate);b=load_model(baseline)
    KNOWN_FLOORS.clear();KNOWN_OUTCOMES.clear()
    board=new_board();player=1;history=list(opening);slowest=0
    for action in opening:board=play_move(board,action,player);player=-player
    while True:
        model=a if side==player else b
        visits=search(board,player,model,simulations=128)
        action=int(np.argmax(visits));assert action in legal_moves(board)
        slowest=max(slowest,LAST_SEARCH.get('cpu_seconds',0))
        assert slowest<.95
        board=play_move(board,action,player);history.append(action);done,winner=game_result(board)
        if done:break
        player=-player
    result='draw' if not winner else 'win' if winner==side else 'loss'
    return {'side':side,'opening':opening,'result':result,'history':history,'max_cpu_seconds':slowest}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('candidate');p.add_argument('baseline')
    p.add_argument('--output',default='runs/arena.json');p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    openings=[[],[3,3],[2,4],[1,5]]
    jobs=[(args.candidate,args.baseline,side,opening,2026103111+i) for i,opening in enumerate(openings) for side in [1,-1]]
    records=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(play,jobs):
            records.append(record);print(json.dumps(record),flush=True)
    report={'candidate':args.candidate,'baseline':args.baseline,'same_search_both_sides':True,
            'total':{r:sum(x['result']==r for x in records) for r in ['win','draw','loss']},'records':records}
    Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps(report['total']),flush=True)
