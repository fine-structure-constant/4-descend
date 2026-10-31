"""Independent array-board WDL checks and cold-start replay timing checks."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import time
import numpy as np
from support.game import new_board,play_move,game_result,legal_moves
from support.runtime import load_model,set_seed


def winner_array(board):
    for r in range(6):
        for c in range(7):
            p=board[r,c]
            if not p:continue
            for dr,dc in [(1,0),(0,1),(1,1),(1,-1)]:
                rr,cc=r+3*dr,c+3*dc
                if 0<=rr<6 and 0<=cc<7 and all(board[r+k*dr,c+k*dc]==p for k in range(1,4)):
                    return int(p)
    return 0


@lru_cache(maxsize=300000)
def brute(raw,player):
    b=np.frombuffer(raw,dtype=np.int8).reshape(6,7)
    winner=winner_array(b)
    if winner:return 1 if winner==player else -1
    if np.all(b[0]!=0):return 0
    best=-1
    for c in range(7):
        empty=np.flatnonzero(b[:,c]==0)
        if not len(empty):continue
        child=b.copy();child[empty[-1],c]=player
        best=max(best,-brute(child.tobytes(),-player))
        if best==1:break
    return best


def validate_data(data):
    p=np.load(data);rng=np.random.RandomState(2026103108)
    candidates=np.flatnonzero(p['occupied']>=34);rng.shuffle(candidates)
    checked=0
    for i in candidates[:100]:
        state=p['states'][i];b=(state[0].astype(np.int8)-state[1].astype(np.int8))
        assert not winner_array(b)
        actual=brute(b.tobytes(),1)
        assert actual==p['values'][i],(i,actual,p['values'][i])
        for c in np.flatnonzero(p['policies'][i]>0):
            child=b.copy();row=np.flatnonzero(child[:,c]==0)[-1];child[row,c]=1
            assert -brute(child.tobytes(),-1)==actual,(i,c)
        checked+=1
    return {'independent_late_positions':checked,'cached_array_states':brute.cache_info().currsize}


def validate_search(path):
    import mcts
    set_seed(2026103108);model=load_model(path)
    folder=Path(__file__).resolve().parent.parent/'v4_release/runs/flash_probe'
    histories=[]
    for name in ['match_live.json','exact12_empty_black_live.json','exact12_random2_white_live.json']:
        moves=[f['action'] for f in json.loads((folder/name).read_text())['frames'][1:]]
        histories.extend(moves[:n] for n in [10,12,13,14,16,20] if n<len(moves))
    records=[]
    for history in histories:
        mcts.KNOWN_OUTCOMES.clear();mcts.KNOWN_FLOORS.clear();board=new_board();player=1
        for c in history:board=play_move(board,c,player);player=-player
        begin=time.process_time();wall=time.perf_counter()
        counts=mcts.search(board,player,model,simulations=128)
        cpu=time.process_time()-begin;elapsed=time.perf_counter()-wall;action=int(np.argmax(counts))
        assert action in legal_moves(board) and cpu<.95 and elapsed<2
        records.append({'history':history,'action':action,'cpu_seconds':cpu,'wall_seconds':elapsed,'search':dict(mcts.LAST_SEARCH)})
    return {'records':records,'max_cpu_seconds':max(r['cpu_seconds'] for r in records)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',default='data/proofs.npz');p.add_argument('--model')
    a=p.parse_args();result={'data':validate_data(a.data)}
    if a.model:result['search']=validate_search(a.model)
    Path('runs/validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
