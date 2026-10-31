"""Offline native search labels for difficult middlegames; pure Python inference."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import torch
from types import SimpleNamespace
from support.game import new_board,play_move,game_result,legal_moves
from support.tactics import position
from support.deep_opening import configure,lookup
from validate_search import brute

BINARY='/tmp/homework4-exact-teacher'


def label(job):
    encoded,player,preferred,seconds=job
    b=np.frombuffer(encoded,dtype=np.int8).reshape(6,7)
    own,mask=position(b,player);n=int(np.count_nonzero(b))
    known=lookup(own,mask,n) if n==12 else None
    text=f'{own} {mask} {known if known is not None else 9} {seconds} {preferred}\n'
    result=subprocess.run([BINARY],input=text,text=True,capture_output=True,timeout=seconds+10,check=True)
    values=[int(v) for v in result.stdout.split()][:7]
    wins=[c for c in legal_moves(b) if values[c]==1]
    if wins:choices=wins;z=1.
    elif all(values[c] in [-1,0,1] for c in legal_moves(b)):
        z=max(values[c] for c in legal_moves(b));choices=[c for c in legal_moves(b) if values[c]==z]
    else:return None
    pi=np.zeros(7,dtype=np.float32);pi[choices]=1/len(choices)
    state=np.stack([b==player,b==-player,b==0]).astype(np.uint8)
    return state,pi,z,n,values


def validate():
    rng=random.Random(7781);checks=0
    for _ in range(200):
        b=new_board();player=1
        for step in range(42):
            if step>=34:
                own,mask=position(b,player)
                text=f'{own} {mask} 9 1 -1\n'
                result=subprocess.run([BINARY],input=text,text=True,capture_output=True,check=True)
                vals=[int(v) for v in result.stdout.split()][:7]
                for c in legal_moves(b):
                    if vals[c] in (-1,0,1):
                        truth=-brute(play_move(b,c,player).tobytes(),-player)
                        assert vals[c]==truth,(vals,truth,c,b)
                checks+=1;break
            choices=[c for c in legal_moves(b) if not game_result(play_move(b,c,player))[0]]
            if not choices:break
            b=play_move(b,rng.choice(choices),player);player=-player
        if checks==30:break
    assert checks==30
    print({'native_teacher_independent_endgames':checks},flush=True)


def main(a):
    validate()
    raw=Path('data/public/book_12ply.dat').read_bytes()
    configure(SimpleNamespace(opening12=torch.frombuffer(bytearray(raw),dtype=torch.uint8)))
    positions={}
    histories=[]
    for line in Path('../standard_v2/data/strong.jsonl').read_text().splitlines():histories.append(json.loads(line)['history'])
    for name in a.reports:
        for r in json.loads(Path(name).read_text())['records']:histories.append(r['history'])
    for h in histories:
        b=new_board();player=1
        for step,c in enumerate(h):
            if 12<=step<=26:positions[(b.tobytes(),player)]=c
            b=play_move(b,c,player)
            if game_result(b)[0]:break
            player=-player
    jobs=[(b,p,c,a.seconds) for (b,p),c in positions.items()]
    random.Random(a.seed).shuffle(jobs)
    jobs=jobs[:a.maximum]
    states=[];policies=[];values=[];moves=[];start=time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for index,result in enumerate(pool.map(label,jobs),1):
            if result is not None:
                s,pi,z,n,sc=result;states.append(s);policies.append(pi);values.append(z);moves.append(n)
            if index%50==0 or index==len(jobs):
                Path(a.output).parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(a.output,states=np.asarray(states),policies=np.asarray(policies),
                                    values=np.asarray(values,dtype=np.float32),occupied=np.asarray(moves))
                print(json.dumps({'attempted':index,'total':len(jobs),'proved':len(states),
                                  'seconds':round(time.time()-start,1)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports',nargs='*',default=['runs/cnn_strong_dev.json',
                    '../endgame_v3/runs/book_endgame_strong.json'])
    p.add_argument('--seconds',type=float,default=6.)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--maximum',type=int,default=900)
    p.add_argument('--seed',type=int,default=202610303)
    p.add_argument('--output',default='data/hard_midgames.npz')
    main(p.parse_args())
