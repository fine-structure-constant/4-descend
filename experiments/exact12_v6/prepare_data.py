"""Collect diverse replay positions and proof-label them with an offline solver."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
import numpy as np

from support.game import new_board, play_move, game_result, legal_moves
from support.tactics import position, safe_moves, possible, COLUMNS, alignment

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
PROCESS = None


def worker_init():
    global PROCESS
    PROCESS = subprocess.Popen([str(ROOT/'tools/label_positions'),
        str(ROOT.parent/'cnn_endgame_v4/data/public/7x6.book')],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)


def label(job):
    history, seconds, source = job
    text = ''.join(str(c+1) for c in history) or '-'
    PROCESS.stdin.write(text+' '+str(seconds)+'\n');PROCESS.stdin.flush()
    fields = PROCESS.stdout.readline().split()
    if len(fields)!=8 or fields[0]!=text:
        raise RuntimeError('Teacher protocol failed')
    scores = np.array(list(map(int,fields[1:])),dtype=np.int16)
    legal = scores!=-1000
    complete = np.all(scores[legal]!=-999)
    winning = scores==1
    if winning.any():
        value=1;optimal=winning
    elif complete:
        value=int(scores[legal].max());optimal=(scores==value)&legal
    else:
        return None
    board=new_board();player=1
    for c in history:
        board=play_move(board,c,player);player=-player
    state=np.stack([board==player,board==-player,board==0]).astype(np.uint8)
    canonical=min(state.tobytes(),state[:,:,::-1].tobytes())
    split=int.from_bytes(hashlib.blake2b(canonical,digest_size=8).digest(),'little')%10==0
    return state,optimal.astype(np.float32)/optimal.sum(),value,scores,len(history),source,int(split),canonical,text


def collect(args):
    rng=random.Random(args.seed);positions={}
    def add(history,source):
        board=new_board();player=1
        for i,c in enumerate(history):
            if 10<=i<=36:
                key=min(board.tobytes(),board[:,::-1].tobytes())
                if key not in positions:positions[key]=(history[:i],source)
            board=play_move(board,c,player)
            if game_result(board)[0]:break
            player=-player
    for path in sorted((ROOT.parent/'v4_release/runs/flash_probe').glob('*live.json')):
        add([f['action'] for f in json.loads(path.read_text())['frames'][1:]],'flash')
    for folder in ['v4_release','cnn_endgame_v4','endgame_v3','standard_v2']:
        for path in sorted((ROOT.parent/folder/'runs').glob('*.json')):
            try: records=json.loads(path.read_text()).get('records',[])
            except (ValueError,AttributeError):continue
            for record in records:
                if 'history' in record:add(record['history'],'replay')
    for game in range(args.games):
        board=new_board();player=1;history=[]
        for step in range(42):
            own,mask=position(board,player);safe,status=safe_moves(own,mask)
            choices=[c for c in legal_moves(board) if safe&COLUMNS[c]] or legal_moves(board)
            # Some exploratory replies and many safe continuations; never create an illegal board.
            if rng.random()<.10:choices=legal_moves(board)
            nonterminal=[c for c in choices if not alignment(own | (possible(mask)&COLUMNS[c]))]
            if nonterminal and rng.random()<.75:choices=nonterminal
            c=rng.choice(choices);history.append(c)
            board=play_move(board,c,player)
            if game_result(board)[0]:break
            player=-player
        add(history,'diverse')
    groups={}
    for history,source in positions.values():
        band='middle' if len(history)<=25 else 'ending'
        groups.setdefault(band,[]).append((history,source))
    selected=[]
    for band,fraction in [('middle',.75),('ending',.25)]:
        entries=groups[band];rng.shuffle(entries)
        entries.sort(key=lambda x: x[1]!='flash')
        selected.extend(entries[:int(args.positions*fraction)])
    rng.shuffle(selected)
    output=ROOT/args.output
    if output.exists():raise FileExistsError(output)
    rows=[];start=time.time();stats=Counter()
    with ProcessPoolExecutor(max_workers=args.workers,initializer=worker_init) as pool:
        for count,result in enumerate(pool.map(label,((h,args.seconds,s) for h,s in selected),chunksize=12),1):
            if result is not None:
                rows.append(result);stats[str(result[2])]+=1
            if count%1000==0 or count==len(selected):
                print(json.dumps({'attempted':count,'total':len(selected),'proved':len(rows),'outcomes':dict(stats),'seconds':round(time.time()-start,1)}),flush=True)
    np.savez_compressed(output,states=np.stack([r[0] for r in rows]),policies=np.stack([r[1] for r in rows]),
        values=np.array([r[2] for r in rows],np.float32),scores=np.stack([r[3] for r in rows]),
        occupied=np.array([r[4] for r in rows]),source=np.array([r[5] for r in rows]),
        histories=np.array([r[8] for r in rows]),
        validation=np.array([r[6] for r in rows],bool))
    report={'positions':len(rows),'attempted':len(selected),'seconds':time.time()-start,
            'outcomes':dict(stats),'validation':sum(r[6] for r in rows),'seed':args.seed,
            'teacher':'Pascal Pons reference solver with per-position deadline; unknowns excluded',
            'source_counts':dict(Counter(r[5] for r in rows))}
    output.with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--games',type=int,default=5000);p.add_argument('--positions',type=int,default=30000)
    p.add_argument('--seconds',type=float,default=.20);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--seed',type=int,default=2026103105);p.add_argument('--output',default='data/proofs.npz')
    collect(p.parse_args())
