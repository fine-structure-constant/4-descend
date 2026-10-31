"""Prove spare-tempo endgames and merge them with general replay, without leakage."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import json
from pathlib import Path
import random
import time
import numpy as np
from prepare_data import worker_init,label
from support.game import new_board,play_move,game_result,legal_moves
from support.tactics import position,safe_moves,COLUMNS,possible,alignment,popcount
from support.constrained import shape,applicable

ROOT=Path(__file__).resolve().parent


def collect(args):
    rng=random.Random(args.seed);candidates={}
    def add(history,source):
        b=new_board();p=1
        for i,c in enumerate(history):
            if 18<=i<=40:
                own,mask=position(b,p);s=shape(mask)
                if s['main_columns']<=4:
                    key=min(b.tobytes(),b[:,::-1].tobytes())
                    candidates.setdefault(key,(history[:i],source,s))
            b=play_move(b,c,p)
            if game_result(b)[0]:break
            p=-p
    for version in ['v4_release','exact12_v5']:
        for path in sorted((ROOT.parent/version/'runs/flash_probe').glob('*live.json')):
            add([f['action'] for f in json.loads(path.read_text())['frames'][1:]],'flash')
        for path in (ROOT.parent/version/'runs').glob('*strong*.json'):
            try:records=json.loads(path.read_text()).get('records',[])
            except (AttributeError,ValueError):continue
            for r in records:
                if 'history' in r:add(r['history'],'replay')
    for game in range(args.games):
        b=new_board();p=1;history=[]
        concentration=rng.uniform(1.5,4.)
        for step in range(42):
            own,mask=position(b,p);safe,status=safe_moves(own,mask)
            choices=[c for c in legal_moves(b) if safe&COLUMNS[c]] or legal_moves(b)
            nonterminal=[c for c in choices if not alignment(own | (possible(mask)&COLUMNS[c]))]
            if nonterminal:choices=nonterminal
            weights=[(1+popcount((mask>>(7*c))&63))**concentration for c in choices]
            c=rng.choices(choices,weights=weights)[0]
            b=play_move(b,c,p);history.append(c)
            if game_result(b)[0]:break
            p=-p
        add(history,'saturation')
    entries=list(candidates.values());rng.shuffle(entries)
    closed=[e for e in entries if e[2]['main_columns']<=3]
    boundary=[e for e in entries if e[2]['main_columns']==4]
    closed.sort(key=lambda e:e[1]!='flash');boundary.sort(key=lambda e:e[1]!='flash')
    selected=closed[:args.positions*2//3]+boundary[:args.positions//3]
    rng.shuffle(selected);rows=[];start=time.time()
    with ProcessPoolExecutor(max_workers=args.workers,initializer=worker_init) as pool:
        for i,r in enumerate(pool.map(label,((h,args.seconds,'closed3' if s['main_columns']<=3 else 'boundary4') for h,source,s in selected),chunksize=12),1):
            if r is not None:rows.append(r)
            if i%2000==0 or i==len(selected):
                print(json.dumps({'attempted':i,'selected':len(selected),'proved':len(rows),'seconds':time.time()-start}),flush=True)
    focused={'states':np.stack([r[0] for r in rows]),'policies':np.stack([r[1] for r in rows]),
             'values':np.array([r[2] for r in rows],np.float32),'scores':np.stack([r[3] for r in rows]),
             'occupied':np.array([r[4] for r in rows]),'source':np.array([r[5] for r in rows]),
             'validation':np.array([r[6] for r in rows],bool),'histories':np.array([r[8] for r in rows])}
    np.savez_compressed(ROOT/args.focused,**focused)
    names=['states','policies','values','scores','occupied','source','validation']
    archive=np.load(ROOT/args.base);raw={k:archive[k] for k in names}
    merged={k:[] for k in names};seen=set()
    for data in [focused,raw]:
        for i,s in enumerate(data['states']):
            key=min(s.tobytes(),s[:,:,::-1].tobytes())
            if key in seen:continue
            seen.add(key)
            for k in names:merged[k].append(data[k][i])
    merged={k:np.asarray(v) for k,v in merged.items()}
    np.savez_compressed(ROOT/args.output,**merged)
    report={'focused_positions':len(rows),'merged_positions':len(seen),'sources':dict(Counter(map(str,merged['source']))),
            'seed':args.seed,'mirror_equivalent_states_deduplicated':True}
    report_name='dataset.json' if args.output=='data/proofs.npz' else Path(args.output).stem+'_dataset.json'
    (ROOT/'runs'/report_name).write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--games',type=int,default=6500);p.add_argument('--positions',type=int,default=15000)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--seconds',type=float,default=.35)
    p.add_argument('--seed',type=int,default=2026103116);p.add_argument('--base',default='../exact12_v5/data/proofs.npz')
    p.add_argument('--focused',default='data/constrained.npz');p.add_argument('--output',default='data/proofs.npz')
    collect(p.parse_args())
