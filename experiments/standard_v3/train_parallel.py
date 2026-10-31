"""Parallel game generation, single-threaded inference per worker, mirrored replay."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from model import Network
from train import self_play, compute_loss
from train_expert import strong_examples
from support.runtime import set_seed, load_model, save_model, sample_batch, append_log


def game_job(job):
    seed, weights, simulations = job
    set_seed(seed)
    model=Network()
    model.load_state_dict(weights)
    model.eval()
    return [(s.numpy(),pi,z) for s,pi,z in self_play(model,simulations)]


def main(a):
    set_seed(a.seed)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    model=load_model(a.model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.0001)
    data=np.load(a.data)
    expert=[(torch.tensor(s,dtype=torch.float32),pi,float(z))
            for s,pi,z in zip(data['states'],data['policies'],data['values'])]
    strong=strong_examples(a.strong)
    replay=[];start=time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for iteration in range(1,a.iterations+1):
            weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            jobs=[(a.seed+iteration*1000+i,weights,a.simulations) for i in range(a.games)]
            for game in pool.map(game_job,jobs):
                replay.extend((torch.from_numpy(s),pi,z) for s,pi,z in game)
            replay=replay[-30000:]
            model.train()
            for step in range(a.steps):
                # Preserve general teacher coverage while learning fresh outcomes.
                batch=random.choices(replay,k=64)+random.choices(expert,k=48)
                batch+=random.choices(strong if strong else replay,k=16)
                s,pi,z=sample_batch(batch,128,'cpu')
                flip=torch.rand(128)<.5;s[flip]=s[flip].flip(-1);pi[flip]=pi[flip].flip(-1)
                loss=compute_loss(model,s,pi,z)
                optimizer.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                optimizer.step()
            save_model(model,out/'model.pt')
            if iteration%5==0:save_model(model,out/('iteration_%d.pt'%iteration))
            row={'iteration':iteration,'replay_positions':len(replay),'loss':loss.item(),
                 'seconds':round(time.time()-start,1)}
            append_log(out/'log.jsonl',row);print(json.dumps(row),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model');p.add_argument('--out',default='runs/cnn_selfplay')
    p.add_argument('--data',default='../standard_v2/data/local.npz')
    p.add_argument('--strong',default='../standard_v2/data/strong.jsonl')
    p.add_argument('--iterations',type=int,default=20)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--games',type=int,default=16)
    p.add_argument('--simulations',type=int,default=96)
    p.add_argument('--steps',type=int,default=50)
    p.add_argument('--seed',type=int,default=20261102)
    main(p.parse_args())
