"""Fine-tune locally trained weights with exact endings and public opening values.

The public opening labels supervise only value, since they do not specify
optimal actions. Policy targets come from proved search or recorded teachers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from model import Network
from support.runtime import load_model,set_seed,save_model,append_log
from train_expert import strong_examples


def split(states):
    train=[];validation=[]
    for i,state in enumerate(states):
        raw=min(state.tobytes(),state[:,:,::-1].tobytes())
        value=int.from_bytes(hashlib.blake2b(raw,digest_size=4).digest(),'little')
        (validation if value%10==0 else train).append(i)
    return np.asarray(train),np.asarray(validation)


def main(a):
    set_seed(a.seed);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    model=Network() if a.from_scratch else load_model(a.model)
    save_model(model,out/"initial.pt")
    expert=np.load('../standard_v2/data/local.npz')
    ending=np.load('../endgame_v3/data/exact_endgames.npz')
    if a.hard:
        extra=np.load(a.hard)
        ending={name:np.concatenate([ending[name]]+[extra[name]]*4)
                for name in ['states','policies','values']}
    opening=np.load('../endgame_v3/data/public_opening.npz')
    strong=strong_examples('../standard_v2/data/strong.jsonl')
    ep=np.concatenate([expert['states'],ending['states'],np.asarray([s.numpy() for s,_,_ in strong])])
    pi=np.concatenate([expert['policies'],ending['policies'],np.asarray([p for _,p,_ in strong])])
    z=np.concatenate([expert['values'],ending['values'],np.asarray([v for _,_,v in strong])])
    policy_mask=np.concatenate([np.ones(len(ep)),np.zeros(len(opening['states']))]).astype(np.float32)
    states=np.concatenate([ep,opening['states']]).astype(np.uint8)
    policies=np.concatenate([pi,np.zeros((len(opening['states']),7),np.float32)]).astype(np.float32)
    values=np.concatenate([z,opening['values']]).astype(np.float32)
    train,val=split(states)
    train_set=set(train.tolist())
    start_end=len(expert['states']);start_strong=start_end+len(ending['states']);start_open=len(ep)
    groups=[]
    for start,stop,size in [(0,start_end,32),(start_end,start_strong,40),
                            (start_strong,start_open,16),(start_open,len(states),40)]:
        ids=np.asarray([i for i in range(start,stop) if i in train_set]);groups.append((ids,size))
    x=torch.tensor(states,dtype=torch.float32);p=torch.tensor(policies);v=torch.tensor(values)
    pm=torch.tensor(policy_mask)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.0002)
    best=float('inf');start=time.time()
    def objective(s,pi,z,mask):
        logits,pred=model(s)
        per_policy=-(pi*torch.log_softmax(logits,dim=1)).sum(1)
        return (per_policy*mask).sum()/mask.sum().clamp_min(1)+(pred-z).square().mean()
    for step in range(1,a.steps+1):
        ids=np.concatenate([np.random.choice(indices,size=size,replace=True) for indices,size in groups])
        s=x[ids].clone();target_pi=p[ids].clone();target_z=v[ids];mask=pm[ids]
        flip=torch.rand(len(ids))<.5;s[flip]=s[flip].flip(-1);target_pi[flip]=target_pi[flip].flip(-1)
        model.train();loss=objective(s,target_pi,target_z,mask)
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if step%250==0:
            model.eval()
            with torch.no_grad():
                metrics=[]
                for name,left,right in [('local',0,start_end),('endgame',start_end,start_strong),
                                         ('strong',start_strong,start_open),('opening',start_open,len(states))]:
                    indices=val[(val>=left)&(val<right)]
                    if not len(indices):continue
                    _,pred=model(x[indices]);mse=(pred-v[indices]).square().mean().item()
                    accuracy=(torch.where(pred>.33,1.,torch.where(pred<-.33,-1.,0.))==v[indices]).float().mean().item()
                    metrics.append((name,mse,accuracy))
                score=sum(m for n,m,acc in metrics if n in ['opening','endgame'])
            row={'step':step,'loss':loss.item(),'validation':metrics,'seconds':round(time.time()-start,1),
                 'positions':len(states),'exact_endgames':len(ending['states']),'public_openings':len(opening['states'])}
            print(json.dumps(row),flush=True);append_log(out/'log.jsonl',row)
            save_model(model,out/('step_%d.pt'%step))
            if score<best:best=score;save_model(model,out/'best.pt')
    save_model(model,out/'model.pt')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='runs/baseline.pt');p.add_argument('--out',default='runs/endgame_finetune')
    p.add_argument('--hard');p.add_argument('--from-scratch',action='store_true');p.add_argument('--steps',type=int,default=4000);p.add_argument('--lr',type=float,default=.0003)
    p.add_argument('--seed',type=int,default=202610302)
    main(p.parse_args())
