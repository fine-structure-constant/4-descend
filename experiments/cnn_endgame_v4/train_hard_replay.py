"""Emphasize proved middlegame corrections while retaining general replay."""
import argparse,json,random,time
from pathlib import Path
import numpy as np
import torch
from support.runtime import load_model,set_seed,save_model,append_log
from train_expert import strong_examples


def main(a):
 set_seed(a.seed);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
 model=load_model(a.model);optimizer=torch.optim.AdamW(model.parameters(),lr=.00015,weight_decay=.0002)
 hard=np.load('data/hard_midgames.npz');ending=np.load('../endgame_v3/data/exact_endgames.npz')
 local=np.load('../standard_v2/data/local.npz');opening=np.load('../endgame_v3/data/public_opening.npz')
 strong=strong_examples('../standard_v2/data/strong.jsonl');start=time.time()
 def take(data,n,policy=True):
  ids=np.random.randint(len(data['states']),size=n)
  return data['states'][ids],data['policies'][ids] if policy else np.zeros((n,7)),data['values'][ids],np.ones(n) if policy else np.zeros(n)
 for step in range(1,a.steps+1):
  chunks=[take(hard,64),take(ending,16),take(local,24),take(opening,16,False)]
  ss=random.choices(strong,k=8)
  chunks.append((np.stack([s.numpy() for s,_,_ in ss]),np.stack([p for _,p,_ in ss]),np.array([z for _,_,z in ss]),np.ones(8)))
  s=torch.tensor(np.concatenate([c[0] for c in chunks]),dtype=torch.float32)
  pi=torch.tensor(np.concatenate([c[1] for c in chunks]),dtype=torch.float32)
  z=torch.tensor(np.concatenate([c[2] for c in chunks]),dtype=torch.float32)
  mask=torch.tensor(np.concatenate([c[3] for c in chunks]),dtype=torch.float32)
  flip=torch.rand(len(s))<.5;s[flip]=s[flip].flip(-1);pi[flip]=pi[flip].flip(-1)
  model.train();logits,value=model(s)
  loss=(-(pi*torch.log_softmax(logits,1)).sum(1)*mask).sum()/mask.sum()+(value-z).square().mean()
  optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
  if step%250==0:
   model.eval()
   with torch.no_grad():
    states=torch.tensor(hard['states'],dtype=torch.float32);lp,_=model(states)
    actions=lp.argmax(1).numpy();accuracy=float(np.mean(hard['policies'][np.arange(len(actions)),actions]>0))
   row={'step':step,'loss':loss.item(),'hard_training_policy_accuracy':accuracy,'seconds':round(time.time()-start,1)}
   print(json.dumps(row),flush=True);append_log(out/'log.jsonl',row)
   save_model(model,out/('step_%d.pt'%step))
 save_model(model,out/'model.pt')


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('model');p.add_argument('--out',default='runs/cnn_replay')
 p.add_argument('--steps',type=int,default=1500);p.add_argument('--seed',type=int,default=202610304)
 main(p.parse_args())
