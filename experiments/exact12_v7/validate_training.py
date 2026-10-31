"""Check domain consequences of regret shaping, including unavoidable losses."""
import json
from pathlib import Path
import numpy as np
import torch
from train import objective


class Probe:
    def __init__(self,scores,value):
        self.logits=torch.zeros((1,7),requires_grad=True)
        self.values=torch.tensor([float(value)],requires_grad=True)
        self.q=torch.tensor(scores.clip(-1,1)[None],dtype=torch.float32,requires_grad=True)
    def training_outputs(self,states):return self.logits,self.values,self.q


def run():
    raw=np.load('data/constrained.npz');scores=raw['scores'];values=raw['values'];records=[]
    for kind in ['avoidable','forced_loss']:
        ids=np.flatnonzero((values>=0)&(scores==-1).any(1)) if kind=='avoidable' else np.flatnonzero((values==-1)&np.all((scores==-1)|(scores==-1000),axis=1))
        i=int(ids[0]);probe=Probe(scores[i],values[i])
        loss=objective(probe,torch.tensor(raw['states'][i:i+1],dtype=torch.float32),
                       torch.tensor(raw['policies'][i:i+1],dtype=torch.float32),torch.tensor(values[i:i+1]),
                       torch.tensor(scores[i:i+1],dtype=torch.float32),torch.tensor([True]))
        loss.backward();gradient=probe.logits.grad.numpy()[0]
        illegal=scores[i]==-1000
        assert np.allclose(gradient[illegal],0)
        if kind=='avoidable':
            optimal=raw['policies'][i]>0;bad=scores[i]==-1
            assert gradient[bad].mean()>gradient[optimal].mean()
        else:
            # Every move loses; uniform logits are already stationary. There
            # must not be an extra penalty blaming a particular legal move.
            assert np.max(np.abs(gradient))<1e-6
        records.append({'case':kind,'illegal_action_gradients_zero':True,'passed':True})
    Path('runs/training_validation.json').write_text(json.dumps(records,indent=2));print(records)


if __name__=='__main__':run()
