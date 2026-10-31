"""Run from the v5 source directory to keep the legacy search isolated."""
import json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import sys
sys.path.insert(0,str(Path.cwd()))
MODEL=None

def init():
 global MODEL
 from support.runtime import load_model,set_seed
 set_seed(2026103119);MODEL=load_model('../../upload-v5/model.pt')

def act(job):
 import mcts
 from support.endgame import SOLVER
 state,pi,z=job
 mcts.KNOWN_OUTCOMES.clear();mcts.KNOWN_FLOORS.clear();SOLVER.table.clear()
 p=1 if state[0].sum()==state[1].sum() else -1
 b=state[0].astype(np.int8)*p-state[1].astype(np.int8)*p
 t=time.process_time();counts=mcts.search(b,p,MODEL,simulations=128);seconds=time.process_time()-t
 c=int(np.argmax(counts));return {'correct':bool(pi[c]>0),'action':c,'cpu_seconds':seconds,'value':float(z),'root_bounds':mcts.LAST_SEARCH.get('outcome_bounds')}

if __name__=='__main__':
 d=np.load('../exact12_v6/data/tempo_ablation.npz');rows=[]
 with ProcessPoolExecutor(max_workers=6,initializer=init) as pool:
  rows=list(pool.map(act,zip(d['states'],d['policies'],d['values'])))
 result={'positions':len(rows),'correct':sum(r['correct'] for r in rows),'mean_cpu_seconds':sum(r['cpu_seconds'] for r in rows)/len(rows),'max_cpu_seconds':max(r['cpu_seconds'] for r in rows),'records':rows}
 Path('../exact12_v6/runs/v5_tempo_ablation.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='records'}))
