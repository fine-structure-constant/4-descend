"""Check exact-leaf agreement, opening/model boundary and move CPU budgets."""
import json,random,time
from pathlib import Path
import numpy as np
from support import _endgame_native,pons_book,movebook
from support.endgame import configure_accelerator
from support.game import new_board,play_move,legal_moves,game_result
from support.tactics import position
from support.runtime import load_model,set_seed
from mcts import search,LAST_SEARCH

set_seed(20261031);model=load_model('runs/model_v4_final.pt')
assert configure_accelerator(model);assert movebook.configure(model);assert pons_book.configure(model)
rng=random.Random(91031);hits=0;checked=0;max_cpu=0.;phase_counts={'opening':0,'model_search':0}
for game in range(100):
 b=new_board();p=1
 for n in range(15):
  if game_result(b)[0]:break
  own,mask=position(b,p)
  a=_endgame_native.book_score(own,mask);z=pons_book.score(own,mask)
  assert a==z,(n,a,z)
  checked+=1;hits+=a is not None
  if game<6 and n in (0,4,10,11,12,13):
   calls=[0]
   hook=model.register_forward_hook(lambda *args:calls.__setitem__(0,calls[0]+1))
   start=time.process_time();visits=search(b,p,model,simulations=128);elapsed=time.process_time()-start
   hook.remove();phase=LAST_SEARCH['phase'];phase_counts[phase]+=1;max_cpu=max(max_cpu,elapsed)
   assert int(np.argmax(visits)) in legal_moves(b)
   assert (phase=='opening')==(n<12),(n,phase)
   if n>=12:assert calls[0]>0,(n,calls)
   if n<12:
    move=int(np.argmax(visits));entry=movebook.lookup(own,mask)
    assert entry[0]&(1<<move)
   assert elapsed<1.,(n,elapsed)
  choices=[c for c in legal_moves(b) if not game_result(play_move(b,c,p))[0]]
  if not choices:break
  b=play_move(b,rng.choice(choices),p);p=-p
report={'exact_leaf_checks':checked,'book_hits':hits,'phase_counts':phase_counts,'max_cpu_seconds':max_cpu}
Path('runs/native_boundary_validation.json').write_text(json.dumps(report,indent=2));print(report)
