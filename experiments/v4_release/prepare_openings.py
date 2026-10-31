"""Predeclare paired stability openings before any new evaluation results."""
import json,random
from pathlib import Path
from support.runtime import load_model,set_seed
from support.movebook import configure,lookup,forward_key,reverse_key
from support.game import new_board,play_move,legal_moves,game_result
from support.tactics import position,safe_moves,popcount

set_seed(20261031);model=load_model('runs/model_v4_final.pt');assert configure(model)
anchors=[[],[3,3],[3,4],[3,2],[2,3],[4,3],[3,4,1,0],[3,2,5,6]]
seen=set();rng=random.Random(2026103101)
for h in anchors:
 b=new_board();p=1
 for c in h:b=play_move(b,c,p);p=-p
 own,mask=position(b,p);seen.add(min(forward_key(own,mask),reverse_key(own,mask)))
while len(anchors)<20:
 b=new_board();p=1;h=[];depth=rng.choice([4,6,8,10]);valid=True
 for step in range(depth):
  choices=[c for c in legal_moves(b) if not game_result(play_move(b,c,p))[0]]
  if not choices:valid=False;break
  c=rng.choice(choices);h.append(c);b=play_move(b,c,p);p=-p
 if not valid:continue
 own,mask=position(b,p);key=min(forward_key(own,mask),reverse_key(own,mask))
 if key in seen:continue
 item=lookup(own,mask);safe,status=safe_moves(own,mask)
 if item is None or item[1]!=0 or status!=0 or popcount(safe)<2:continue
 seen.add(key);anchors.append(h)
path=Path('runs/stability_openings.json');path.write_text(json.dumps({'seed':2026103101,'anchors':8,'balanced_draw_openings':12,'groups':anchors},indent=2))
print(json.dumps(anchors))
