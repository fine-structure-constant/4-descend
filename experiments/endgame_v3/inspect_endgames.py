"""Prove outcomes of replayed strong positions and locate lost alternatives."""
import argparse
import json
from pathlib import Path
import time
from support.game import new_board, play_move, legal_moves
from support.tactics import position, safe_moves, COLUMNS
from support.endgame import ExactSolver

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('report')
p.add_argument('--seconds',type=float,default=1.)
p.add_argument('--start',type=int,default=16)
p.add_argument('--output',required=True)
a=p.parse_args()
report=json.loads(Path(a.report).read_text());rows=[]
for game in report['records']:
 if game['result']!='loss':continue
 board=new_board();player=1;solver=ExactSolver()
 for step,played in enumerate(game['history']):
  if player==game['side'] and step>=a.start:
   own,mask=position(board,player)
   t=time.process_time();scores=solver.score_moves(own,mask,a.seconds,preferred=played)
   proven=[v for v in scores if v is not None]
   safe,_=safe_moves(own,mask)
   row={'game':game['game'],'ply':step+1,'played':played,'scores':scores,
        'legal_columns':legal_moves(board),'safe_columns':[c for c in range(7) if safe&COLUMNS[c]],
        'cpu_seconds':round(time.process_time()-t,3),'nodes':solver.nodes,
        'prefix':game['history'][:step]}
   rows.append(row);print(json.dumps(row),flush=True)
   Path(a.output).write_text(json.dumps(rows,indent=2))
  board=play_move(board,played,player);player=-player
