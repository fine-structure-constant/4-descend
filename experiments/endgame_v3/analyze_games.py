"""Replay losses and find search-detectable turning points."""
import argparse
import json
from pathlib import Path
import numpy as np
from support.game import new_board, play_move
from support.tactics import Teacher, position

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('report')
p.add_argument('--depth',type=int,default=9)
a=p.parse_args()
report=json.loads(Path(a.report).read_text())
for game in report['records']:
    if game['result']!='loss': continue
    board=new_board(); player=1
    for step,action in enumerate(game['history']):
        if player==game['side'] and step>=len(game['opening']):
            scores=Teacher(a.depth).scores(*position(board,player))
            best=max(scores)
            if best>scores[action]+10000 or (best>10000 and scores[action]<10000):
                print(json.dumps({'game':game['game'],'ply':step+1,'played':action,
                                  'recommended':int(np.argmax(scores)),'scores':scores,
                                  'prefix':game['history'][:step]}),flush=True)
                break
        board=play_move(board,action,player);player=-player
