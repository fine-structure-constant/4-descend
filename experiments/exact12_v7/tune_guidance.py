"""Compare floating-threat guidance only on the internal development split."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time
import numpy as np
MODEL=None


def init():
    global MODEL
    from support.runtime import load_model,set_seed
    set_seed(2026103125);MODEL=load_model('runs/refined/model.pt')


def act(job):
    import mcts
    from support.tactics import Teacher,heuristic,fast_heuristic,SearchTimeout
    from support.endgame import SOLVER
    state,policy,scores,mode,weight,index=job
    evaluator=heuristic if mode=='legacy' else fast_heuristic
    def guide(own,mask,seconds):
        teacher=Teacher(node_limit=10000000,evaluator=evaluator)
        teacher.deadline=time.process_time()+seconds;scores=None;depth=0
        for target in (4,6,8,10,12,14,16,20,42):
            try:current=teacher.scores(own,mask,target)
            except SearchTimeout:break
            scores,depth=current,target
            legal=[x for x in scores if x>-20000]
            if max(legal)>10000 or all(abs(x)>10000 for x in legal):break
        return scores,depth
    mcts.root_guidance=guide;mcts.GUIDE_WEIGHT=weight
    mcts.KNOWN_FLOORS.clear();mcts.KNOWN_OUTCOMES.clear();SOLVER.table.clear()
    board=state[0].astype(np.int8)-state[1].astype(np.int8)
    counts=mcts.search(board,1,MODEL,simulations=128);action=int(counts.argmax())
    return {'index':int(index),'mode':mode,'weight':weight,'action':action,'correct':bool(policy[action]>0),
            'played_wdl':int(scores[action]),'depth':mcts.LAST_SEARCH.get('guidance_depth',0)}


if __name__=='__main__':
    from support.tactics import position,winning_cells,possible
    from support.constrained import shape
    d=np.load('data/refinement.npz');ids=[]
    for i,state in enumerate(d['states']):
        if not d['validation'][i] or not 14<=d['occupied'][i]<=26:continue
        own,mask=position(state[0].astype(np.int8)-state[1].astype(np.int8),1)
        if shape(mask)['main_columns']<=3 or winning_cells(own,mask)&possible(mask):continue
        if (d['scores'][i]==-1).any() and d['values'][i]>=0:ids.append(i)
    rng=np.random.RandomState(2026103125);rng.shuffle(ids);ids=ids[:48]
    variants=[('legacy',-1),('legacy',.10),('supported',.20),('supported',.10)]
    jobs=[(d['states'][i],d['policies'][i],d['scores'][i],mode,weight,i) for mode,weight in variants for i in ids]
    with ProcessPoolExecutor(max_workers=8,initializer=init) as pool:rows=list(pool.map(act,jobs))
    summary=[]
    for mode,weight in variants:
        items=[r for r in rows if r['mode']==mode and r['weight']==weight]
        summary.append({'mode':mode,'weight':weight,'correct':sum(r['correct'] for r in items),'positions':len(items),
                        'mean_completed_depth':sum(r['depth'] for r in items)/len(items)})
    report={'internal_validation_only':True,'seed':2026103125,'summary':summary,'records':rows}
    Path('runs/guidance_comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(summary,indent=2))
