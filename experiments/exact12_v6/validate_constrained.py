"""Cross-check every emitted bound against separately generated C++ proof labels."""
import json
from pathlib import Path
import random
import numpy as np
from support.constrained import ConstrainedSolver,shape,LINES
from support.tactics import position,winning_cells,possible


def run():
    assert len(set(LINES))==69
    data=np.load('data/constrained.npz');states=data['states'];scores=data['scores'];values=data['values']
    solver=ConstrainedSolver();plain=ConstrainedSolver();checked=0;proved=0;comparisons=0;max_cpu=0
    tempos=0;critical=0;dead_cases=0;examples=[];hard=0
    order=list(range(len(states)));random.Random(2026103118).shuffle(order)
    for i in order:
        b=(states[i,0].astype(np.int8)-states[i,1].astype(np.int8));own,mask=position(b,1);s=shape(mask)
        if s['main_columns']>3:continue
        bounds=solver.analyze(own,mask,seconds=.70)
        checked+=1;max_cpu=max(max_cpu,solver.last['cpu_seconds'])
        for c,bound in enumerate(bounds):
            if bound is None:assert scores[i,c]==-1000;continue
            if scores[i,c] in [-1,0,1]:
                assert bound[0]<=scores[i,c]<=bound[1],(i,c,bound,int(scores[i,c]))
        solved=solver.last['complete'] or solver.last['proved_win']
        if solved:
            proved+=1
            assert max(b[0] for b in bounds if b is not None)==values[i]
        if not (winning_cells(own,mask)&possible(mask)):hard+=1
        if s['tempo_columns']:
            tempos+=1
            optimal=np.flatnonzero(data['policies'][i]>0)
            if values[i]>=0 and all(s['spaces'][c]==1 for c in optimal):
                critical+=1
                if len(examples)<8:
                    examples.append({'history_1based':str(data['histories'][i]),'spaces':s['spaces'],
                                     'optimal_columns_1based':(optimal+1).tolist(),'value':int(values[i]),'scores':scores[i].tolist()})
        if solver.last['dead_tempo_columns'] and dead_cases<300:
            dead_cases+=1;other=plain.analyze(own,mask,seconds=.70,merge_dead=False)
            for a,b in zip(bounds,other):
                if a is not None and b is not None:assert max(a[0],b[0])<=min(a[1],b[1])
            if solved and (plain.last['complete'] or plain.last['proved_win']):
                assert max(b[0] for b in other if b is not None)==values[i]
            comparisons+=1
    report={'positions':checked,'optimal_result_proved':proved,'non_immediate_win_positions':hard,
            'positions_with_top_spare_moves':tempos,'only_optimal_actions_are_spare_top_moves':critical,
            'dead_tempo_merge_crosschecks':comparisons,'max_cpu_seconds':max_cpu,'examples':examples,'passed':True}
    Path('runs/constrained_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':run()
