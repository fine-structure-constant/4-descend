"""Trace an avoidable loss back to the first wrong move and prove nearby replies."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import random
import numpy as np
from prepare_data import worker_init,label
from support.game import new_board,play_move,game_result,legal_moves


def main():
    rng=random.Random(2026103120);jobs={}
    root=Path(__file__).resolve().parent
    histories=[]
    for path in (root.parent/'exact12_v5/runs/flash_probe').glob('v5_*_live.json'):
        histories.append([f['action'] for f in json.loads(path.read_text())['frames'][1:]])
    for path in [root.parent/'exact12_v5/runs/strong_check.json']:
        histories.extend(r['history'] for r in json.loads(path.read_text())['records'])
    for history in histories:
        b=new_board();player=1
        for i,c in enumerate(history):
            if 12<=i<=24:
                prefix=history[:i];jobs[tuple(prefix)]=(prefix,1.,'refutation')
                # Explore a different opponent reply, including refusing to
                # supply a threatened cell. Outcomes, not shapes, set penalties.
                for alternative in rng.sample(legal_moves(b),min(2,len(legal_moves(b)))):
                    child=play_move(b,alternative,player)
                    if game_result(child)[0]:continue
                    for reply in rng.sample(legal_moves(child),min(2,len(legal_moves(child)))):
                        after=play_move(child,reply,-player)
                        if game_result(after)[0]:continue
                        fork=prefix+[alternative,reply];jobs[tuple(fork)]=(fork,1.,'refutation')
            b=play_move(b,c,player)
            if game_result(b)[0]:break
            player=-player
    rows=[]
    with ProcessPoolExecutor(max_workers=8,initializer=worker_init) as pool:
        for r in pool.map(label,jobs.values(),chunksize=8):
            if r is not None:rows.append(r)
    data={'states':np.stack([r[0] for r in rows]),'policies':np.stack([r[1] for r in rows]),
          'values':np.array([r[2] for r in rows],np.float32),'scores':np.stack([r[3] for r in rows]),
          'occupied':np.array([r[4] for r in rows]),'source':np.array([r[5] for r in rows]),
          'validation':np.array([r[6] for r in rows],bool)}
    np.savez_compressed(root/'data/refutations.npz',**data)
    archive=np.load(root/'data/proofs.npz');base={k:archive[k] for k in data}
    seen=set();merged={k:[] for k in data}
    for d in [data,base]:
        for i,s in enumerate(d['states']):
            key=min(s.tobytes(),s[:,:,::-1].tobytes())
            if key in seen:continue
            seen.add(key)
            for k in merged:merged[k].append(d[k][i])
    np.savez_compressed(root/'data/refinement.npz',**{k:np.asarray(v) for k,v in merged.items()})
    report={'refutation_positions':len(rows),'attempted':len(jobs),'merged_positions':len(seen),
            'purpose':'Penalize avoidable WDL downgrades before the final forced loss, using exact teacher values.'}
    (root/'runs/refutations.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
