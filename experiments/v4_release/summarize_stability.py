"""Summarize fixed paired matches; bootstrap by opening, not by individual move."""
import argparse,json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('reports',nargs='+');p.add_argument('--out',default='runs/stability_summary.json')
a=p.parse_args();records=[];pairs=[];hashes=None
for path in a.reports:
 d=json.loads(Path(path).read_text())
 if hashes is None:hashes=d['sha256']
 assert hashes==d['sha256'],'Different inference sources or parameters across batches'
 r=d['records'];assert len(r)%2==0,'Incomplete pair'
 records+=r
 for i in range(0,len(r),2):
  assert r[i]['opening']==r[i+1]['opening']
  assert r[i]['side']==1 and r[i+1]['side']==-1
  score=lambda row:1. if row['result']=='win' else .5 if row['result']=='draw' else 0.
  pairs.append((score(r[i])+score(r[i+1]))/2)
values=np.asarray(pairs)
rng=np.random.RandomState(2026103102)
bootstrap=values[rng.randint(len(values),size=(50000,len(values)))].mean(1)
interval=np.quantile(bootstrap,[.025,.975]).tolist()
total={r:sum(x['result']==r for x in records) for r in ['win','draw','loss']}
by_side={str(side):{r:sum(x['side']==side and x['result']==r for x in records) for r in ['win','draw','loss']} for side in [1,-1]}
max_cpu=max(t for row in records for t in row['move_cpu_seconds'])
match_wins=int(np.sum(values>.5));match_draws=int(np.sum(values==.5));match_losses=int(np.sum(values<.5))
passed=(len(records)>=40 and values.mean()>=.70 and match_wins/len(values)>=.70 and interval[0]>.5 and max_cpu<1.)
report={'games':len(records),'opening_groups':len(values),'total':total,'by_side':by_side,
        'score_rate':float(values.mean()),'paired_matches':{'win':match_wins,'draw':match_draws,'loss':match_losses},
        'paired_bootstrap_95_interval':interval,'max_cpu_seconds':max_cpu,
        'criterion':{'minimum_games':40,'minimum_score_rate':.70,'minimum_pair_win_rate':.70,
                     'lower_95_bound_must_exceed':.5,'max_move_cpu_seconds':1.},
        'stable_criterion_passed':bool(passed),'sha256':hashes,'reports':a.reports}
Path(a.out).write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))
