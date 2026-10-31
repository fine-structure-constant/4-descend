"""Load the actual exported strategy in a clean directory and check its resources."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import shutil
import sys
import tempfile
import time
import zipfile


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('package');a=p.parse_args()
    package=Path(a.package).resolve();source=Path(__file__).resolve().parent
    assert set(p.name for p in package.iterdir())=={'code.zip','model.pt'}
    assert (package/'code.zip').stat().st_size<5000000
    assert (package/'model.pt').stat().st_size<100000000
    records=[]
    with tempfile.TemporaryDirectory(prefix='exact12-v5-package-') as temporary:
        root=Path(temporary)
        with zipfile.ZipFile(package/'code.zip') as z:
            assert z.testzip() is None
            assert all(Path(name).suffix in {'.py','.md','.txt'} for name in z.namelist())
            z.extractall(root)
        (root/'models').mkdir();shutil.copy2(package/'model.pt',root/'models/model.pt')
        sys.path.insert(0,str(root));start=time.process_time();wall=time.perf_counter()
        from strategy import Bot
        bot=Bot({'seed':2026103113})
        init_cpu=time.process_time()-start;init_wall=time.perf_counter()-wall
        assert init_cpu<15 and init_wall<30
        from support.game import new_board,play_move,legal_moves
        prefix=[1,5,3,2,1,3,3,3,1,1,5,3,3,4]
        for history in [[],prefix[:11],prefix[:12],prefix[:13],prefix]:
            board=new_board();player=1
            for c in history:board=play_move(board,c,player);player=-player
            begin=time.process_time();wall=time.perf_counter()
            action=bot.act({'game':'connect4','board':board.tolist(),'player':player,'legal_actions':legal_moves(board)})
            cpu=time.process_time()-begin;elapsed=time.perf_counter()-wall
            assert action in legal_moves(board) and cpu<.95 and elapsed<2
            records.append({'occupied':len(history),'action':action,'cpu_seconds':cpu,'wall_seconds':elapsed})
    report={'files':{name:{'bytes':(package/name).stat().st_size,'sha256':hashlib.sha256((package/name).read_bytes()).hexdigest()}
                     for name in ['code.zip','model.pt']},'init_cpu_seconds':init_cpu,'init_wall_seconds':init_wall,
            'peak_memory_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'checks':records,'passed':True}
    (source/'runs/package_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
