"""Export Python-only inference, validate exact buffers and enforce decimal limits."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
from mcts import search
from support.game import new_board
from support.runtime import load_model,set_seed
from support.movebook import configure as configure_movebook
from support.pons_book import configure as configure_scores
from support import native


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('model');p.add_argument('--out',required=True)
    a=p.parse_args();out=Path(a.out).resolve();root=Path(__file__).resolve().parent
    if out.exists():p.error('Choose a new directory; existing releases are preserved')
    set_seed(2026103107);model=load_model(a.model)
    assert configure_movebook(model) and configure_scores(model)
    assert sum(parameter.numel() for parameter in model.parameters())<1000000
    assert native.available() and native.warmup()
    assert native.analyze(0, 0, 50) is not None
    native.close()
    counts=search(new_board(),1,model,simulations=128)
    assert counts.shape==(7,) and np.isfinite(counts).all() and counts.sum()==128
    names=['model.py','mcts.py','support/__init__.py','support/bot.py','support/game.py',
           'support/runtime.py','support/tactics.py','support/endgame.py','support/opening.py',
           'support/deep_opening.py','support/movebook.py','support/pons_book.py','support/constrained.py',
           'support/corrections.py','support/midbook.py','support/native.py',
           'support/native_payload.py','REFERENCES.md']
    out.mkdir(parents=True)
    with zipfile.ZipFile(out/'code.zip','x',zipfile.ZIP_DEFLATED) as z:
        z.writestr('strategy.py','from support.bot import Bot\n')
        for name in names:
            assert Path(name).suffix in {'.py','.md','.txt'}
            z.write(root/name,name)
        z.write(root/'tools/AGPL-3.0.txt','AGPL-3.0.txt')
    shutil.copy2(a.model,out/'model.pt')
    sizes={name:(out/name).stat().st_size for name in ['code.zip','model.pt']}
    assert sizes['code.zip']<5000000 and sizes['model.pt']<100000000
    report={'output':str(out),'artifacts':{name:{'bytes':size,'sha256':hashlib.sha256((out/name).read_bytes()).hexdigest()}
            for name,size in sizes.items()},'learned_parameters':sum(p.numel() for p in model.parameters()),
            'code_format':'Python/text archive with a self-contained offline helper; no compiler or network required at inference'}
    (root/'runs/release_manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
