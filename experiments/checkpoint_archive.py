"""Lossless checkpoint deduplication; restores the original file byte for byte.

Large tensor payloads in torch ZIP files are shared in .checkpoint_store.
Headers, pickle metadata, small tensors and ZIP trailers remain in each .pack.
No pickle is executed. An original is removed only after reconstruction SHA-256
matches it and its size/mtime are unchanged. Delivered upload-* folders are out
of scope. Usage: pack [--apply], restore checkpoint.pt.pack [--out path].
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile
import zipfile
import zlib

ROOT=Path(__file__).resolve().parent
STORE=ROOT/'.checkpoint_store'
KEEP={ROOT/'exact12_v5/runs/v5/model.pt',ROOT/'exact12_v6/runs/refined/model.pt',
      ROOT/'exact12_v7/runs/release_v7/model.pt'}


def digest(data):return hashlib.sha256(data).hexdigest()


def disk_usage():return int(subprocess.check_output(['du','-s','-B1',str(ROOT)],text=True).split()[0])


def blob_data(store,sha,cache):
    if not re.fullmatch('[0-9a-f]{64}',sha):raise ValueError('Invalid blob key')
    if sha not in cache:
        raw=zlib.decompress((store/(sha+'.zlib')).read_bytes())
        if digest(raw)!=sha:raise ValueError('Corrupt shared tensor '+sha)
        cache[sha]=raw
    return cache[sha]


def read_pack(path,cache):
    with zipfile.ZipFile(path) as z:
        manifest=json.loads(z.read('manifest.json'));literal=z.read('literals.bin')
    if manifest['format']!='torch-zip-lossless-v1':raise ValueError('Unsupported archive')
    store=(path.parent/manifest['store']).resolve()
    if os.path.commonpath([str(store),str(ROOT)])!=str(ROOT):raise ValueError('Store must be inside experiments')
    def chunks():
        for part in manifest['parts']:
            data=blob_data(store,part['blob'],cache) if 'blob' in part else literal[part['offset']:part['offset']+part['size']]
            if len(data)!=part['size']:raise ValueError('Truncated archive segment')
            yield data
    return manifest,chunks


def verify_pack(path,cache):
    manifest,chunks=read_pack(path,cache);h=hashlib.sha256();size=0
    for data in chunks():h.update(data);size+=len(data)
    if h.hexdigest()!=manifest['sha256'] or size!=manifest['bytes']:
        raise ValueError('Reconstruction mismatch: '+str(path))
    return manifest


def pack_file(path,cache):
    if path.is_symlink() or ROOT not in path.resolve().parents:raise ValueError('Refusing an external file')
    stat=path.stat();data=path.read_bytes();sha=digest(data);packed=path.with_name(path.name+'.pack')
    if packed.exists():
        manifest=verify_pack(packed,cache)
        if manifest['sha256']!=sha:raise ValueError('A different archive already exists: '+str(packed))
    else:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            spans=[]
            for info in z.infolist():
                if info.compress_size<1_000_000:continue
                name_size,extra_size=struct.unpack_from('<HH',data,info.header_offset+26)
                begin=info.header_offset+30+name_size+extra_size;end=begin+info.compress_size
                assert 0<=begin<=end<=len(data)
                spans.append((begin,end))
        literal=bytearray();parts=[];cursor=0
        def add_literal(raw):
            if raw:
                parts.append({'offset':len(literal),'size':len(raw)});literal.extend(raw)
        for begin,end in sorted(spans):
            assert begin>=cursor
            add_literal(data[cursor:begin]);raw=data[begin:end];key=digest(raw)
            destination=STORE/(key+'.zlib')
            if not destination.exists():
                encoded=zlib.compress(raw,level=3)
                assert digest(zlib.decompress(encoded))==key
                with tempfile.NamedTemporaryFile(dir=STORE,prefix='.blob-',delete=False) as f:
                    f.write(encoded);temporary=Path(f.name)
                temporary.replace(destination)
            if key not in cache:
                persisted=zlib.decompress(destination.read_bytes())
                if digest(persisted)!=key:raise ValueError('Stored tensor checksum failed')
                cache[key]=persisted
            parts.append({'blob':key,'size':len(raw)});cursor=end
        add_literal(data[cursor:])
        manifest={'format':'torch-zip-lossless-v1','source':str(path.relative_to(ROOT)),
                  'sha256':sha,'bytes':len(data),'mode':stat.st_mode&0o777,
                  'mtime_ns':stat.st_mtime_ns,'store':os.path.relpath(STORE,path.parent),'parts':parts}
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.checkpoint-',delete=False) as f:temporary=Path(f.name)
        try:
            with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
                z.writestr('manifest.json',json.dumps(manifest,separators=(',',':')))
                z.writestr('literals.bin',literal)
            # Verification must resolve the blob store relative to the same parent.
            verify_pack(temporary,cache)
            temporary.replace(packed)
        finally:
            if temporary.exists():temporary.unlink()
    now=path.stat()
    if now.st_size!=stat.st_size or now.st_mtime_ns!=stat.st_mtime_ns:
        raise RuntimeError('Checkpoint changed while being packed; original retained: '+str(path))
    verify_pack(packed,cache)
    path.unlink()
    return {'original':str(path.relative_to(ROOT)),'archive':str(packed.relative_to(ROOT)),
            'original_bytes':stat.st_size,'archive_bytes':packed.stat().st_size,'sha256':sha}


def restore(path,out):
    path=path.resolve();manifest,chunks=read_pack(path,{})
    if out.exists():
        if digest(out.read_bytes())==manifest['sha256']:return manifest
        raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out.parent,prefix='.restore-',delete=False) as f:
        temporary=Path(f.name);h=hashlib.sha256();size=0
        for raw in chunks():f.write(raw);h.update(raw);size+=len(raw)
    try:
        if h.hexdigest()!=manifest['sha256'] or size!=manifest['bytes']:raise ValueError('Restoration checksum failed')
        os.chmod(temporary,manifest['mode']);temporary.replace(out)
        os.utime(out,ns=(manifest['mtime_ns'],manifest['mtime_ns']))
    finally:
        if temporary.exists():temporary.unlink()
    return manifest


def pack_all(apply):
    before=disk_usage();paths=[p for p in ROOT.rglob('*.pt') if p.is_file() and not p.is_symlink()
                             and p.stat().st_size>=2_000_000 and p not in KEEP]
    if not apply:
        print(json.dumps({'candidates':len(paths),'original_bytes':sum(p.stat().st_size for p in paths),
                          'kept_full_checkpoints':[str(p.relative_to(ROOT)) for p in sorted(KEEP)]},indent=2));return
    STORE.mkdir(exist_ok=True);cache={};records=[]
    for i,path in enumerate(sorted(paths)):
        records.append(pack_file(path,cache))
        if (i+1)%10==0:print(json.dumps({'packed':i+1,'total':len(paths)}),flush=True)
    cache.clear()
    # Exercise full file restoration independently from the in-memory blob cache.
    if records:
        with tempfile.TemporaryDirectory(prefix='homework4-checkpoint-audit-') as folder:
            test=Path(folder)/'restored.pt';restore(ROOT/records[0]['archive'],test)
            assert digest(test.read_bytes())==records[0]['sha256']
    removed=[]
    for p in sorted(ROOT.rglob('__pycache__')):
        if p.is_dir():removed.append({'path':str(p.relative_to(ROOT)),'type':'regenerable_python_cache'});shutil.rmtree(p)
    # Keep the sorted canonical tables and the canonical public book.
    duplicate=ROOT/'cnn_endgame_v4/data/public/downloads/7x6.book'
    canonical=ROOT/'cnn_endgame_v4/data/public/7x6.book'
    if duplicate.exists() and canonical.exists() and digest(duplicate.read_bytes())==digest(canonical.read_bytes()):
        removed.append({'path':str(duplicate.relative_to(ROOT)),'type':'byte_identical_download','canonical':str(canonical.relative_to(ROOT))});duplicate.unlink()
    for raw,sorted_file in [(ROOT/'cnn_endgame_v4/data/public/movebook12_raw.bin',ROOT/'cnn_endgame_v4/data/public/movebook12.bin'),
                            (ROOT/'v4_release/runs/movebook_first12_wdl.raw.bin',ROOT/'v4_release/runs/movebook_first12_wdl.sorted.bin')]:
        if raw.exists() and sorted_file.exists():
            assert raw.stat().st_size==sorted_file.stat().st_size
            removed.append({'path':str(raw.relative_to(ROOT)),'type':'unsorted_intermediate','canonical':str(sorted_file.relative_to(ROOT))});raw.unlink()
    after=disk_usage();report={'before_bytes':before,'after_bytes':after,'freed_bytes':before-after,
        'packed_checkpoints':records,'removed_regenerable_files':removed,'byte_exact_restore_test_passed':True,
        'kept_full_checkpoints':[str(p.relative_to(ROOT)) for p in sorted(KEEP)],
        'delivered_upload_folders_untouched':True,'shared_store':str(STORE.relative_to(ROOT))}
    (ROOT/'cleanup_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['packed_checkpoints','removed_regenerable_files']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('pack');a.add_argument('--apply',action='store_true')
    a=sub.add_parser('restore');a.add_argument('archive',type=Path);a.add_argument('--out',type=Path)
    args=p.parse_args()
    if args.command=='pack':pack_all(args.apply)
    else:
        out=args.out or args.archive.with_name(args.archive.name.removesuffix('.pack') if hasattr(str,'removesuffix') else args.archive.name[:-5])
        result=restore(args.archive,out);print(json.dumps({'restored':str(out),'sha256':result['sha256'],'bytes':result['bytes']},indent=2))
