"""Convert the CC-BY-4.0 UCI/Tromp dataset into training data and an exact book.

Tromp, J. (1995). Connect-4 [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C59P43 ; https://archive.ics.uci.edu/dataset/26/connect+4
Only exact retrograde conclusions are retained; missing forced positions remain
unknown. The generated book is an offline search aid, not a downloaded network.
"""
import base64
from pathlib import Path
import struct
import zlib
import numpy as np
from support.tactics import possible, COLUMNS, alignment, safe_moves, ORDER


def mirror(bits):
    return sum(((bits >> (7*c)) & 127) << (7*(6-c)) for c in range(7))


def canonical(own,mask):
    key=own+mask
    other=mirror(key)
    return (other,mirror(own),mirror(mask)) if other<key else (key,own,mask)


def main():
    book={};states=[];values=[]
    for line in Path('data/public/connect-4.data').read_text().splitlines():
        cells=line.split(',');b=np.array([{'b':0,'x':1,'o':-1}[v] for v in cells[:42]],dtype=np.int8).reshape(7,6).T[::-1].copy()
        own=mask=0
        for c in range(7):
            for r in range(6):
                bit=1<<(7*c+5-r)
                if b[r,c]:mask|=bit
                if b[r,c]==1:own|=bit
        assert np.count_nonzero(b)==8
        key,_,_=canonical(own,mask)
        z={'win':1,'draw':0,'loss':-1}[cells[42]]
        assert key not in book or book[key]==z
        book[key]=z
        states.append(np.stack([b==1,b==-1,b==0]).astype(np.uint8));values.append(z)
    Path('data').mkdir(exist_ok=True)
    np.savez_compressed('data/public_opening.npz',states=np.asarray(states),values=np.asarray(values,dtype=np.float32))
    levels=[{0:(0,0)}]
    for ply in range(8):
        following={}
        for own,mask in levels[-1].values():
            moves=possible(mask)
            for c in ORDER:
                bit=moves&COLUMNS[c]
                if not bit or alignment(own|bit):continue
                k,p,m=canonical(own^mask,mask|bit)
                following[k]=(p,m)
        levels.append(following)
        print({'ply':ply+1,'states':len(following)},flush=True)
    bounds={k:(v,v) for k,v in book.items()}
    for ply in range(8,-1,-1):
        for key,(own,mask) in levels[ply].items():
            if key in bounds:continue
            moves,status=safe_moves(own,mask)
            if status:
                bounds[key]=(status,status)
                continue
            if ply==8:
                bounds[key]=(-1,1);continue
            lower=upper=-1
            for c in ORDER:
                bit=moves&COLUMNS[c]
                if not bit:continue
                child_key,_,_=canonical(own^mask,mask|bit)
                lo,hi=bounds.get(child_key,(-1,1))
                lower=max(lower,-hi);upper=max(upper,-lo)
            bounds[key]=(lower,upper)
    exact={k:lo for k,(lo,hi) in bounds.items() if lo==hi}
    packed=b''.join(struct.pack('<Qb',key,value) for key,value in sorted(exact.items()))
    encoded=base64.b85encode(zlib.compress(packed,9)).decode()
    text='''"""Offline exact opening outcomes, CC BY 4.0.
Derived from Tromp, J. (1995), Connect-4, UCI Machine Learning Repository.
https://doi.org/10.24432/C59P43
https://archive.ics.uci.edu/dataset/26/connect+4
Adaptations: bitboard encoding, mirror canonicalization, exact retrograde backup.
Unknown positions are omitted. No external model weights are used.
"""
import base64
import struct
import zlib
from functools import lru_cache

_DATA = (
'''
    for i in range(0,len(encoded),100):text+='    '+repr(encoded[i:i+100])+'\n'
    text+=''' )

@lru_cache(maxsize=1)
def _table():
    raw=zlib.decompress(base64.b85decode(_DATA))
    return dict(struct.iter_unpack('<Qb',raw))

def lookup(own,mask):
    key=own+mask
    mirrored=sum(((key>>(7*c))&127)<<(7*(6-c)) for c in range(7))
    return _table().get(min(key,mirrored))
'''
    Path('support/opening.py').write_text(text)
    print({'raw_rows':len(states),'exact_positions':len(exact),'root_bounds':bounds[0],
           'encoded_bytes':len(encoded),'unknown':sum(lo!=hi for lo,hi in bounds.values())},flush=True)


if __name__=='__main__':main()
