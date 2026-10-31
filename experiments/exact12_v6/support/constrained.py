"""Exact WDL for three main columns, retaining every one-cell tempo column.

Only proved outcome bounds may constrain the network. A CPU deadline returns
unknown bounds. Dead top cells may be merged only when every four-cell line
through the cell already contains both colours, so neither player can use it.
This preserves the number of spare turns and never treats a live top as a pass.
"""
import time
from support.tactics import BOARD,BOTTOM,COLUMNS,ORDER,winning_cells,popcount


LINES=[]
CELL_LINES={}
for c in range(7):
    for r in range(6):
        for dc,dr in [(1,0),(0,1),(1,1),(1,-1)]:
            if 0<=c+3*dc<7 and 0<=r+3*dr<6:
                line=sum(1<<((c+k*dc)*7+r+k*dr) for k in range(4))
                LINES.append(line)
                for k in range(4):CELL_LINES.setdefault(1<<((c+k*dc)*7+r+k*dr),[]).append(line)


def shape(mask):
    spaces=[6-popcount((mask>>(7*c))&63) for c in range(7)]
    return {'spaces':spaces,'main_columns':sum(x>=2 for x in spaces),
            'tempo_columns':sum(x==1 for x in spaces),'empty':sum(spaces)}


def applicable(mask):
    s=shape(mask)
    return s['main_columns']<=3 or s['empty']<=12


class Deadline(Exception):pass


class ConstrainedSolver:
    def __init__(self,capacity=350000):
        self.capacity=capacity;self.table={};self.nodes=0
        self.deadline=float('inf');self.wall_deadline=float('inf')
        self.dead=0;self.last={}

    def configure(self,own,mask):
        dead=0;opponent=own^mask
        for c in range(7):
            if (mask>>(7*c))&63 !=31:continue
            bit=1<<(7*c+5)
            if all(line&own and line&opponent for line in CELL_LINES.get(bit,[])):
                dead|=bit
        # Proof bounds keyed with a different set of compressed cells cannot mix.
        if dead!=self.dead:self.table.clear()
        self.dead=dead
        if len(self.table)>self.capacity:self.table.clear()

    def key(self,own,mask,tempos):
        return ((own&~self.dead)+(mask&~self.dead))|(tempos<<49)

    def negamax(self,own,mask,remaining,alpha,beta,tempos):
        self.nodes+=1
        if self.nodes&127==0:
            if time.process_time()>=self.deadline or time.perf_counter()>=self.wall_deadline:raise Deadline()
        if remaining==0:return 0
        moves=(mask+BOTTOM)&BOARD
        if winning_cells(own,mask)&moves:return 1
        opponent=own^mask;threats=winning_cells(opponent,mask);forced=moves&threats
        if forced:
            if forced&(forced-1):return -1
            moves=forced
        moves&=~(threats>>1)
        if not moves:return -1
        if remaining<=2:return 0
        key=self.key(own,mask,tempos);old=self.table.get(key,7)
        bound=old&7
        lo=-1 if bound&1 else 0 if bound&2 else 1
        hi=1 if bound&4 else 0 if bound&2 else -1
        if lo==hi:return lo
        if lo>=beta:return lo
        if hi<=alpha:return hi
        alpha=max(alpha,lo);beta=min(beta,hi)
        a0,b0=alpha,beta;preferred=(old>>3)-1
        inert=moves&self.dead
        if inert:moves=(moves&~self.dead)|(inert&-inert)
        choices=[]
        for c in ORDER:
            bit=moves&COLUMNS[c]
            if bit:
                # Threat count orders work; it does not establish an outcome.
                priority=(100 if c==preferred else 0)+popcount(winning_cells(own|bit,mask|bit))
                choices.append((priority,c,bit))
        if len(choices)>1:choices.sort(reverse=True)
        best=-1;best_col=choices[0][1]
        for _,c,bit in choices:
            value=-self.negamax(opponent,mask|bit,remaining-1,-beta,-alpha,tempos-int(bool(bit&self.dead)))
            if value>best:best,best_col=value,c
            alpha=max(alpha,value)
            if alpha>=beta:break
        if best<=a0:new=1 if best==-1 else 3 if best==0 else 7
        elif best>=b0:new=4 if best==1 else 6 if best==0 else 7
        else:new=1<<(best+1)
        merged=bound&new
        if not merged:raise AssertionError('Contradictory exact bounds')
        self.table[key]=merged|((best_col+1)<<3)
        return best

    def analyze(self,own,mask,seconds=.70,preferred=None,known=None,minimum=None,merge_dead=True):
        begin=time.process_time();self.configure(own,mask)
        if not merge_dead:self.dead=0;self.table.clear()
        self.nodes=0;self.deadline=begin+seconds;self.wall_deadline=time.perf_counter()+1.25
        s=shape(mask);remaining=s['empty'];tempos=popcount(self.dead&~mask)
        moves=(mask+BOTTOM)&BOARD;opponent=own^mask
        bounds=[None]*7;aliases={};dead_columns=[]
        for c in range(7):
            bit=moves&COLUMNS[c]
            if not bit:continue
            if bit&self.dead:dead_columns.append(c)
            if winning_cells(own,mask)&bit:bounds[c]=(1,1)
            elif remaining==1:bounds[c]=(0,0)
            elif winning_cells(opponent,mask|bit)&((mask|bit)+BOTTOM)&BOARD:bounds[c]=(-1,-1)
            else:bounds[c]=(-1,1)
        if dead_columns:
            for c in dead_columns[1:]:aliases[c]=dead_columns[0]
        order=[c for c in ORDER if bounds[c] is not None and c not in aliases]
        if preferred in order:order.remove(preferred);order.insert(0,preferred)
        complete=False
        # A proved winning child is sufficient for an optimal action. Draws
        # require excluding all winning alternatives before being called best.
        if not any(b==(1,1) for b in bounds):
            for sweep in range(8):
                for c in order:
                    if bounds[c][0]==bounds[c][1]:continue
                    if time.process_time()>=begin+seconds:break
                    bit=moves&COLUMNS[c]
                    self.deadline=min(begin+seconds,time.process_time()+.08*(sweep+1))
                    try:
                        value=-self.negamax(opponent,mask|bit,remaining-1,-1,1,tempos-int(bool(bit&self.dead)))
                        bounds[c]=(value,value)
                    except Deadline:pass
                    for alias,representative in aliases.items():bounds[alias]=bounds[representative]
                    if bounds[c]==(1,1):break
                    target=known if known in (0,1) else minimum
                    if target in (0,1):
                        candidates=[d for d,b in enumerate(bounds) if b is not None and b[1]>=target]
                        if len(candidates)==1:
                            d=candidates[0];bounds[d]=(target,target if known is not None else bounds[d][1])
                if any(b==(1,1) for b in bounds):break
                if all(b is None or b[0]==b[1] for b in bounds):complete=True;break
                if time.process_time()>=begin+seconds:break
        else:complete=True
        for alias,representative in aliases.items():bounds[alias]=bounds[representative]
        complete=all(b is None or b[0]==b[1] for b in bounds)
        self.last={**s,'nodes':self.nodes,'dead_tempo_columns':popcount(self.dead),'complete':complete,
                   'proved_win':any(b==(1,1) for b in bounds),'bounds':bounds,
                   'cpu_seconds':time.process_time()-begin,'cached_states':len(self.table)}
        return bounds


SOLVER=ConstrainedSolver()
