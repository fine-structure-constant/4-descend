"""Resume exact12, distill proof labels, then reanalyse mixed self-play.

Lookup buffers remain frozen. Native code generates offline labels only.
Example: python train.py --model ../../upload-v4-exact12/model.pt --out runs/v5
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from model import Network
from mcts import search
from support.game import new_board,play_move,game_result,result_value,encode
from support.runtime import set_seed,save_model,append_log,load_model,sample_action
ROOT=Path(__file__).resolve().parent


def self_play(model,simulations):
    """Homework-compatible interface; exploration respects proven constraints."""
    board=new_board();player=1;history=[]
    while True:
        counts=search(board,player,model,simulations,add_noise=True)
        pi=counts/counts.sum();action=sample_action(pi)
        history.append((encode(board,player),pi,player))
        board=play_move(board,action,player);done,winner=game_result(board)
        if done:break
        player=-player
    return [(s,pi,result_value(winner,p)) for s,pi,p in history]


def compute_loss(model,states,target_pi,target_z):
    """Original policy/value objective retained for gradient checks."""
    logits,values=model(states)
    return -(target_pi*torch.log_softmax(logits,1)).sum(1).mean()+(values-target_z).square().mean()


def canonical(state):
    return min(state.tobytes(),state[:,:,::-1].tobytes())


def load_baseline(path):
    model=Network();state=torch.load(path,map_location='cpu',weights_only=True)
    result=model.load_state_dict(state,strict=False)
    assert not result.unexpected_keys
    assert set(result.missing_keys)<={'deepbook','action_values.weight','action_values.bias'}
    if 'deepbook' not in state:
        raw=(ROOT.parent/'cnn_endgame_v4/data/public/7x6.book').read_bytes()
        assert len(raw)==33554524
        model.deepbook.copy_(torch.from_numpy(np.frombuffer(raw,dtype=np.uint8).copy()))
    from support.movebook import configure
    assert configure(model),'Validated exact12 movebook is required'
    return model.eval()


def metrics(model,data,ids):
    correct=0;n=0;square=0.;winning=0;blunders=0
    model.eval()
    with torch.no_grad():
        for begin in range(0,len(ids),512):
            ix=ids[begin:begin+512];s=torch.tensor(data['states'][ix],dtype=torch.float32)
            logits,v=model(s);legal=torch.tensor(data['states'][ix,2,0,:]>0)
            action=logits.masked_fill(~legal,-1e9).argmax(1).numpy()
            good=data['policies'][ix,action]>0;z=data['values'][ix]
            correct+=int(good.sum());n+=len(ix);square+=float((v-torch.tensor(z)).square().sum())
            winning+=int((z==1).sum());blunders+=int(((z==1)&~good).sum())
    return {'policy_optimal_rate':correct/max(1,n),'value_mse':square/max(1,n),
            'winning_position_blunder_rate':blunders/max(1,winning),'positions':n}


def objective(model,states,policies,values,scores,exact):
    logits,v,q=model.training_outputs(states);legal=states[:,2,0,:]>0
    logp=torch.log_softmax(logits.masked_fill(~legal,-1e9),1)
    set_loss=-torch.logsumexp(logp.masked_fill(~(policies>0),-1e9),1)
    distribution_loss=-(policies*logp).sum(1)
    policy=torch.where(exact,.85*set_loss+.15*distribution_loss,distribution_loss).mean()
    known=(scores>=-1)&(scores<=1)&legal
    qloss=((q-scores.clamp(-1,1)).square()*known).sum()/known.sum().clamp_min(1)
    return policy+(v-values).square().mean()+.35*qloss


def play_job(job):
    checkpoint,index,seed,simulations=job;set_seed(seed+index)
    model=load_model(checkpoint)
    opponent=load_baseline(ROOT.parents[1]/'upload-v4-exact12/model.pt') if index%3==0 else model
    board=new_board();player=1;history=[];samples=[]
    while True:
        active=model if (player==1)==(index%2==0) else opponent
        counts=search(board,player,active,simulations=simulations,add_noise=True)
        pi=counts/counts.sum()
        if len(history)<18:
            pi=np.sqrt(pi);pi/=pi.sum();action=int(np.random.choice(7,p=pi))
        else:action=int(np.argmax(pi))
        samples.append((encode(board,player).numpy().astype(np.uint8),pi.astype(np.float32),player,list(history)))
        board=play_move(board,action,player);history.append(action);done,winner=game_result(board)
        if done:break
        player=-player
    return samples,winner,history


def build_replay(checkpoint,args,round_index,validation_keys):
    import prepare_data
    states=[];policies=[];values=[];scores=[];exact=[];games=[];jobs=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for samples,winner,history in pool.map(play_job,[(str(checkpoint),i,args.seed+round_index*10000,args.simulations) for i in range(args.games)]):
            games.append({'history':history,'winner':winner})
            for state,pi,player,prefix in samples:
                if canonical(state) in validation_keys:continue
                states.append(state);policies.append(pi);values.append(result_value(winner,player))
                scores.append(np.full(7,-999,np.int16));exact.append(False)
                if 12<=len(prefix)<=30:jobs.append((prefix,args.teacher_seconds,'selfplay'))
    proof_rows=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=prepare_data.worker_init) as pool:
        for result in pool.map(prepare_data.label,jobs,chunksize=8):
            if result is not None and result[7] not in validation_keys:proof_rows.append(result)
    replacements={r[7]:r for r in proof_rows}
    for i,state in enumerate(states):
        r=replacements.get(canonical(state))
        if r is not None:
            mirrored=not np.array_equal(state,r[0])
            policies[i]=r[1][::-1].copy() if mirrored else r[1]
            values[i]=r[2];scores[i]=r[3][::-1].copy() if mirrored else r[3];exact[i]=True
    return {'states':np.stack(states),'policies':np.stack(policies),'values':np.array(values,np.float32),
            'scores':np.stack(scores),'exact':np.array(exact,bool)},games,len(proof_rows)


def train(args):
    set_seed(args.seed);out=ROOT/args.out;out.mkdir(parents=True,exist_ok=False)
    model=load_baseline(args.model)
    buffer_hashes={name:hashlib.sha256(value.numpy().tobytes()).hexdigest() for name,value in model.named_buffers()}
    raw=np.load(ROOT/args.data);data={name:raw[name] for name in raw.files};data['exact']=np.ones(len(data['states']),bool)
    val=np.flatnonzero(data['validation']);train_ids=np.flatnonzero(~data['validation'])
    validation_keys={canonical(data['states'][i]) for i in val}
    assert not any(canonical(data['states'][i]) in validation_keys for i in train_ids)
    groups=[train_ids[(data['occupied'][train_ids]>=12)&(data['occupied'][train_ids]<=25)],
            train_ids[data['occupied'][train_ids]>=26],train_ids[np.isin(data['source'][train_ids],['flash','replay'])],
            train_ids[data['occupied'][train_ids]<=12]]
    groups=[g if len(g) else train_ids for g in groups]
    tensors={k:torch.tensor(data[k],dtype=torch.bool if k=='exact' else torch.float32)
             for k in ['states','policies','values','scores','exact']}
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=2e-4)
    baseline=metrics(model,data,val);best=float('inf');started=time.time();global_step=0
    save_model(model,out/'initial.pt')
    (out/'config.json').write_text(json.dumps({**vars(args),'baseline_validation':baseline,'lookup_sha256':buffer_hashes,
                                             'train_positions':len(train_ids),'validation_positions':len(val)},indent=2))
    print(json.dumps({'stage':'initial','validation':baseline}),flush=True);replay=None
    for stage in range(args.rounds+1):
        if stage:
            checkpoint=out/('generation_%d.pt'%stage);save_model(model,checkpoint)
            replay,games,proofs=build_replay(checkpoint,args,stage,validation_keys)
            np.savez_compressed(out/('replay_%d.npz'%stage),**replay)
            (out/('selfplay_%d.json'%stage)).write_text(json.dumps({'records':games,'reanalysed_positions':proofs},indent=2))
            print(json.dumps({'stage':'selfplay','round':stage,'games':len(games),'positions':len(replay['states']),'proved':proofs}),flush=True)
        steps=args.steps if not stage else args.finetune_steps
        for step in range(1,steps+1):
            global_step+=1;sizes=[args.batch//2,args.batch//4,args.batch//5];sizes.append(args.batch-sum(sizes))
            ids=np.concatenate([np.random.choice(g,n) for g,n in zip(groups,sizes)])
            batch={k:v[ids].clone() for k,v in tensors.items()}
            if replay is not None:
                n=args.batch//4;ix=np.random.randint(len(replay['states']),size=n)
                for k in batch:batch[k][:n]=torch.tensor(replay[k][ix],dtype=batch[k].dtype)
            flip=torch.rand(args.batch)<.5;batch['states'][flip]=batch['states'][flip].flip(-1)
            for k in ['policies','scores']:batch[k][flip]=batch[k][flip].flip(-1)
            model.train();loss=objective(model,**batch)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            if step%args.validate_every==0 or step==steps:
                measured=metrics(model,data,val);selection=(1-measured['policy_optimal_rate'])+.25*measured['value_mse']
                row={'stage':'proofs' if not stage else 'mixed_selfplay','round':stage,'step':global_step,
                     'loss':float(loss),'validation':measured,'seconds':round(time.time()-started,1)}
                append_log(out/'log.jsonl',row);print(json.dumps(row),flush=True)
                if selection<best:
                    best=selection;save_model(model,out/'best.pt');(out/'best_metrics.json').write_text(json.dumps(row,indent=2))
        for group in optimizer.param_groups:group['lr']*=.6
    for name,value in model.named_buffers():
        assert hashlib.sha256(value.numpy().tobytes()).hexdigest()==buffer_hashes[name],name
    save_model(model,out/'last.pt');best_model=load_model(out/'best.pt');save_model(best_model,out/'model.pt')
    print(json.dumps({'stage':'complete','baseline':baseline,'best':json.loads((out/'best_metrics.json').read_text()),
                      'lookup_buffers_unchanged':True}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='../../upload-v4-exact12/model.pt');p.add_argument('--out',default='runs/v5')
    p.add_argument('--data',default='data/proofs.npz');p.add_argument('--steps',type=int,default=5000)
    p.add_argument('--finetune-steps',type=int,default=1000);p.add_argument('--rounds',type=int,default=2)
    p.add_argument('--games',type=int,default=24);p.add_argument('--workers',type=int,default=6)
    p.add_argument('--simulations',type=int,default=64);p.add_argument('--teacher-seconds',type=float,default=.5)
    p.add_argument('--batch',type=int,default=256);p.add_argument('--lr',type=float,default=.0002)
    p.add_argument('--validate-every',type=int,default=500);p.add_argument('--seed',type=int,default=2026103106)
    train(p.parse_args())
