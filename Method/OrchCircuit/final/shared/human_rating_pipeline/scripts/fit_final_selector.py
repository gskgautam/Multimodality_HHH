#!/usr/bin/env python3
"""Fit a deployable OrchCircuit selector for a future external test.
Uses the published CV architecture and its grouped 15% validation recipe.
It does not access or score any external test records and makes no API calls.
"""
import copy, hashlib, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=(next((p for p in Path(__file__).resolve().parents if (p / "README.md").is_file()), Path.cwd()))
EXP=ROOT/'Method/OrchCircuit/final'
PRIVATE=ROOT/'Datasets/Ch3EF/data/ch3ef_full1416/orchcircuit_final_fit_PRIVATE'
DEVICE='cuda:0' if torch.cuda.is_available() else 'cpu'
TOKENS,DIM=64,3584
SEED=20261111
sys.path.insert(0,str(EXP))
import run_selector_cv as cv

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def read_json(path):return json.loads(Path(path).read_text())

def main():
    PRIVATE.mkdir(parents=True,exist_ok=True)
    checkpoint_path=PRIVATE/'orchcircuit_selector.pt'
    summary_path=PRIVATE/'fit_summary.json'
    if checkpoint_path.exists():raise SystemExit(f'Refusing to overwrite existing checkpoint: {checkpoint_path}')
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(SEED);torch.cuda.set_device(0)
    by,orders,dimension,per_metric,wr,ss,truth,info=cv.labels_and_orders()
    n=len(orders);oi={o:i for i,o in enumerate(orders)}
    featdir=EXP/'data_selected_answer'
    manifests=[read_json(featdir/f'features_shard{i}.json') for i in (0,1)]
    dones=[read_json(featdir/f'features_shard{i}.done.json') for i in (0,1)]
    for i in (0,1):
        if manifests[i]['sample_orders']!=orders:raise RuntimeError(f'feature order mismatch on shard {i}')
        if set(dones[i]['completed_models'])!=set(manifests[i]['models']):raise RuntimeError(f'incomplete shard {i}')
    f0=np.memmap(featdir/'features_shard0.f16',dtype=np.float16,mode='r',shape=(n,12,TOKENS,DIM))
    f1=np.memmap(featdir/'features_shard1.f16',dtype=np.float16,mode='r',shape=(n,12,TOKENS,DIM))
    m0={m:i for i,m in enumerate(manifests[0]['models'])};m1={m:i for i,m in enumerate(manifests[1]['models'])}
    def getx(batch_orders):
        ix=np.asarray([oi[o] for o in batch_orders],dtype=np.int64)
        x=np.empty((len(ix),24,TOKENS,DIM),dtype=np.float16)
        for mi,m in enumerate(cv.MODELS):x[:,mi]=f0[ix,m0[m]] if m in m0 else f1[ix,m1[m]]
        return torch.from_numpy(x).to(DEVICE,dtype=torch.float32)
    group_of,image_hashes=cv.make_groups(by,orders)
    train_orders,val_orders=cv.validation_split(orders,group_of,dimension,SEED)
    train_groups={group_of[o] for o in train_orders};val_groups={group_of[o] for o in val_orders}
    if train_groups & val_groups:raise RuntimeError('query/image group leakage in final fit split')
    oi_array={o:oi[o] for o in orders}
    model=cv.OrchCircuitAll24().to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01)
    train_by={d:[o for o in train_orders if dimension[o]==d] for d in cv.DIMS}
    val_by={d:[o for o in val_orders if dimension[o]==d] for d in cv.DIMS}
    def loss_for(batch):
        pred=model(getx(batch));losses=[]
        for d in cv.DIMS:
            ix_np=np.asarray([oi_array[o] for o in batch if dimension[o]==d],dtype=np.int64)
            if not len(ix_np):continue
            mask=torch.as_tensor([dimension[o]==d for o in batch],device=DEVICE)
            if d=='Helpful':
                y=torch.as_tensor(wr[ix_np],device=DEVICE,dtype=torch.long)
                losses.append(F.cross_entropy(pred['wr'][mask].reshape(-1,3),y.reshape(-1)))
            elif d=='Harmless':
                y=torch.as_tensor(ss[ix_np],device=DEVICE,dtype=torch.float32)
                losses.append(F.binary_cross_entropy_with_logits(pred['ss'][mask],y))
            else:
                yt=torch.as_tensor(truth[ix_np],device=DEVICE,dtype=torch.float32)
                yi=torch.as_tensor(info[ix_np],device=DEVICE,dtype=torch.float32)
                losses.append(F.binary_cross_entropy_with_logits(pred['truth'][mask],yt)+F.binary_cross_entropy_with_logits(pred['info'][mask],yi))
        return torch.stack(losses).mean()
    def batches(pool,shuffle):
        queues={d:list(pool[d]) for d in cv.DIMS}
        if shuffle:
            for d in cv.DIMS:random.shuffle(queues[d])
        out=[]
        while any(queues[d] for d in cv.DIMS):
            b=[]
            for d in cv.DIMS:b.extend(queues[d][:2]);queues[d]=queues[d][2:]
            if b:out.append(b)
        return out
    best=float('inf');beststate=None;stale=0;bestepoch=None;started=time.time();log=[]
    for ep in range(1,21):
        model.train();tl=[]
        for batch in batches(train_by,True):
            loss=loss_for(batch);opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();tl.append(float(loss.detach().cpu()))
        model.eval();vl=[]
        with torch.inference_mode():
            for batch in batches(val_by,False):vl.append(float(loss_for(batch).detach().cpu()))
        v=float(np.mean(vl));line={'epoch':ep,'train_loss':float(np.mean(tl)),'validation_loss':v}
        log.append(line);print(json.dumps(line),flush=True)
        if v<best-1e-5:
            best=v;beststate={k:x.detach().cpu().clone() for k,x in model.state_dict().items()};stale=0;bestepoch=ep
        else:
            stale+=1
            if stale>=4:break
    if beststate is None:raise RuntimeError('training did not produce a checkpoint')
    model.load_state_dict(beststate);model.eval()
    val_raw={}
    with torch.inference_mode():
        for batch in batches(val_by,False):
            out=model(getx(batch))
            for j,o in enumerate(batch):val_raw[o]={k:v[j].detach().cpu().numpy() for k,v in out.items()}
    val_wr=[];val_wry=[];val_ss=[];val_ssy=[];val_tr=[];val_try=[];val_in=[];val_iny=[]
    for o,z in val_raw.items():
        i=oi[o]
        if dimension[o]=='Helpful':val_wr.extend(z['wr']);val_wry.extend(wr[i])
        elif dimension[o]=='Harmless':val_ss.extend(z['ss']);val_ssy.extend(ss[i])
        else:
            val_tr.extend(z['truth']);val_try.extend(truth[i]);val_in.extend(z['info']);val_iny.extend(info[i])
    calibration={'wr_temperature':cv.fit_temperature(np.asarray(val_wr),np.asarray(val_wry)),
      'ss':cv.fit_binary(np.asarray(val_ss),np.asarray(val_ssy)),
      'truth':cv.fit_binary(np.asarray(val_tr),np.asarray(val_try)),
      'info':cv.fit_binary(np.asarray(val_in),np.asarray(val_iny))}
    config={'architecture':'OrchCircuitAll24 from run_selector_cv.py','seed':SEED,'train_orders':train_orders,'validation_orders':val_orders,
      'group_ids':{str(o):int(group_of[o]) for o in orders},'image_sha256_by_order':image_hashes,
      'calibration':calibration,'best_epoch':bestepoch,'best_validation_loss':best,
      'train_count':len(train_orders),'validation_count':len(val_orders),'device':DEVICE,
      'feature_manifest_hashes':[sha(featdir/f'features_shard{i}.json') for i in (0,1)],
      'training_script_sha256':sha(EXP/'run_selector_cv.py'),
      'note':'Final fit for external evaluation only. It is not the saved model underlying the historical OOF predictions; performance must be evaluated on untouched external records.'}
    torch.save({'state_dict':beststate,'config':config,'candidate_models':cv.MODELS},checkpoint_path)
    checkpoint_path.chmod(0o600)
    summary={'status':'fit complete; no external test accessed','split':'grouped 85/15 validation using existing validation_split function and seed 20261111',
      'best_epoch':bestepoch,'best_validation_loss':best,'train_count':len(train_orders),'validation_count':len(val_orders),
      'train_dimension_counts':{d:sum(dimension[o]==d for o in train_orders) for d in cv.DIMS},
      'validation_dimension_counts':{d:sum(dimension[o]==d for o in val_orders) for d in cv.DIMS},
      'group_leakage':False,'calibration':calibration,'elapsed_seconds':round(time.time()-started,1),
      'checkpoint_path':str(checkpoint_path),'checkpoint_sha256':sha(checkpoint_path),
      'caveat':'This checkpoint uses the same selector architecture but is a new final fit; it does not inherit the historical cross-validation OOF score as its test performance.'}
    summary_path.write_text(json.dumps(summary,indent=2)+'\n')
    (PRIVATE/'training_log.json').write_text(json.dumps(log,indent=2)+'\n')
    summary_path.chmod(0o600);(PRIVATE/'training_log.json').chmod(0o600)
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
