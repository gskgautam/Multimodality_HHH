#!/usr/bin/env python3
"""Leakage-safe, grouped 5-fold cross-validation of a 24-candidate OrchCircuit selector.

Trains on stored evaluator labels and frozen Qwen2-VL answer features. No API calls.
"""
import argparse,copy,json,random,time,hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=(next((p for p in Path(__file__).resolve().parents if (p / "README.md").is_file()), Path.cwd()))
EXP=ROOT/'Method/OrchCircuit/final'
DATA=ROOT/'Datasets/Ch3EF/data/ch3ef_full1416'
RUN=None
FIG4=EXP/'figure4'
VARIANT='full'
MODELS=[
'lfm3-vl-2b','qwen3.5-2b','gemma4-e2b','qwen2.5-vl-7b','qwen3-vl-8b','minicpm-v-4.5-8b',
'gemma3-12b','gemma-4-12b','pixtral-12b','gemma3-27b','qwen3.6-27b','qwen3.8-27b',
'qwen3.5-35b-a3b','qwen3.6-35b-a3b','minicpm-v-4.7-35b-a3b',
'gpt-6.1-terra','gpt-6.1-sol','gpt-6-luna','gemini-4-argon','gemini-3.8-flash','gemini-3.6-flash',
'claude-opus-5-5','claude-sonnet-5-5','claude-haiku-5-5']
OPEN=set(MODELS[:15])
DIMS=['Helpful','Harmless','Honest']
SEED=20261111
DEVICE='cuda:0' if torch.cuda.is_available() else 'cpu'
TOKENS,DIM=64,3584
def readjl(p): return [json.loads(s) for s in p.read_text().splitlines() if s.strip()] if p.exists() else []

class OrchCircuitAll24(nn.Module):
    """Single-pass shared set transformer with criterion-specific prediction heads."""
    def __init__(self, variant='full'):
        super().__init__()
        if variant not in {'full','no_set_attention','no_candidate_identity','mean_token_pool'}:
            raise ValueError(f'unknown ablation variant: {variant}')
        self.variant=variant
        h=64
        self.input_projection=nn.Sequential(nn.LayerNorm(DIM),nn.Linear(DIM,h),nn.GELU())
        self.token_attention=None if variant=='mean_token_pool' else nn.Linear(h,1)
        self.candidate_identity=None if variant=='no_candidate_identity' else nn.Embedding(24,h)
        self.set_layer=None if variant=='no_set_attention' else nn.TransformerEncoderLayer(d_model=h,nhead=4,dim_feedforward=128,
            dropout=.15,batch_first=True,norm_first=True)
        self.wr_head=nn.Sequential(nn.LayerNorm(h),nn.Linear(h,32),nn.GELU(),nn.Dropout(.15),nn.Linear(32,3))
        self.ss_head=nn.Sequential(nn.LayerNorm(h),nn.Linear(h,32),nn.GELU(),nn.Dropout(.15),nn.Linear(32,1))
        self.truth_head=nn.Sequential(nn.LayerNorm(h),nn.Linear(h,32),nn.GELU(),nn.Dropout(.15),nn.Linear(32,1))
        self.info_head=nn.Sequential(nn.LayerNorm(h),nn.Linear(h,32),nn.GELU(),nn.Dropout(.15),nn.Linear(32,1))
    def forward(self,x):
        # x: batch x 24 candidates x 64 frozen image/question/answer tokens x hidden
        h=self.input_projection(x)
        if self.token_attention is None:
            z=h.mean(dim=2)
        else:
            w=torch.softmax(self.token_attention(h).squeeze(-1),dim=-1)
            z=torch.einsum('bnl,bnld->bnd',w,h)
        if self.candidate_identity is not None:
            z=z+self.candidate_identity(torch.arange(z.shape[1],device=z.device))[None,:,:]
        if self.set_layer is not None:
            z=self.set_layer(z)
        return {'wr':self.wr_head(z),'ss':self.ss_head(z).squeeze(-1),
                'truth':self.truth_head(z).squeeze(-1),'info':self.info_head(z).squeeze(-1)}

def initialize_matched_modules(model, seed, fold):
    """Give shared modules identical initial weights across ablation variants."""
    stable_offsets={'input_projection':101,'token_attention':211,'candidate_identity':307,
                    'set_layer':401,'wr_head':503,'ss_head':601,'truth_head':701,'info_head':809}
    fingerprints={}
    for name, offset in stable_offsets.items():
        module=getattr(model,name,None)
        if module is None:
            continue
        component_seed=(int(seed)+int(fold)*1000003+offset*9176)%(2**31-1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(component_seed)
            for sub in module.modules():
                if isinstance(sub,nn.MultiheadAttention):
                    sub._reset_parameters()
                elif callable(getattr(sub,'reset_parameters',None)):
                    sub.reset_parameters()
        digest=hashlib.sha256()
        for key,tensor in module.state_dict().items():
            digest.update(key.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        fingerprints[name]=digest.hexdigest()
    return fingerprints

def norm(s): return ' '.join(str(s).lower().split())
def make_groups(records,orders):
    parent={o:o for o in orders};path_hash={};hashes_by_order={}
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[b]=a
    seen={}
    for o in orders:
        r=records[o];images=r.get('image',[])
        if isinstance(images,str):images=[images]
        keys=['q:'+norm(r.get('query',''))];ih=[]
        for im in images:
            p=(DATA/im).resolve()
            if p not in path_hash:path_hash[p]=hashlib.sha256(p.read_bytes()).hexdigest()
            h=path_hash[p];ih.append(h);keys.append('i:'+h)
        hashes_by_order[str(o)]=ih
        for k in keys:
            if k in seen:union(o,seen[k])
            else:seen[k]=o
    return {o:find(o) for o in orders},hashes_by_order

def grouped_folds(orders,group_of,dimension,seed=SEED):
    grouped=defaultdict(list)
    for o in orders: grouped[group_of[o]].append(o)
    rng=random.Random(seed); keys=list(grouped);rng.shuffle(keys)
    keys.sort(key=lambda g:len(grouped[g]),reverse=True)
    dims=DIMS; totals={d:sum(dimension[o]==d for o in orders) for d in dims}
    target={d:totals[d]/5 for d in dims}
    loads=[{d:0 for d in dims} for _ in range(5)]
    fold_orders=[[] for _ in range(5)]
    for g in keys:
        rows=grouped[g]; gc={d:sum(dimension[o]==d for o in rows) for d in dims}
        empty=[f for f in range(5) if not fold_orders[f]]
        candidates=empty if empty else list(range(5))
        choices=[]
        for f in candidates:
            cost=.15*sum((((loads[f][d]+gc[d]-target[d])/max(target[d],1))**2)-((loads[f][d]-target[d])/max(target[d],1))**2 for d in dims)
            total_target=len(orders)/5
            cost+=20.0*(((len(fold_orders[f])+len(rows)-total_target)/max(total_target,1))**2-((len(fold_orders[f])-total_target)/max(total_target,1))**2)
            choices.append(cost)
        f=candidates[min(range(len(candidates)),key=lambda j:(choices[j],len(fold_orders[candidates[j]]),candidates[j]))]
        fold_orders[f].extend(rows)
        for d in dims: loads[f][d]+=gc[d]
    mapping={o:f for f,rows in enumerate(fold_orders) for o in rows}
    if len(mapping)!=len(orders): raise RuntimeError('fold assignment lost or duplicated records')
    return mapping,fold_orders,loads

def validation_split(train_orders,group_of,dimension,seed):
    bygroup=defaultdict(list)
    for o in train_orders: bygroup[group_of[o]].append(o)
    counts={d:sum(dimension[o]==d for o in train_orders) for d in DIMS}
    target={d:max(1,round(.15*counts[d])) for d in DIMS}
    got={d:0 for d in DIMS}
    keys=list(bygroup);random.Random(seed).shuffle(keys)
    remaining=set(keys);val=[]
    while any(got[d]<target[d] for d in DIMS):
        best=None;best_score=-1e9
        for g in remaining:
            add={d:sum(dimension[o]==d for o in bygroup[g]) for d in DIMS}
            score=0.0
            for d in DIMS:
                deficit=max(0,target[d]-got[d])
                score+=min(deficit,add[d])/target[d]
                score-=.1*max(0,add[d]-deficit)/target[d]
            if score>best_score:best,best_score=g,score
        if best is None: break
        remaining.remove(best);val.extend(bygroup[best])
        for d in DIMS:got[d]+=sum(dimension[o]==d for o in bygroup[best])
    if any(got[d]==0 for d in DIMS):raise RuntimeError(f'validation split misses an HHH dimension: {got}')
    v=set(val)
    return sorted(o for o in train_orders if o not in v),sorted(v)

def labels_and_orders():
    sample=json.loads((DATA/'sample.json').read_text())['records']
    by={int(r['sample_order']):r for r in sample}
    oracle=json.loads((ROOT/'Method/OrchCircuit/final/inputs/oracle_ceiling_24models.json').read_text())
    per_metric={m:sorted(oracle['metrics'][m]['common_sample_orders']) for m in ('WR','SS','TI')}
    orders=sorted(set().union(*(set(v) for v in per_metric.values())))
    dim={o:by[o]['dimension'] for o in orders}
    order_index={o:i for i,o in enumerate(orders)}
    n=len(orders);m=len(MODELS)
    wr=np.full((n,m),-1,np.int64);ss=np.full((n,m),-1,np.int64)
    truth=np.full((n,m),-1,np.int64);info=np.full((n,m),-1,np.int64)
    for mi,slug in enumerate(MODELS):
        folder=ROOT/'Baselines'/('Open' if slug in OPEN else 'Closed')/slug/'runs/ch3ef-full1416'
        for row in readjl(folder/'wr_ti_gpt4o.jsonl'):
            o=row.get('sample_order')
            if o is None:o=next((int(r['sample_order']) for r in sample if r['dimension']==row.get('dimension') and str(r['id'])==str(row.get('record_id'))),None)
            if o not in order_index:continue
            i=order_index[o]
            if row.get('task')=='wr_judge' and row.get('preference') is not None:
                p=row['preference']
                if p=='qwen':p='qwen2.5-vl-7b'
                wr[i,mi]=0 if p==slug else (1 if p=='tie' else 2)
            if row.get('task')=='ti_judge' and isinstance(row.get('truthful'),bool) and isinstance(row.get('informative'),bool):
                truth[i,mi]=int(row['truthful']);info[i,mi]=int(row['informative'])
        safety=folder/'ss_beaver_dam.json'
        if not safety.exists():safety=folder/'ss_beaver_dam_valid_subset.json'
        if safety.exists():
            for row in json.loads(safety.read_text()).get('records',[]):
                o=row.get('sample_order')
                if o is None:
                    o=next((int(r['sample_order']) for r in sample if r['dimension']==row.get('dimension') and str(r['id'])==str(row.get('record_id'))),None)
                if o in order_index and isinstance(row.get('unsafe'),bool):ss[order_index[o],mi]=int(row['unsafe'])
    for met,needed,dimname in [('WR',wr,'Helpful'),('SS',ss,'Harmless'),('TI',truth,'Honest')]:
        for o in per_metric[met]:
            i=order_index[o]
            arr=needed if met!='TI' else truth
            if np.any(arr[i]<0):raise RuntimeError(f'incomplete {met} labels at common sample {o}')
            if met=='TI' and np.any(info[i]<0):raise RuntimeError(f'incomplete TI informative labels at common sample {o}')
            if dim[o]!=dimname:raise RuntimeError(f'wrong dimension for {met} sample {o}')
    return by,orders,dim,per_metric,wr,ss,truth,info

def metrics(metric,values):
    n=len(values)
    if metric=='WR':
        w=sum(v==0 for v in values);t=sum(v==1 for v in values);l=sum(v==2 for v in values)
        return {'n':n,'wins':w,'ties':t,'losses':l,'pct':100*w/n}
    if metric=='SS':
        u=sum(values);return {'n':n,'unsafe':int(u),'pct':100*u/n}
    tr=sum(v[0] for v in values);inf=sum(v[1] for v in values)
    return {'n':n,'truthful':int(tr),'informative':int(inf),'both':int(sum(a and b for a,b in values)),
            'pct':100*(tr/n)*(inf/n),'truthful_pct':100*tr/n,'informative_pct':100*inf/n}

def fit_binary(logits,targets):
    x=torch.as_tensor(logits,device=DEVICE,dtype=torch.float32).reshape(-1)
    y=torch.as_tensor(targets,device=DEVICE,dtype=torch.float32).reshape(-1)
    if len(y)<10 or y.min()==y.max(): return (1.0,0.0)
    a=nn.Parameter(torch.ones((),device=DEVICE));b=nn.Parameter(torch.zeros((),device=DEVICE))
    opt=torch.optim.LBFGS([a,b],lr=.2,max_iter=35,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad();loss=F.binary_cross_entropy_with_logits(a*x+b,y);loss.backward();return loss
    opt.step(closure)
    return float(a.detach().cpu()),float(b.detach().cpu())

def fit_temperature(logits,targets):
    x=torch.as_tensor(logits,device=DEVICE,dtype=torch.float32).reshape(-1,3)
    y=torch.as_tensor(targets,device=DEVICE,dtype=torch.long).reshape(-1)
    if len(y)<10 or len(torch.unique(y))<2:return 1.0
    logt=nn.Parameter(torch.zeros((),device=DEVICE))
    opt=torch.optim.LBFGS([logt],lr=.2,max_iter=35,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad();loss=F.cross_entropy(x/torch.exp(logt).clamp(.05,20),y);loss.backward();return loss
    opt.step(closure)
    return float(torch.exp(logt.detach()).clamp(.05,20).cpu())

def group_bootstrap(metric,orders,group_of,selector_values,baseline_values,reps=100000,seed=SEED):
    keys=sorted({group_of[o] for o in orders});pos={g:i for i,g in enumerate(keys)}
    grouped=defaultdict(list)
    for o in orders:grouped[group_of[o]].append(o)
    counts=np.asarray([len(grouped[g]) for g in keys],dtype=np.float64)
    if metric=='TI':
        st=np.zeros(len(keys));si=np.zeros(len(keys));bt=np.zeros((len(keys),24));bi=np.zeros((len(keys),24))
        for gi,g in enumerate(keys):
            for o in grouped[g]:
                sv=selector_values[o];st[gi]+=sv[0];si[gi]+=sv[1]
                for m in range(24):
                    bv=baseline_values[m][o];bt[gi,m]+=bv[0];bi[gi,m]+=bv[1]
    else:
        ss=np.zeros(len(keys));bs=np.zeros((len(keys),24))
        for gi,g in enumerate(keys):
            for o in grouped[g]:
                sv=selector_values[o]
                ss[gi]+= (sv==0) if metric=='WR' else sv
                for m in range(24):
                    bv=baseline_values[m][o]
                    bs[gi,m]+= (bv==0) if metric=='WR' else bv
    alpha=.05/(3*24);lo_q=alpha/2;hi_q=1-alpha/2
    rng=np.random.default_rng(seed);draws=np.empty((reps,24),np.float64)
    for start in range(0,reps,500):
        b=min(500,reps-start);sampled=rng.integers(0,len(keys),size=(b,len(keys)))
        den=counts[sampled].sum(axis=1)
        if metric=='TI':
            aT=st[sampled].sum(axis=1);aI=si[sampled].sum(axis=1)
            bT=bt[sampled].sum(axis=1);bI=bi[sampled].sum(axis=1)
            draws[start:start+b]=100*((aT[:,None]/den[:,None])*(aI[:,None]/den[:,None])-(bT/den[:,None])*(bI/den[:,None]))
        elif metric=='WR':
            a=ss[sampled].sum(axis=1);bb=bs[sampled].sum(axis=1)
            draws[start:start+b]=100*(a[:,None]-bb)/den[:,None]
        else:
            a=ss[sampled].sum(axis=1);bb=bs[sampled].sum(axis=1)
            draws[start:start+b]=100*(bb-a[:,None])/den[:,None]
    draws.sort(axis=0)
    low=draws[max(0,int(lo_q*(reps-1))),:];high=draws[min(reps-1,int(hi_q*(reps-1))),:]
    return [{'model':MODELS[m],'delta_ci_pp':[float(low[m]),float(high[m])],
             'beats_model_simultaneously':bool(low[m]>0)} for m in range(24)]

def main():
    global RUN, SEED, VARIANT
    ap=argparse.ArgumentParser(description='Run a locked-fold OrchCircuit component ablation.')
    ap.add_argument('--variant',required=True,choices=['full','no_set_attention','no_candidate_identity','mean_token_pool'])
    ap.add_argument('--training-seed',type=int,required=True)
    ap.add_argument('--panel',choices=['a','c'],default='a')
    ap.add_argument('--init-scheme',choices=['matched','native'],default='matched')
    args=ap.parse_args()
    VARIANT=args.variant
    SEED=args.training_seed
    RUN=FIG4/args.panel/'runs'/f'{VARIANT}_seed{SEED}_{args.init_scheme}'
    if RUN.exists():
        raise FileExistsError(f'Output directory already exists; preserve it and choose a new run name: {RUN}')
    RUN.mkdir(parents=True,exist_ok=False)
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(SEED)
    by,orders,dimension,per_metric,wr,ss,truth,info=labels_and_orders()
    n=len(orders); oi={o:i for i,o in enumerate(orders)}
    featdir=EXP/'data_selected_answer'
    manifests=[json.loads((featdir/f'features_shard{i}.json').read_text()) for i in (0,1)]
    dones=[json.loads((featdir/f'features_shard{i}.done.json').read_text()) for i in (0,1)]
    for i in (0,1):
        if manifests[i]['sample_orders']!=orders:raise RuntimeError('feature sample order mismatch')
        if set(dones[i]['completed_models'])!=set(manifests[i]['models']):raise RuntimeError(f'feature shard {i} incomplete: {dones[i]}')
    f0=np.memmap(featdir/'features_shard0.f16',dtype=np.float16,mode='r',shape=(n,12,TOKENS,DIM))
    f1=np.memmap(featdir/'features_shard1.f16',dtype=np.float16,mode='r',shape=(n,12,TOKENS,DIM))
    m0={m:i for i,m in enumerate(manifests[0]['models'])};m1={m:i for i,m in enumerate(manifests[1]['models'])}
    def getx(batch_orders):
        ix=np.asarray([oi[o] for o in batch_orders],dtype=np.int64)
        x=np.empty((len(ix),24,TOKENS,DIM),dtype=np.float16)
        for mi,m in enumerate(MODELS):
            x[:,mi]=f0[ix,m0[m]] if m in m0 else f1[ix,m1[m]]
        return torch.from_numpy(x).to(DEVICE,dtype=torch.float32)
    group_of,image_hashes=make_groups(by,orders)
    foldmap,fold_orders,fold_loads=grouped_folds(orders,group_of,dimension)
    (RUN/'folds.json').write_text(json.dumps({'split_seed':20261111,'training_seed':SEED,'variant':VARIANT,'method':'largest-first size-balanced grouped 5-fold with dimension-aware tie-break; groups link normalized query text or identical SHA-256 image content','orders_by_fold':fold_orders,'dimension_counts_by_fold':fold_loads,'group_ids':{str(o):int(group_of[o]) for o in orders},'image_sha256_by_order':image_hashes},indent=2)+'\n')
    wrpred=np.full(n,-1,np.int64);sspred=np.full(n,-1,np.int64);tipred=np.full((n,2),-1,np.int64)
    fold_logs=[]
    for fi in range(5):
        test_orders=sorted(fold_orders[fi]);remaining=[o for o in orders if foldmap[o]!=fi]
        train_orders,val_orders=validation_split(remaining,group_of,dimension,20261111+fi*17)
        print(f'FOLD {fi}: train={len(train_orders)} val={len(val_orders)} test={len(test_orders)}; dims={fold_loads[fi]}',flush=True)
        torch.manual_seed(SEED+fi);np.random.seed(SEED+fi);random.seed(SEED+fi)
        model=OrchCircuitAll24(VARIANT)
        init_fingerprints=initialize_matched_modules(model,SEED,fi) if args.init_scheme=='matched' else {}
        # Align the training/dropout RNG streams after construction and module initialization.
        torch.manual_seed(SEED+fi);np.random.seed(SEED+fi);random.seed(SEED+fi)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(SEED+fi)
        model=model.to(DEVICE)
        opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01)
        best=float('inf');beststate=None;stale=0;train_by={d:[o for o in train_orders if dimension[o]==d] for d in DIMS}
        val_by={d:[o for o in val_orders if dimension[o]==d] for d in DIMS}
        def loss_for(batch):
            ix=torch.as_tensor([oi[o] for o in batch],device=DEVICE)
            pred=model(getx(batch));losses=[]
            for d in DIMS:
                ix_np=np.asarray([oi[o] for o in batch if dimension[o]==d],dtype=np.int64)
                if not len(ix_np):continue
                ii=torch.as_tensor(ix_np,device=DEVICE)
                if d=='Helpful':
                    y=torch.as_tensor(wr[ix_np],device=DEVICE,dtype=torch.long)
                    losses.append(F.cross_entropy(pred['wr'][torch.as_tensor([dimension[o]=='Helpful' for o in batch],device=DEVICE)].reshape(-1,3),y.reshape(-1)))
                elif d=='Harmless':
                    mask=torch.as_tensor([dimension[o]=='Harmless' for o in batch],device=DEVICE)
                    y=torch.as_tensor(ss[ix_np],device=DEVICE,dtype=torch.float32)
                    losses.append(F.binary_cross_entropy_with_logits(pred['ss'][mask],y))
                else:
                    mask=torch.as_tensor([dimension[o]=='Honest' for o in batch],device=DEVICE)
                    yt=torch.as_tensor(truth[ix_np],device=DEVICE,dtype=torch.float32)
                    yi=torch.as_tensor(info[ix_np],device=DEVICE,dtype=torch.float32)
                    losses.append(F.binary_cross_entropy_with_logits(pred['truth'][mask],yt)+F.binary_cross_entropy_with_logits(pred['info'][mask],yi))
            return torch.stack(losses).mean()
        def batches_by_dim(pool,shuffle=True):
            queues={d:list(pool[d]) for d in DIMS}
            if shuffle:
                for d in DIMS:random.shuffle(queues[d])
            out=[]
            while any(queues[d] for d in DIMS):
                b=[]
                for d in DIMS:
                    b.extend(queues[d][:2]);queues[d]=queues[d][2:]
                if b:out.append(b)
            return out
        for ep in range(1,21):
            model.train();tl=[]
            for batch in batches_by_dim(train_by,True):
                l=loss_for(batch);opt.zero_grad(set_to_none=True);l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();tl.append(float(l.detach().cpu()))
            model.eval();vl=[]
            with torch.inference_mode():
                for batch in batches_by_dim(val_by,False):vl.append(float(loss_for(batch).detach().cpu()))
            v=float(np.mean(vl))
            print(f' fold={fi} epoch={ep} train={np.mean(tl):.4f} val={v:.4f}',flush=True)
            if v<best-1e-5:
                best=v;beststate={k:x.detach().cpu().clone() for k,x in model.state_dict().items()};stale=0;bestepoch=ep
            else:
                stale+=1
                if stale>=4:break
        if beststate is None:raise RuntimeError('no checkpoint')
        model.load_state_dict(beststate);model.eval()
        def collect(pool):
            got=[]
            pool_by_dim={d:[o for o in pool if dimension[o]==d] for d in DIMS}
            with torch.inference_mode():
                for batch in batches_by_dim(pool_by_dim,False):
                    x=model(getx(batch))
                    for j,o in enumerate(batch):
                        got.append((o,{k:v[j].detach().cpu().numpy() for k,v in x.items()}))
            return got
        val_raw=dict(collect(val_orders))
        val_wr=[];val_wry=[];val_ss=[];val_ssy=[];val_tr=[];val_try=[];val_in=[];val_iny=[]
        for o,z in val_raw.items():
            ix=oi[o]
            if dimension[o]=='Helpful':val_wr.extend(z['wr']);val_wry.extend(wr[ix]);
            if dimension[o]=='Harmless':val_ss.extend(z['ss']);val_ssy.extend(ss[ix])
            if dimension[o]=='Honest':
                val_tr.extend(z['truth']);val_try.extend(truth[ix]);val_in.extend(z['info']);val_iny.extend(info[ix])
        temperature=fit_temperature(np.asarray(val_wr),np.asarray(val_wry))
        ss_cal=fit_binary(np.asarray(val_ss),np.asarray(val_ssy))
        tr_cal=fit_binary(np.asarray(val_tr),np.asarray(val_try))
        in_cal=fit_binary(np.asarray(val_in),np.asarray(val_iny))
        test_raw=dict(collect(test_orders))
        for o,z in test_raw.items():
            i=oi[o]
            if dimension[o]=='Helpful':
                p=torch.softmax(torch.as_tensor(z['wr']/temperature),dim=-1).numpy()
                wrpred[i]=int(np.argmax(p[:,0]))
            elif dimension[o]=='Harmless':
                p=1/(1+np.exp(-np.clip(ss_cal[0]*z['ss']+ss_cal[1],-30,30)))
                sspred[i]=int(np.argmin(p))
            else:
                pt=1/(1+np.exp(-np.clip(tr_cal[0]*z['truth']+tr_cal[1],-30,30)))
                pi=1/(1+np.exp(-np.clip(in_cal[0]*z['info']+in_cal[1],-30,30)))
                tipred[i]=np.asarray([int(np.argmax(pt*pi)),0])
        fold_logs.append({'fold':fi,'variant':VARIANT,'training_seed':SEED,'split_seed':20261111,'train_n':len(train_orders),'validation_n':len(val_orders),'test_n':len(test_orders),
          'init_scheme':args.init_scheme,'initialization_fingerprints':init_fingerprints,
          'best_epoch':bestepoch,'best_validation_loss':best,'wr_temperature':temperature,'ss_calibration':ss_cal,
          'truth_calibration':tr_cal,'info_calibration':in_cal})
        (RUN/f'fold{fi}.json').write_text(json.dumps(fold_logs[-1],indent=2)+'\n')
        print('FOLD_DONE',fold_logs[-1],flush=True)
        del model,opt
        if torch.cuda.is_available():torch.cuda.empty_cache()
    if np.any(wrpred[np.array([oi[o] for o in per_metric['WR']])]<0) or np.any(sspred[np.array([oi[o] for o in per_metric['SS']])]<0) or np.any(tipred[np.array([oi[o] for o in per_metric['TI']])]<0):
        raise RuntimeError('some OOF test predictions are missing')
    selector={};baseline={}
    for met in ('WR','SS','TI'):
        selector[met]={};baseline[met]={m:{} for m in MODELS}
        for o in per_metric[met]:
            i=oi[o];chosen=int(wrpred[i] if met=='WR' else (sspred[i] if met=='SS' else tipred[i,0]))
            if met=='WR':selector[met][o]=int(wr[i,chosen])
            elif met=='SS':selector[met][o]=int(ss[i,chosen])
            else:selector[met][o]=(int(truth[i,chosen]),int(info[i,chosen]))
            for mi,m in enumerate(MODELS):
                if met=='WR':baseline[met][m][o]=int(wr[i,mi])
                elif met=='SS':baseline[met][m][o]=int(ss[i,mi])
                else:baseline[met][m][o]=(int(truth[i,mi]),int(info[i,mi]))
    summary={'variant':VARIANT,'training_seed':SEED,'split_seed':20261111,'init_scheme':args.init_scheme,'candidate_count':24,'common_valid_n':{m:len(per_metric[m]) for m in per_metric},
      'outer_cv':f'five-fold grouped by normalized query or identical SHA-256 image content; inner grouped 15% validation for early stopping and calibration; fold seed 20261111; training seed {SEED}',
      'prediction_rule':{'WR':'choose candidate with maximum calibrated probability of judge win; tie and loss are separate train classes','SS':'choose candidate with minimum calibrated unsafe probability','TI':'choose candidate with maximum calibrated truthful probability times informative probability'},
      'selector_oof':{m:metrics(m,list(selector[m].values())) for m in ('WR','SS','TI')},
      'baselines_common_subset':{m:{slug:metrics(m,[baseline[m][slug][o] for o in per_metric[m]]) for slug in MODELS} for m in ('WR','SS','TI')},
      'folds':fold_logs,'warning':'All outcomes are automated-judge-relative; no human-verified correctness labels. Candidate pool and test cohort are restricted to items with valid outputs/labels for all 24 baselines.'}
    for met in ('WR','SS','TI'):
        mm=summary['baselines_common_subset'][met]
        best=max(mm,key=lambda x:mm[x]['pct']) if met!='SS' else min(mm,key=lambda x:mm[x]['pct'])
        summary.setdefault('best_fixed_on_oof_cohort',{})[met]={'model':best,**mm[best]}
        summary.setdefault('paired_simultaneous_95ci',{})[met]=group_bootstrap(met,per_metric[met],group_of,selector[met],list(baseline[met].values()))
        passed=all(x['beats_model_simultaneously'] for x in summary['paired_simultaneous_95ci'][met])
        summary.setdefault('all24_simultaneous_lower_ci_above_zero',{})[met]=passed
    summary['overall_all24_gate']=all(summary['all24_simultaneous_lower_ci_above_zero'].values())
    out=RUN/'results.json';out.write_text(json.dumps(summary,indent=2)+'\n')
    def show(m,x):
        if m=='WR':return f"{x['pct']:.2f}% ({x['wins']}/{x['n']} wins; {x['ties']} ties)"
        if m=='SS':return f"{x['pct']:.2f}% ({x['unsafe']}/{x['n']} flagged)"
        return f"{x['pct']:.2f}% ({x['truthful']}/{x['n']} truthful × {x['informative']}/{x['n']} informative)"
    lines=['# OrchCircuit all-24 selector: grouped out-of-fold results','',
      'These are cross-validated automated-judge scores on the strict common-valid subset. The per-item selector is trained without the held-out query/image groups. It selects one of 24 cached model answers; it does not generate a new answer.','',
      '| Metric | OrchCircuit OOF | Best fixed baseline on same records | OOF minus best (pp) | n | Simultaneous CI lower bound >0 vs every baseline? |',
      '|---|---:|---:|---:|---:|:---:|']
    for met in ('WR','SS','TI'):
        s=summary['selector_oof'][met];b=summary['best_fixed_on_oof_cohort'][met]
        delta=(b['pct']-s['pct']) if met=='SS' else (s['pct']-b['pct'])
        lines.append(f"| {met} | {show(met,s)} | {show(met,b)} ({b['model']}) | {delta:+.2f} | {s['n']} | {'Yes' if summary['all24_simultaneous_lower_ci_above_zero'][met] else 'No'} |")
    lines+=['','The simultaneous confidence intervals use a paired bootstrap over connected query/image-content groups, Bonferroni-adjusted over 72 model-by-metric comparisons. A positive point estimate alone is not evidence that the selector beats all 24. The strict all-24 claim passes only if the adjusted lower confidence bound is above zero for every metric versus every baseline.','',
      '## Interpretation','',summary['warning'],'']
    (RUN/'results.md').write_text('\n'.join(lines))
    predfile=RUN/'oof_selected_candidates.jsonl'
    with predfile.open('w') as f:
        for o in orders:
            i=oi[o]
            if dimension[o]=='Helpful':selected=MODELS[int(wrpred[i])]
            elif dimension[o]=='Harmless':selected=MODELS[int(sspred[i])]
            else:selected=MODELS[int(tipred[i,0])]
            f.write(json.dumps({'sample_order':o,'dimension':dimension[o],'selected_candidate':selected,
              'wr_selected_candidate':MODELS[int(wrpred[i])] if dimension[o]=='Helpful' else None,
              'ss_selected_candidate':MODELS[int(sspred[i])] if dimension[o]=='Harmless' else None,
              'ti_selected_candidate':MODELS[int(tipred[i,0])] if dimension[o]=='Honest' else None})+'\n')
    print(json.dumps({'output':str(out),'overall_all24_gate':summary['overall_all24_gate'],
      'selector_oof':summary['selector_oof'],'best_fixed':summary['best_fixed_on_oof_cohort']},indent=2),flush=True)
if __name__=='__main__':main()
