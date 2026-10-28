"""Resumable training with separate true-geometry and synthetic-aesthetic masks."""
import json,math,os,random,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from .student_model import ChoreographyStudent,MAX_ACTIONS,student_loss
from .choreography import MATERIALS,VERBS,CHANNELS,PARAM_SIZE,PLAN_SIZE
from .action_training import device_setup,pool


class StudentDataset(Dataset):
    def __init__(self,manifest,split):
        self.root=Path(manifest).parent
        self.records=[r for r in json.loads(Path(manifest).read_text(encoding='utf-8')) if r['split']==split]
        self.songs=[];self.index=[]
        for ident,r in enumerate(self.records):
            z=dict(np.load(self.root/r['tensor'],allow_pickle=False));z['whole']=pool(z['audio'][:,:768]);z['whole_plan']=pool(z['plan'][z['plan_indices']])
            program=json.loads((self.root/r['program']).read_text(encoding='utf-8'));self.songs.append((z,program))
            a,b=float(z['beats'][0]),float(z['beats'][-1])
            def block(start,end):
                calls=[c for c in program['calls'] if start<=c['beat']<end]
                if len(calls)>MAX_ACTIONS:
                    mid=(start+end)/2;block(start,mid);block(mid,end);return
                self.index.append((ident,start,end,True));self.index.append((ident,start,end,False))
            for start in np.arange(a,b,16):block(float(start),float(min(b+1/192,start+16)))

    def __len__(self):return len(self.index)

    def preference_sampling_weights(self, boost=1.):
        from .score import TempoMap
        weights=[]
        for ident,start,end,_ in self.index:
            r=self.records[ident];program=self.songs[ident][1];tempo=TempoMap(program['tempo_map'])
            left,right=tempo.beat_to_time(start),tempo.beat_to_time(end)
            watched=any(left<t['end_seconds'] and right>t['start_seconds'] for t in r.get('reviewed_intervals',[]))
            # Equal weights for the paired real/synthetic views retain their 1:1 mix.
            weights.append(float(boost) if watched else 1.)
        return weights

    def __getitem__(self,i):
        ident,start,end,synthetic=self.index[i];z,program=self.songs[ident];r=self.records[ident]
        mask=(z['beats']>=start)&(z['beats']<end);indices=np.flatnonzero(mask)
        if len(indices)==0:indices=np.array([int(np.argmin(abs(z['beats']-start)))])
        # Long slow-tempo intervals are sampled, never silently cut off.
        indices=indices[::max(1,math.ceil(len(indices)/512))]
        calls=[c for c in program['calls'] if start<=c['beat']<end]
        n=len(calls);material=np.zeros(MAX_ACTIONS,np.int64);verb=material.copy();obj=material.copy()
        params=np.zeros((MAX_ACTIONS,PARAM_SIZE+2),np.float32);pmask=params.copy()
        for j,c in enumerate(calls):
            material[j]=MATERIALS.index(c['material']);verb[j]=VERBS.index(c['verb']);obj[j]=c['object']
            params[j]=[(c['beat']-start)/(end-start),min(c['duration'],end-start)/(end-start),*c['to']]
            pmask[j,:2]=1;pmask[j,np.asarray(CHANNELS[c['verb']],dtype=int)+2]=1
        weight=.25 if synthetic else 0.
        event_weights=np.full(MAX_ACTIONS,weight,np.float32)
        whole_weights=np.full(len(z['whole']),weight,np.float32)
        from .score import TempoMap
        tempo=TempoMap(program['tempo_map'])
        reviewed=r.get('reviewed_intervals',[]);rejected=r.get('rejected_intervals',[])
        def interval_weight(left,right):
            w=1. if any(t['start_seconds']<=left and right<=t['end_seconds'] for t in reviewed) else .25
            if any(left<t['end_seconds'] and right>t['start_seconds'] for t in rejected):w=0.
            return w if synthetic else 0.
        if synthetic:
            left,right=float(z['times'][indices[0]]),float(z['times'][indices[-1]])
            if calls:right=max(right,max(tempo.beat_to_time(c['beat']+c['duration']) for c in calls))
            weight=interval_weight(left,right)
            for j,c in enumerate(calls):event_weights[j]=interval_weight(tempo.beat_to_time(c['beat']),tempo.beat_to_time(c['beat']+c['duration']))
            for j in range(len(whole_weights)):whole_weights[j]=interval_weight(j*2,min(float(z['times'][-1])+1/16,(j+1)*2))
        return dict(audio=z['audio'][indices],times=z['times'][indices],motion=z['motion'][indices],accent=z['accent'][indices],
            mask=np.ones(len(indices),bool),indices=(indices//32).astype(np.int64),whole=z['whole'],whole_plan=z['whole_plan'],
            count=n,material=material,verb=verb,object=obj,parameters=params,parameter_mask=pmask,event_mask=np.arange(MAX_ACTIONS)<n,
            synthetic_weight=np.float32(weight),event_weights=event_weights,whole_weights=whole_weights)


def collate(rows):
    local=max(len(r['audio']) for r in rows);whole=max(len(r['whole']) for r in rows)
    result={}
    for key in rows[0]:
        values=[]
        for r in rows:
            v=np.asarray(r[key])
            if key in ('audio','times','motion','accent','mask','indices','whole','whole_plan','whole_weights'):
                length=whole if key in ('whole','whole_plan','whole_weights') else local
                v=np.pad(v,((0,length-len(v)),)+((0,0),)*(v.ndim-1))
            values.append(v)
        result[key]=torch.as_tensor(np.stack(values))
        if key in ('count','material','verb','object','indices'):
            result[key]=result[key].long()
    result['whole_mask']=torch.tensor(np.stack([np.arange(whole)>=len(r['whole']) for r in rows]))
    return result


def train(manifest,output,initialize=None,resume=None,epochs=40,batch_size=2,accumulate=16,device='cuda',seed=42,patience=5,review_boost=1.):
    if not 1<=review_boost<=32:raise ValueError('review_boost must be between 1 and 32')
    device_setup(device);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    data=StudentDataset(manifest,'train');val=StudentDataset(manifest,'val')
    if not len(data) or not len(val):raise ValueError('Student training and validation records required')
    vocabpath=Path(manifest).parent/'vocabulary.npy';vocabulary=np.load(vocabpath)
    model=ChoreographyStudent(vocabulary).to(device);transferred=[];start=0;best=float('inf');bad=0;steps=0
    if initialize:
        source=torch.load(initialize,map_location='cpu',weights_only=False)['model'];target=model.state_dict()
        selected={k:v for k,v in source.items() if k in target and v.shape==target[k].shape and k!='vocabulary'}
        model.load_state_dict(selected,strict=False);transferred=list(selected)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    if resume:
        c=torch.load(resume,map_location='cpu',weights_only=False);model.plan_memory=c.get('plan_memory',False);model.load_state_dict(c['model']);optimizer.load_state_dict(c['optimizer'])
        digest=__import__('hashlib').sha256(Path(manifest).read_bytes()).hexdigest()
        if c.get('manifest_sha256')!=digest:raise ValueError('Resume manifest changed; use --initialize for a new supervision stage')
        if c.get('review_boost',1.)!=review_boost:raise ValueError('Resume preference sampling changed')
        start=c['epoch']+1;best=c['best'];bad=c['bad'];steps=c['steps'];transferred=c['transferred']
        random.setstate(c['python_rng']);np.random.set_state(c['numpy_rng']);torch.set_rng_state(c['torch_rng'])
        if device.startswith('cuda') and c.get('cuda_rng') is not None:torch.cuda.set_rng_state_all(c['cuda_rng'])
    sampler=WeightedRandomSampler(data.preference_sampling_weights(review_boost),len(data),replacement=True) if review_boost>1 else None
    loader=DataLoader(data,batch_size=batch_size,shuffle=sampler is None,sampler=sampler,collate_fn=collate);validation=DataLoader(val,batch_size=batch_size,collate_fn=collate)
    parameters=sum(p.numel() for p in model.parameters());history=[]
    print(json.dumps(dict(parameters=parameters,train_windows=len(data),validation_windows=len(val),transferred=len(transferred))),flush=True)
    for epoch in range(start,epochs):
        model.train();optimizer.zero_grad(set_to_none=True);losses=[]
        for index,batch in enumerate(loader):
            batch={k:v.to(device) for k,v in batch.items()}
            # Half the batches receive teacher plans; half learn to execute predicted plans.
            teacher=(index%2==0)
            with torch.autocast(device_type='cuda' if device.startswith('cuda') else 'cpu',dtype=torch.bfloat16,enabled=device.startswith('cuda')):
                pred=model.forward_actions(batch,teacher);loss=student_loss(pred,batch)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite student training loss')
            (loss/accumulate).backward();losses.append(float(loss.detach()))
            if (index+1)%accumulate==0 or index+1==len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();optimizer.zero_grad(set_to_none=True);steps+=1
        model.eval();values=[]
        with torch.inference_mode():
            for batch in validation:
                batch={k:v.to(device) for k,v in batch.items()}
                with torch.autocast(device_type='cuda' if device.startswith('cuda') else 'cpu',dtype=torch.bfloat16,enabled=device.startswith('cuda')):
                    values.append(float(student_loss(model.forward_actions(batch,False),batch)))
        value=float(np.mean(values));improved=value<best-1e-4
        if improved:best=value;bad=0
        else:bad+=1
        state=dict(format='choreography-student-v1',plan_memory=model.plan_memory,model=model.state_dict(),vocabulary=vocabulary,optimizer=optimizer.state_dict(),epoch=epoch,
            best=best,bad=bad,steps=steps,transferred=transferred,parameters=parameters,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else None,seed=seed,review_boost=review_boost,manifest_sha256=__import__('hashlib').sha256(Path(manifest).read_bytes()).hexdigest())
        torch.save(state,root/'last.pt')
        if improved:torch.save(state,root/'best.pt')
        row=dict(epoch=epoch,train_loss=float(np.mean(losses)),validation_loss=value,best=best,steps=steps,
            peak_gib=torch.cuda.max_memory_allocated()/1024**3 if device.startswith('cuda') else None)
        history.append(row)
        with (root/'metrics.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        (root/'summary.json').write_text(json.dumps(dict(parameters=parameters,best=best,steps=steps,epoch=epoch,peak_gib=row['peak_gib'],transferred=len(transferred),review_boost=review_boost,plan_memory=model.plan_memory),indent=2))
        if bad>=patience:break
    return json.loads((root/'summary.json').read_text())
