"""Both variants execute student predictions. No synthetic compiler at inference."""
import json,math
from pathlib import Path
import numpy as np
import torch
from scipy.signal import find_peaks
from .student_model import ChoreographyStudent,MAX_ACTIONS
from .student_training import collate
from .action_training import pool,device_setup
from .choreography import (PROFILE,ACTION_VERSION,MATERIALS,VERBS,DEFAULT,CHANNELS,plan_vectors,vectors_plan,validate_plan,validate_program,states,score_program)
from .audio import Features,sha256
from .music_analysis import analyze_features
from .score import TempoMap


def load_student(checkpoint,device='cpu'):
    device_setup(device)
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ck.get('format')!='choreography-student-v1':raise ValueError('Not a choreography student checkpoint')
    model=ChoreographyStudent(ck['vocabulary'],plan_memory=ck.get('plan_memory',False)).to(device);model.load_state_dict(ck['model']);model.eval()
    return model,ck


def section_plan_vectors(raw,sections):
    if len(raw)==0 or not np.isfinite(raw).all():raise ValueError('Invalid student plan features')
    result=[]
    for section in sections:
        # Inverse tempo arithmetic can represent time zero as -1e-16.
        left=min(len(raw)-1,max(0,math.floor(max(0.,section['start_seconds'])/2)))
        right=min(len(raw),max(left+1,math.ceil(section['end_seconds']/2)))
        result.append(raw[left:right].mean(0))
    return np.asarray(result)


@torch.inference_mode()
def generate(checkpoint,features,output,planner='model',plan=None,audio=None,device='cpu',seed=42,condition='correct',conditioning_features=None):
    if planner not in ('model','llm'):raise ValueError('Unknown plan source')
    f=Features(features)
    if audio and sha256(audio)!=f.metadata['audio_sha256']:raise ValueError('Audio/cache mismatch')
    if f.metadata.get('backend')!='mert':raise ValueError('Student requires MERT features')
    from .action_data import MERT_REVISION
    if f.metadata.get('revision')!=MERT_REVISION:raise ValueError('Student requires pinned MERT revision')
    analysis=analyze_features(features,audio)
    times=np.arange(0,f.metadata['duration'],1/16);s,t=f.sample(times);a=np.c_[s,t]
    conditioning_hash=f.metadata['audio_sha256']
    if condition=='zero':a[:]=0
    elif condition=='shuffle':
        chunks=[a[i:i+32] for i in range(0,len(a),32)];order=np.random.default_rng(seed).permutation(len(chunks));a=np.concatenate([chunks[i] for i in order])
    elif condition=='swap':
        if not conditioning_features:raise ValueError('Swap needs another song feature cache')
        other=Features(conditioning_features)
        if other.metadata.get('backend')!='mert' or other.metadata.get('revision')!=MERT_REVISION:raise ValueError('Conditioning feature revision mismatch')
        if other.metadata['audio_sha256']==f.metadata['audio_sha256']:raise ValueError('Swap needs a different song')
        beat_grid=np.array([f.tempo.time_to_beat(float(t)) for t in times]);beats_other=np.array([other.tempo.time_to_beat(0),other.tempo.time_to_beat(other.metadata['duration'])])
        lo,hi=f.tempo.time_to_beat(0),f.tempo.time_to_beat(f.metadata['duration'])
        mapped=beats_other[0]+np.mod(beat_grid-lo,max(beats_other[1]-beats_other[0],1e-8))
        stamps=np.array([other.tempo.beat_to_time(float(b)) for b in mapped]);s,t=other.sample(stamps);a=np.c_[s,t]
        conditioning_hash=other.metadata['audio_sha256']
    elif condition!='correct':raise ValueError('Unknown audio condition')
    whole=pool(a[:,:768]);model,ck=load_student(checkpoint,device)
    if planner=='llm':
        if not plan:raise ValueError('LLM variant requires a saved validated plan')
        plan=json.loads(Path(plan).read_text(encoding='utf-8')) if isinstance(plan,(str,Path)) else plan
        validate_plan(plan,analysis)
    else:
        # Predict macro structure directly from whole-track audio, with no API call.
        from .action_model import position
        g=model.global_in(torch.tensor(whole[None],device=device))+position(torch.arange(len(whole),device=device)[None]*2,256)
        g=model.planner(g);raw=model.structure(g).sigmoid()[0].cpu().numpy()
        plan=vectors_plan(section_plan_vectors(raw,analysis['sections']),analysis)
    vectors=plan_vectors(plan)
    sec_start=np.array([s['start_seconds'] for s in analysis['sections']])
    whole_idx=np.clip(np.searchsorted(sec_start,np.arange(len(whole))*2,side='right')-1,0,len(vectors)-1)
    whole_plan=vectors[whole_idx]
    tm=f.tempo;beats=np.array([tm.time_to_beat(float(t)) for t in times]);lo,hi=tm.time_to_beat(0),tm.time_to_beat(f.metadata['duration'])
    program=dict(profile=PROFILE,version=ACTION_VERSION,duration=float(f.metadata['duration']),tempo_map=f.tempo.to_list(),seed=seed,
        groups=[dict(id=i,name='model-object-'+str(i)) for i in range(8)],calls=[],hits=[])
    accent_track=np.zeros(len(times));counts=[]
    for start in np.arange(lo,hi,16):
        end=min(hi,start+16);indices=np.flatnonzero((beats>=start)&(beats<end))
        if not len(indices):continue
        sampled=indices[::max(1,math.ceil(len(indices)/512))]
        batch=dict(audio=torch.tensor(a[sampled][None],device=device),times=torch.tensor(times[sampled][None],dtype=torch.float32,device=device),
            whole=torch.tensor(whole[None],device=device),whole_mask=torch.zeros((1,len(whole)),dtype=torch.bool,device=device),
            whole_plan=torch.tensor(whole_plan[None],device=device),indices=torch.tensor((sampled//32)[None],device=device),mask=torch.ones((1,len(sampled)),dtype=torch.bool,device=device))
        # Match training: predicted continuous plans vs validated external plans.
        pred=model.forward_actions(batch,planner=='llm')
        count=int(pred['count'][0].argmax());counts.append(count)
        accent_track[indices]=np.interp(times[indices],times[sampled],pred['accent'][0].sigmoid().cpu().numpy())
        params=pred['parameters'][0].cpu().numpy();materials=pred['material'][0].argmax(-1).cpu().numpy()
        verbs=pred['verb'][0].argmax(-1).cpu().numpy();objects=pred['object'][0].argmax(-1).cpu().numpy()
        order=sorted(range(count),key=lambda j:float(params[j,0]))
        for j in order:
            at=float(start+params[j,0]*(end-start));duration=float(min(params[j,1]*(end-start),hi-at))
            obj=int(objects[j]);material=MATERIALS[int(materials[j])];verb=VERBS[int(verbs[j])]
            previous=states(program,tm.beat_to_time(at))[obj]
            source=previous['values'].copy()
            target=source.copy()
            if verb=='draw':source[10]=0.
            for k in CHANNELS[verb]:target[k]=float(params[j,k+2])
            if verb=='exit':target[4]=0.
            if verb=='split':target[8]=max(source[8],target[8]);target[9]=max(source[9],target[9])
            if verb in ('merge','gather'):target[8]=min(source[8],target[8])
            if verb=='merge':target[9]=min(source[9],target[9])
            if verb=='scatter':target[8]=max(source[8],target[8])
            if verb=='hold':duration=min(duration,hi-at)
            section=min(len(plan['sections'])-1,max(0,int(np.searchsorted(sec_start,tm.beat_to_time(at),side='right')-1)))
            call=dict(object=obj,material=material,verb=verb,beat=at,duration=duration,
                **{'from':source,'to':target},motif_id=plan['sections'][section]['motif_id'],curve='smooth')
            if verb=='enter' and previous['material']!=material and source[4]>.001:
                call['blend_material']=True
            for key in ('beat','duration'):
                tick=int(round(call[key]*192));call[key+'_tick']=tick;call[key+'_residual']=call[key]-tick/192
            program['calls'].append(call)
        program['calls'].sort(key=lambda c:(c['beat'],c['object']))
    peaks,_=find_peaks(accent_track,height=.5,distance=2,prominence=.05)
    program['hits']=[[float(times[i]),float(accent_track[i])] for i in peaks]
    validate_program(program)
    score=score_program(program,dict(origin='learned_choreography',title='Music Choreographer / '+('LLM plan + same student' if planner=='llm' else 'Learned plan + student'),
        planner=planner,plan_encoding='continuous_student_prediction' if planner=='model' else 'external_structured_plan',condition=condition,audio=f.metadata,checkpoint_sha256=sha256(checkpoint),model_epoch=ck['epoch'],seed=seed,
        conditioning_audio_sha256=conditioning_hash,conditioning_time_mapping='declared one-to-one beat feature warp with bounded wrapping' if condition=='swap' else 'original timestamps',
        complete_generation=bool(program['calls']),decoder='four-layer learned compound-action queries; chronology and reference constraints',
        lifecycle_policy='model-selected entrances; hidden transforms prepare already-created objects',
        aesthetic_supervision='separately marked LLM synthetic and human-selected intervals'),plan)
    score.save(output)
    return dict(output=str(output),actions=len(program['calls']),materials=sorted({c['material'] for c in program['calls']}),verbs=sorted({c['verb'] for c in program['calls']}),counts=counts,planner=planner)
