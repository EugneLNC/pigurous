"""Traceable synthetic expansion; this compiler is never used by student inference."""
import json,copy,math
from pathlib import Path
import numpy as np
from .choreography import (DEFAULT,CHANNELS,PROFILE,ACTION_VERSION,plan_vectors,validate_plan,validate_program,states)
from .score import TempoMap


def demonstration(plan,analysis,motion,times,accents,seed=42):
    validate_plan(plan,analysis)
    program=dict(profile=PROFILE,version=ACTION_VERSION,duration=analysis['duration'],tempo_map=analysis['tempo_map'],seed=seed,
        groups=[dict(id=i,name='independent-'+str(i)) for i in range(8)],calls=[],hits=[])
    tm=TempoMap(program['tempo_map'])
    for s in plan['sections']:
        a,b=s['start_beat'],s['end_beat'];n=len(s['materials'])
        for obj,material in enumerate(s['materials']):
            previous=states(program,tm.beat_to_time(a))[obj]['values'].copy()
            target=previous.copy();anchor=s['anchor'];target[0]=.5+anchor[0]*.25;target[1]=.5+anchor[1]*.3
            if s['layout']=='bilateral':target[0]+=(obj-(n-1)/2)*.28
            elif s['layout']=='ring':target[0]+=.22*math.cos(obj*math.tau/n);target[1]+=.25*math.sin(obj*math.tau/n)
            elif s['layout']=='diagonal':target[0]+=(obj-(n-1)/2)*.2;target[1]+=(obj-(n-1)/2)*.16
            elif s['layout']=='field':target[0]+=.28*math.sin(obj*2.4);target[1]+=.25*math.cos(obj*2.4)
            target[2]=s['scale']*(.65 if obj else .9);target[4]=.30 if obj else .78
            target[5]=(s['hue']+obj*.13)%1;target[6]=s['variation'];target[9]=s['density']*.2
            target[10]=1;target[11]=s['intensity'];target=np.clip(target,0,1).tolist()
            program['calls'].append(dict(object=obj,material=material,verb='enter',beat=a,duration=min(1.,(b-a)/8),
                **{'from':previous,'to':target},motif_id=s['motif_id'],role='background' if obj else 'foreground'))
            verbs=[v for v in s['verbs'] if v not in ('enter','exit')]
            for j,verb in enumerate(verbs):
                at=a+(j+1)*(b-a)/max(2,len(verbs)+2);dur=min((b-a)/max(2,len(verbs)+2),b-at)
                sample=int(np.argmin(abs(times-tm.beat_to_time(at))));m=motion[sample]
                source=target.copy();dest=target.copy();phase=(j+obj*.5)*1.8+s['variation']*2
                if verb=='translate':dest[0]=.5+.25*math.sin(phase)+.15*(m[1]-.5);dest[1]=.5+.20*math.cos(phase)+.15*(m[2]-.5);dest[7]=.25+.5*s['variation']
                elif verb=='rotate':dest[3]=.25+.5*((source[3]+.17+s['variation']*.3)%1)
                elif verb=='scale':dest[2]=min(.85,s['scale']*(.65+.4*math.sin(phase)))
                elif verb=='draw':source[10]=0;dest[10]=1
                elif verb=='morph':dest[6]=(source[6]+.35+s['variation']*.2)%1
                elif verb=='split':dest[8]=.2+.45*s['intensity'];dest[9]=.15+.3*s['density']
                elif verb=='merge':dest[8]=0;dest[9]=0
                elif verb=='gather':dest[0]=.5+anchor[0]*.15;dest[1]=.5+anchor[1]*.15;dest[8]=0
                elif verb=='scatter':dest[0]=.5+.3*math.sin(phase);dest[1]=.5+.25*math.cos(phase);dest[8]=.35+.25*s['intensity']
                dest=np.clip(dest,0,1).tolist()
                program['calls'].append(dict(object=obj,material=material,verb=verb,beat=at,duration=dur,
                    **{'from':source,'to':dest},motif_id=s['motif_id'],role='background' if obj else 'foreground'))
                for k in CHANNELS[verb]:target[k]=dest[k]
        for obj in range(n,8):
            previous=states(program,tm.beat_to_time(a))[obj]['values'].copy()
            if previous[4]>.001:
                dest=previous.copy();dest[4]=0
                program['calls'].append(dict(object=obj,material='stroke',verb='exit',beat=a,duration=min(1.,b-a),**{'from':previous,'to':dest}))
        program['calls'].sort(key=lambda c:(c['beat'],c['object']))
    # True human note times, carried separately from the LLM aesthetic proposal.
    program['hits']=[[float(times[i]),float(accents[i])] for i in np.flatnonzero(accents>=.5)]
    for obj in range(8):
        at=tm.time_to_beat(max(0,program['duration']-.6));source=states(program,tm.beat_to_time(at))[obj]['values'].copy();dest=source.copy();dest[4]=0
        program['calls'].append(dict(object=obj,material='stroke',verb='exit',beat=at,duration=tm.time_to_beat(program['duration'])-at,**{'from':source,'to':dest}))
    program['calls'].sort(key=lambda c:(c['beat'],c['object']))
    program['supervision']='LLM macro plan + authored action expansion + human chart motion/timing; not direct human effect authorship'
    return validate_program(program)


def aesthetic_labels(record,candidate,path,judgments):
    eligible=[j for j in judgments if j.get('chart_sha256')==record['chart_sha256'] and candidate in j.get('reviewed_candidates',[0,1])]
    for j in eligible:
        expected=j.get('candidate_plan_hashes',{}).get(str(candidate))
        if expected and __import__('hashlib').sha256(Path(path).read_bytes()).hexdigest()!=expected:
            raise ValueError('Human preference points to a changed candidate plan')
    reviewed=[j for j in eligible if j.get('preferred_candidate')==candidate]
    rejected=[j for j in eligible if j.get('preferred_candidate')!=candidate]
    return dict(aesthetic_source='human_selected_llm' if reviewed else 'llm_unreviewed',
        reviewed_intervals=[dict(start_seconds=j['start_seconds'],end_seconds=j['end_seconds']) for j in reviewed],
        rejected_intervals=[dict(start_seconds=j['start_seconds'],end_seconds=j['end_seconds']) for j in rejected])


def prepare_student(manifest, plans, output, reviews=None):
    rows=json.loads(Path(manifest).read_text(encoding='utf-8'));source_root=Path(manifest).parent
    root=Path(output);root.mkdir(parents=True,exist_ok=True);records=[]
    judgments=json.loads(Path(reviews).read_text(encoding='utf-8')) if reviews else []
    for r in rows:
        dest=Path(plans)/r['chart_sha256'][:16]
        if not (dest/'plan-0.json').exists():continue
        analysis=json.loads((dest/'analysis.json').read_text(encoding='utf-8'))
        z=dict(np.load(source_root/r['tensor'],allow_pickle=False))
        for path in sorted(dest.glob('plan-*.json')):
            plan=json.loads(path.read_text(encoding='utf-8'));validate_plan(plan,analysis)
            program=demonstration(plan,analysis,z['motion'],z['times'],z['accent'])
            if r.get('chart') and Path(r['chart']).exists():
                from .importer import NativeChart
                native=NativeChart(json.loads(Path(r['chart']).read_text(encoding='utf-8-sig')),lines_only=True)
                program['hits']=[[native.tempo.beat_to_time(note['startTime']),1.] for _,note in native.notes
                    if not note.get('isFake',0) and 0<=native.tempo.beat_to_time(note['startTime'])<=program['duration']]
                program['hits'].sort()
                program['accent_source']='original native non-fake note timestamps'
            name=r['chart_sha256'][:16]+'-'+path.stem
            (root/(name+'.program.json')).write_text(json.dumps(program,ensure_ascii=False),encoding='utf-8')
            (root/(name+'.plan.json')).write_text(json.dumps(plan,ensure_ascii=False),encoding='utf-8')
            (root/(name+'.analysis.json')).write_text(json.dumps(analysis,ensure_ascii=False),encoding='utf-8')
            boundaries=np.array([[s['start_seconds'],s['end_seconds']] for s in analysis['sections']])
            indices=np.clip(np.searchsorted(boundaries[:,0],z['times'],side='right')-1,0,len(boundaries)-1)
            np.savez_compressed(root/(name+'.npz'),audio=z['audio'].astype(np.float32),motion=z['motion'].astype(np.float32),
                accent=z['accent'].astype(np.float32),times=z['times'].astype(np.float32),beats=z['beats'].astype(np.float32),
                plan=plan_vectors(plan),plan_indices=indices,tempo=z['tempo'],metadata=z['metadata'])
            candidate=plan.get('candidate',0)
            records.append(dict(r,tensor=name+'.npz',program=name+'.program.json',plan=name+'.plan.json',analysis=name+'.analysis.json',
                candidate=candidate,**aesthetic_labels(r,candidate,path,judgments)))
    (root/'manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    return dict(records=len(records),splits={s:sum(r['split']==s for r in records) for s in ['train','val','test']})
