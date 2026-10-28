"""Auxiliary chart reconstruction and generated control diagnostics, not aesthetic scores."""
import argparse,json,collections
from pathlib import Path
import numpy as np,torch
from choreographer.student_generation import load_student
from choreographer.student_training import StudentDataset,collate
from choreographer.choreography import states,MATERIALS,VERBS

p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',default='data/student-v3/manifest.json');p.add_argument('--scores',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
model,ck=load_student(a.checkpoint,a.device)
training_rows=[r for r in json.loads(Path(a.manifest).read_text(encoding='utf-8')) if r['split']=='train']
motion_total=np.zeros(12);motion_frames=0
for r in training_rows:
 with np.load(Path(a.manifest).parent/r['tensor'],allow_pickle=False) as z:
  motion_total+=z['motion'].sum(0);motion_frames+=len(z['motion'])
motion_prior=torch.tensor(motion_total/max(1,motion_frames),dtype=torch.float32,device=a.device)
results=[]
with torch.inference_mode():
 for split in ['val','test']:
  d=StudentDataset(a.manifest,split);mse=[];baseline=[];plan_mse=[];counts=[];parameter_mae=[];accuracy={k:[] for k in ['material','verb','object']}
  for i in range(0,len(d),2):
   row=d[i];b={k:v.to(a.device) for k,v in collate([row]).items()};pred=model.forward_actions(b,False)
   mse.append(float(((pred['motion']-b['motion'])**2)[b['mask']].mean()))
   baseline.append(float(((motion_prior-b['motion'])**2)[b['mask']].mean()))
   plan_mse.append(float(((pred['structure']-b['whole_plan'])**2)[~b['whole_mask']].mean()))
   counts.append(float((pred['count'].argmax(-1)-b['count']).abs().float().mean()))
   em=b['event_mask'];pm=b['parameter_mask']*em[...,None]
   for k in accuracy:
    if em.any():accuracy[k].append(float((pred[k].argmax(-1)[em]==b[k][em]).float().mean()))
   if pm.sum():parameter_mae.append(float(((pred['parameters']-b['parameters']).abs()*pm).sum()/pm.sum()))
  results.append(dict(split=split,windows=len(mse),auxiliary_motion_mse=float(np.mean(mse)) if mse else None,
    constant_training_mean_motion_mse=float(np.mean(baseline)) if baseline else None,synthetic_plan_mse=float(np.mean(plan_mse)) if plan_mse else None,
    synthetic_action_reconstruction=dict(plan_source='student_prediction',count_mae=float(np.mean(counts)) if counts else None,
       field_accuracy={k:float(np.mean(v)) if v else None for k,v in accuracy.items()},active_parameter_mae=float(np.mean(parameter_mae)) if parameter_mae else None)))
def descriptor(program):
 calls=program['calls'];mat=np.array([sum(c['material']==x for c in calls) for x in MATERIALS],float);verbs=np.array([sum(c['verb']==x for c in calls) for x in VERBS],float)
 params=np.array([c['to'] for c in calls]);return np.r_[mat/max(1,len(calls)),verbs/max(1,len(calls)),params.mean(0),params.std(0)] if len(calls) else np.zeros(len(MATERIALS)+len(VERBS)+24)
def trajectory(program):
 return np.array([[states(program,t)[g['id']]['values'] for g in program['groups']] for t in np.linspace(0,program['duration'],120)],np.float32)
train=[];sources=[];train_trajectories=[]
root=Path(a.manifest).parent
for r in json.loads(Path(a.manifest).read_text(encoding='utf-8')):
 if r['split']=='train':
  program=json.loads((root/r['program']).read_text(encoding='utf-8'));train.append(descriptor(program));sources.append(r['program']);train_trajectories.append(trajectory(program))
records=[]
for path in sorted(Path(a.scores).glob('*-model.json')):
 paired=path.with_name(path.name.replace('-model.json','-llm.json'))
 if not paired.exists():continue
 score=json.loads(path.read_text(encoding='utf-8'));llm=json.loads(paired.read_text(encoding='utf-8'));program=score['metadata']['presentation'];other=llm['metadata']['presentation']
 def track(p):return trajectory(p)
 neighbors=[]
 if train:
  distance=np.linalg.norm(np.asarray(train)-descriptor(program),axis=1);j=int(distance.argmin());neighbors=[dict(source=sources[j],distance=float(distance[j]))]
  trajectory_distance=abs(np.asarray(train_trajectories)-track(program)).mean((1,2,3));k=int(trajectory_distance.argmin())
  trajectory_neighbor=dict(source=sources[k],normalized_time_control_mae=float(trajectory_distance[k]),samples=120)
 else:trajectory_neighbor=None
 variants={}
 for suffix in ['llm-zero','llm-shuffle','plan-swap','audio-swap']:
  candidate=path.with_name(path.name.replace('-model.json','-'+suffix+'.json'))
  if candidate.exists():
   wrong=json.loads(candidate.read_text(encoding='utf-8'))['metadata']['presentation'];variants[suffix]=float(abs(track(other)-track(wrong)).mean())
 records.append(dict(song=path.name,actions_model=len(program['calls']),actions_llm=len(other['calls']),
    model_materials=dict(collections.Counter(c['material'] for c in program['calls'])),model_verbs=dict(collections.Counter(c['verb'] for c in program['calls'])),
    planner_control_mae=float(abs(track(program)-track(other)).mean()),ablations_control_mae=variants,nearest_training_descriptors=neighbors))
 records[-1]['nearest_training_trajectory']=trajectory_neighbor
 for label,p in [('model',program),('llm',other)]:
  controls=track(p);active=controls[:,:,4]>.003
  records[-1][label+'_motion_summary']=dict(mean_active_groups=float(active.sum(1).mean()),position_span_x=float(np.ptp(controls[:,:,0],axis=0).mean()),
      position_span_y=float(np.ptp(controls[:,:,1],axis=0).mean()),final_visible_groups=int(active[-1].sum()),
      material_usage=dict(collections.Counter(c['material'] for c in p['calls'])),verb_usage=dict(collections.Counter(c['verb'] for c in p['calls'])))
out=dict(checkpoint_epoch=ck['epoch'],teacher_forced_auxiliary=results,generated=records,
    limitation='Auxiliary 12-field chart reconstruction and synthetic action/plan prediction are not scores for professional choreography. Control trajectory and descriptor retrieval screen copying but cannot prove originality. Ablation differences establish sensitivity, not beneficial audio or plan use. Existing and sequentially evaluated test songs are exploratory. Training choices are not blind-test votes; no human blind preferences have been entered.')
Path(a.output).write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(dict(records=len(records),output=a.output)))
