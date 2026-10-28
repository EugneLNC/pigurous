"""Measure imitation only inside approved clips; never turn it into a blind vote."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from choreographer.student_generation import load_student
from choreographer.student_training import StudentDataset,collate

p=argparse.ArgumentParser();p.add_argument('--manifest',default='data/student-feedback-v3/manifest.json')
p.add_argument('--checkpoint',action='append',required=True);p.add_argument('--output',required=True);a=p.parse_args()
d=StudentDataset(a.manifest,'train');indices=[]
for i,(_,_,_,synthetic) in enumerate(d.index):
    if synthetic and (d[i]['event_weights'][d[i]['event_mask']]==1).any():indices.append(i)
results=[]
for checkpoint in a.checkpoint:
    model,ck=load_student(checkpoint,'cpu');sources=[]
    for teacher in [False,True]:
        values={k:[] for k in ['material_ce','verb_ce','object_ce','parameter_mae']};events=0
        with torch.inference_mode():
            for i in indices:
                b=collate([d[i]]);out=model.forward_actions(b,teacher);mask=b['event_mask']&(b['event_weights']==1);events+=int(mask.sum())
                for k in ['material','verb','object']:
                    values[k+'_ce'].extend(F.cross_entropy(out[k].transpose(1,2),b[k],reduction='none')[mask].tolist())
                pm=b['parameter_mask']*mask[...,None]
                errors=((out['parameters']-b['parameters']).abs()*pm).sum(-1)/pm.sum(-1).clamp_min(1)
                values['parameter_mae'].extend(errors[mask].tolist())
        sources.append(dict(plan_source='demonstration' if teacher else 'student_prediction',complete_watched_events=events,
                            **{k:float(np.mean(v)) for k,v in values.items()}))
    results.append(dict(checkpoint=checkpoint,epoch=ck['epoch'],watched_reconstruction=sources))
Path(a.output).write_text(json.dumps(dict(results=results,limitation='Reconstruction of 120 seconds selected for training is an imitation diagnostic, not held-out aesthetics or a human preference rate.'),indent=2),encoding='utf-8')
print('Measured',len(results),'checkpoints on',len(indices),'watched windows')
