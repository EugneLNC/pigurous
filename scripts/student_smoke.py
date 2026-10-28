import json,tempfile
from pathlib import Path
import numpy as np,torch
from choreographer.student_model import ChoreographyStudent,student_loss
from choreographer.student_training import StudentDataset,collate
from choreographer.action_training import device_setup
device='cuda' if torch.cuda.is_available() else 'cpu';device_setup(device)
dataset=StudentDataset('data/student-v3/manifest.json','train')
model=ChoreographyStudent(np.load('data/actions-v1/vocabulary.npy')).to(device)
batch={k:v.to(device) for k,v in collate([dataset[0],dataset[2]]).items()}
with torch.autocast(device_type='cuda' if device=='cuda' else 'cpu',dtype=torch.bfloat16,enabled=device=='cuda'):
 loss=student_loss(model.forward_actions(batch,True),batch)
loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
assert torch.isfinite(loss)
out=dict(device=device,parameters=sum(p.numel() for p in model.parameters()),loss=float(loss.detach()),peak_gib=torch.cuda.max_memory_allocated()/1024**3 if device=='cuda' else None)
Path('artifacts/student-smoke.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
