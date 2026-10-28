import argparse,json,zipfile,hashlib
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(a.archive) as z:
 inventory=json.loads(z.read('result-inventory.json'))
 for item in inventory:
  dest=(root/item['path']).resolve()
  if not dest.is_relative_to(root.resolve()):raise ValueError('Unsafe result archive path')
  data=z.read(item['path'])
  if hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Result checksum mismatch')
  dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
source=root/'runs/student/best.pt';ck=torch.load(source,map_location='cpu',weights_only=False)
inference={k:ck[k] for k in ('format','model','vocabulary','epoch','parameters','seed','manifest_sha256')}
inference['training_checkpoint_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
inference['plan_memory']=ck.get('plan_memory',False)
torch.save(inference,root/'best.pt')
(root/'download-receipt.json').write_text(json.dumps(dict(verified_files=len(inventory),inference_sha256=hashlib.sha256((root/'best.pt').read_bytes()).hexdigest(),source_sha256=inference['training_checkpoint_sha256']),indent=2))
print(json.dumps(dict(output=str(root/'best.pt'),verified_files=len(inventory),epoch=ck['epoch'])))
