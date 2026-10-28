"""Extract only the exact inventoried research bundle into a new owned directory."""
import hashlib,json,zipfile,sys
from pathlib import Path
base=Path('/path/to/music-choreographer');name=sys.argv[1]
if name not in ('student-pilot-v3','student-joint-v3','student-feedback-v3'):raise ValueError('Unexpected research snapshot')
archive=base/(name+'.zip');root=base/'research-choreography-v3'/name
if root.exists():raise RuntimeError('Refuse to overwrite existing snapshot directory')
with zipfile.ZipFile(archive) as z:
 inventory=json.loads(z.read('upload-inventory.json'))
 for item in inventory:
  rel=Path(item['path']);target=(root/rel).resolve()
  if rel.is_absolute() or not target.is_relative_to(root.resolve()):raise ValueError('Unsafe upload inventory')
  data=z.read(item['path'])
  if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Upload hash mismatch')
 root.mkdir(parents=True)
 for item in inventory:
  target=root/item['path'];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(item['path']))
(root/'.choreographer-owned').write_text(name)
(root/'upload-receipt.json').write_text(json.dumps(dict(archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(inventory),directory=str(root)),indent=2))
print(json.dumps(dict(directory=str(root),files=len(inventory))))
