"""Explicit code/derived-data upload inventory; excludes credentials and raw media."""
import argparse,json,hashlib,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--name',default='student-pilot-v3');p.add_argument('--data',default='data/student-v3');p.add_argument('--train-only',action='store_true');a=p.parse_args()
root=Path.cwd();data=root/a.data;files=list((root/'src/choreographer').glob('*.py'))
files += [root/'scripts/shared_gpu.py',root/'tests/test_student.py',root/'tests/test_choreography.py',root/'tests/fixtures/choreography_parity.json']
records=json.loads((data/'manifest.json').read_text(encoding='utf-8'))
included=[r for r in records if r['split']!='test'] if a.train_only else records
for r in included:files += [data/r[k] for k in ('tensor','program','plan','analysis')]
files += [data/'manifest.json',data/'vocabulary.npy']
files += [data/k for k in ('reviews.json','style-brief.json') if (data/k).exists()]
files += [root/'artifacts/actions-server/best.pt']
files=sorted(set(files))
def destination(f):return ('data/student-v3/'+f.name) if f.is_relative_to(data) else f.relative_to(root).as_posix()
inventory=[dict(path=destination(f),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files]
output=root/'artifacts'/(a.name+'.zip')
with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_STORED) as z:
 for f in files:z.write(f,destination(f))
 z.writestr('upload-inventory.json',json.dumps(inventory,indent=2))
summary=dict(path=str(output),files=len(files),bytes=output.stat().st_size,sha256=hashlib.sha256(output.read_bytes()).hexdigest(),records=len(json.loads((data/'manifest.json').read_text(encoding='utf-8'))),contents='Project Python code, derived audio features, separately marked synthetic action labels and plans, tests, original action checkpoint; no raw audio, API credential or other project files')
(root/'artifacts'/(a.name+'-inventory.json')).write_text(json.dumps(dict(summary=summary,inventory=inventory),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary),flush=True)
