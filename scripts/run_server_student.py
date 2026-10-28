"""Owned-directory server training, one allowed GPU subprocess at a time."""
import json,os,subprocess,sys,zipfile,hashlib
from pathlib import Path
name=sys.argv[1]
if name not in ('student-pilot-v3','student-joint-v3','student-feedback-v3'):raise ValueError('Unexpected snapshot')
base=Path('/path/to/music-choreographer');root=base/'research-choreography-v3'/name
if not (root/'.choreographer-owned').is_file():raise RuntimeError('Owned research marker required')
python=str(base/'.venv/bin/python');env=dict(os.environ,PYTHONPATH=str(root/'src'),PYTHONDONTWRITEBYTECODE='1')
art=root/'artifacts';art.mkdir(exist_ok=True)
test=[python,'-m','pytest','-c','/dev/null','-p','no:cacheprovider','tests/test_student.py','tests/test_choreography.py','-q']
result=subprocess.run(test,cwd=root,env=env,capture_output=True,text=True)
(art/'tests.log').write_text(result.stdout+result.stderr)
if result.returncode:raise RuntimeError('Server tests failed: '+result.stdout+result.stderr)
print(result.stdout,flush=True)
epochs=8 if name=='student-pilot-v3' else 40
if name=='student-feedback-v3':epochs=12
initialize='artifacts/actions-server/best.pt'
if name=='student-joint-v3':initialize=str(base/'research-choreography-v3/student-pilot-v3/runs/student/best.pt')
if name=='student-feedback-v3':initialize=str(base/'research-choreography-v3/student-joint-v3/runs/student/best.pt')
for total,resume in [(2,False),(epochs,True)]:
 command=[python,'scripts/shared_gpu.py','--required-mb','4096','--',python,'-m','choreographer','student','train','data/student-v3/manifest.json',
     '-o','runs/student','--device','cuda','--epochs',str(total),'--batch-size','2','--accumulate','16']
 if name=='student-feedback-v3':command+=['--review-boost','8']
 if resume:command+=['--resume','runs/student/last.pt']
 else:command+=['--initialize',initialize]
 print('RUN',' '.join(command),flush=True)
 with (art/('train-resume.log' if resume else 'train-first.log')).open('w') as log:
  result=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
 if result.returncode:raise RuntimeError('Student training failed; inspect owned experiment log')
files=[root/'runs/student/best.pt',root/'runs/student/last.pt',root/'runs/student/summary.json',root/'runs/student/metrics.jsonl',root/'upload-receipt.json',*art.glob('*.log')]
receipt=[dict(path=p.relative_to(root).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in files]
out=root/'student-result.zip'
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_STORED) as z:
 for p in files:z.write(p,p.relative_to(root).as_posix())
 z.writestr('result-inventory.json',json.dumps(receipt,indent=2))
print(json.dumps(dict(result=str(out),summary=json.loads((root/'runs/student/summary.json').read_text()))),flush=True)
