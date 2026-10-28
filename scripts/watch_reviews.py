import subprocess,sys,time,json
from pathlib import Path
for iteration in range(90):
 subprocess.run([sys.executable,'-B','scripts/review_candidates.py'],check=True)
 p=Path('preview/public/reviews/manifest.json')
 if p.exists():
  items=json.loads(p.read_text(encoding='utf-8'))['items']
  if len(items)==6 and all(len(i['candidates'])==2 for i in items):print('ALL SIX REVIEW PAIRS READY',flush=True);break
 time.sleep(20)
else:raise RuntimeError('Review rendering pending: not all remote plans finished')
