"""Independent bounded remote planning jobs, with a shared atomic token ledger."""
import argparse,json,concurrent.futures
from pathlib import Path
from choreographer.music_analysis import analyze_features
from choreographer.llm_planning import create_plan
p=argparse.ArgumentParser();p.add_argument('--manifest',default='data/actions-v3/manifest.json');p.add_argument('--split',default='train,val');p.add_argument('--workers',type=int,default=3);p.add_argument('--exclude-reviews',action='store_true');p.add_argument('--limit',type=int);a=p.parse_args()
root=Path('data/choreography-v3');rows=json.loads(Path(a.manifest).read_text(encoding='utf-8'));split=a.split.split(',')
excluded=set()
if a.exclude_reviews:
 excluded={r['chart_sha256'] for r in json.loads((root/'review-selection.json').read_text(encoding='utf-8'))}
todo=[r for r in rows if r['split'] in split and r['chart_sha256'] not in excluded and not (root/r['chart_sha256'][:16]/'plan-0.json').exists()]
if a.limit:todo=todo[:a.limit]
def work(r):
 ident=r['chart_sha256'][:16];d=root/ident;d.mkdir(parents=True,exist_ok=True)
 features=r.get('features') or str(Path(a.manifest).parent/(r['audio_hash'][:16]+'.features.npz'))
 analysis=analyze_features(features,r['audio']);(d/'analysis.json').write_text(json.dumps(analysis,ensure_ascii=False,indent=2),encoding='utf-8')
 print('REQUEST',ident,r['split'],flush=True)
 result=create_plan(analysis,d/'plan-0.json')
 print('COMPLETE',ident,json.dumps(result),flush=True)
 return ident
errors=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=min(3,a.workers)) as ex:
 futures={ex.submit(work,r):r for r in todo}
 for future in concurrent.futures.as_completed(futures):
  try:future.result()
  except Exception as e:
   r=futures[future];errors.append(dict(chart_sha256=r['chart_sha256'],reason=str(e)));print('FAILED',r['chart_sha256'][:16],str(e),flush=True)
(root/('planning-errors-'+a.split.replace(',','-')+'.json')).write_text(json.dumps(errors,indent=2),encoding='utf-8')
print('FINISHED',len(todo),'jobs',len(errors),'failures',flush=True)
