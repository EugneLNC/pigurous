"""Save new inference-only plans; preserve original plans and all song splits."""
import argparse,json,concurrent.futures,shutil
from pathlib import Path
from choreographer.llm_planning import create_plan

p=argparse.ArgumentParser();p.add_argument('--limit',type=int);p.add_argument('--output',default='data/choreography-style-eval-v3');a=p.parse_args()
source=Path('data/choreography-v3');root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
brief=json.loads((source/'style-brief.json').read_text(encoding='utf-8'))['brief']
rows=[r for r in json.loads(Path('data/actions-v3/manifest.json').read_text(encoding='utf-8')) if r['split']=='test' and not (root/r['chart_sha256'][:16]/'plan-0.json').exists()]
if a.limit:rows=rows[:a.limit]
def work(r):
    d=root/r['chart_sha256'][:16];d.mkdir(exist_ok=True)
    shutil.copyfile(source/r['chart_sha256'][:16]/'analysis.json',d/'analysis.json')
    analysis=json.loads((d/'analysis.json').read_text(encoding='utf-8'))
    print('REQUEST',r['chart_sha256'][:16],flush=True)
    result=create_plan(analysis,d/'plan-0.json',style_brief=brief)
    print('COMPLETE',r['chart_sha256'][:16],json.dumps(result),flush=True)
    return result
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:list(ex.map(work,rows))
(root/'source.json').write_text(json.dumps(dict(inference_only=True,style_source='explicit_user_statement',training_labels_unchanged=True,plan_root=str(root)),indent=2))
