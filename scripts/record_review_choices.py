"""Run only with actual human selections; no auto-selected aesthetic votes."""
import argparse,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--choices',required=True);p.add_argument('--output',default='data/choreography-v3/user-reviews.json');a=p.parse_args()
items=json.loads(Path('preview/public/reviews/manifest.json').read_text(encoding='utf-8'))['items'];records=[];seen=set()
for ident,choice in re.findall(r'([1-6])\s*([ABXabx])',a.choices):
 i=int(ident);choice=choice.upper()
 if i in seen:raise ValueError('Duplicate song selection')
 seen.add(i);r=next(x for x in items if x['index']==i)
 records.append(dict(chart_sha256=r['chart_sha256'],preferred_candidate=None if choice=='X' else ['A','B'].index(choice),
     start_seconds=r['start_seconds'],end_seconds=r['end_seconds'],source='explicit_user_review',
     candidate_plan_hashes={c['id']:c['plan_sha256'] for c in r['candidates']}))
if not records:raise ValueError('No actual selections provided')
Path(a.output).write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8');print('Recorded',len(records),'explicit watched-interval choices')
