"""Regenerate only rejected candidates using the user's stated visual direction."""
import json,concurrent.futures
from pathlib import Path
from choreographer.llm_planning import create_plan

root=Path('data/choreography-v3')
brief='Human direction: prefer smooth, natural transitions like the watched candidate 1A. Keep coherent object identity, meaningful translation and gradual grouping across section boundaries. Retain pauses, purposeful contrasts and sharp musical accents when justified. Avoid arbitrary replacements or motion that consists only of rotation. This is a visual preference, not a fixed song or genre template.'
(root/'style-brief.json').write_text(json.dumps(dict(source='explicit_user_statement',reference='1A, seconds 10-30',brief=brief),ensure_ascii=False,indent=2),encoding='utf-8')
reviews=json.loads((root/'user-reviews.json').read_text(encoding='utf-8'))
tasks=[(r,c) for r in reviews if r['preferred_candidate'] is None for c in (2,3)]
def work(task):
    r,c=task;d=root/r['chart_sha256'][:16];out=d/('plan-'+str(c)+'.json')
    if out.exists():return
    print('REQUEST',r['chart_sha256'][:16],c,flush=True)
    result=create_plan(json.loads((d/'analysis.json').read_text(encoding='utf-8')),out,candidate=c,style_brief=brief)
    print('COMPLETE',c,json.dumps(result),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(work,tasks))
