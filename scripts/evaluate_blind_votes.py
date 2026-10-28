"""Read actual anonymized votes with a private mapping and bootstrap by song."""
import argparse,csv,json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('votes');p.add_argument('--key',required=True);p.add_argument('-o','--output',required=True);a=p.parse_args()
key={r['trial']:r for r in json.loads(Path(a.key).read_text(encoding='utf-8'))};groups={};seen=set()
with Path(a.votes).open(encoding='utf-8-sig',newline='') as f:
    for row in csv.DictReader(f):
        if not row.get('preference','').strip():continue
        trial=int(row['trial']);participant=row['participant'].strip();choice=row['preference'].strip().upper()
        if trial not in key or not participant or choice not in ('A','B','TIE') or (trial,participant) in seen:
            raise ValueError('Unknown trial, invalid preference or duplicate vote')
        seen.add((trial,participant));r=key[trial]
        value=.5 if choice=='TIE' else float(choice==r['llm_preferred_label'])
        groups.setdefault(r['song_id'],[]).append(value)
report=dict(comparison='LLM whole-song plan vs same student with predicted plan',acceptance_met=False,status='pending_actual_ratings')
if groups:
    means=np.array([np.mean(v) for v in groups.values()]);rng=np.random.default_rng(42)
    ci=np.quantile(rng.choice(means,(10000,len(means)),replace=True).mean(1),[.025,.975]);rate=float(means.mean())
    report.update(status='rated',songs=len(means),votes=len(seen),llm_preference_rate=rate,song_bootstrap_ci95=ci.tolist(),
                  acceptance_met=len(means)>=20 and rate>=.6 and ci[0]>.5)
report['scope']='Training clip choices are excluded. This tests plan source; it does not substitute for full-vs-local-model aesthetic validation.'
Path(a.output).write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
