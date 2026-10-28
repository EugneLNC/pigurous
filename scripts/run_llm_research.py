"""Cache real remote plans; raw audio/features never enter the API request."""
import argparse,json
from pathlib import Path
import numpy as np
from choreographer.music_analysis import analyze_features
from choreographer.llm_planning import create_plan

p=argparse.ArgumentParser()
p.add_argument('--manifest',default='data/actions-v1/manifest.json')
p.add_argument('--output',default='data/choreography-v3')
p.add_argument('--initial',action='store_true')
a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
rows=json.loads(Path(a.manifest).read_text(encoding='utf-8'))
training=[r for r in rows if r['split']=='train']
descriptors=[]
for r in training:
    with np.load(Path(a.manifest).parent/r['tensor']) as z:
        m=z['motion'];descriptors.append(np.r_[m.mean(0),m.std(0)])
d=np.asarray(descriptors); d=(d-d.mean(0))/np.maximum(d.std(0),.05)
selected=[int(np.argmax(np.linalg.norm(d,axis=1)))]
while len(selected)<min(6,len(training)):
    distance=((d[:,None]-d[selected][None])**2).sum(-1).min(-1);distance[selected]=-1
    selected.append(int(np.argmax(distance)))
review_ids={training[i]['chart_sha256'] for i in selected}
(root/'review-selection.json').write_text(json.dumps([training[i] for i in selected],ensure_ascii=False,indent=2),encoding='utf-8')
todo=([training[i] for i in selected]+[r for r in rows if r['split']=='test']) if a.initial else rows
progress=[]
for r in todo:
    ident=r['chart_sha256'][:16];dest=root/ident;dest.mkdir(exist_ok=True)
    features=r.get('features')
    if not features or not Path(features).exists():
        features=str(Path(a.manifest).parent/(r['audio_hash'][:16]+'.features.npz'))
    analysis=analyze_features(features,r['audio'])
    (dest/'analysis.json').write_text(json.dumps(analysis,ensure_ascii=False,indent=2),encoding='utf-8')
    count=2 if r['chart_sha256'] in review_ids else 1
    for candidate in range(count):
        output=dest/('plan-'+str(candidate)+'.json')
        if output.exists():continue
        print('REQUEST',ident,'candidate',candidate,'sections',len(analysis['sections']),flush=True)
        result=create_plan(analysis,output,candidate=candidate)
        print('COMPLETE',ident,candidate,json.dumps(result),flush=True)
    progress.append(dict(chart_sha256=r['chart_sha256'],song_id=r['song_id'],split=r['split'],directory=ident))
    (root/'planning-progress.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2),encoding='utf-8')
