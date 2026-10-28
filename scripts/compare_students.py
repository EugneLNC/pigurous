"""Same-checkpoint full-song generation, isolation ablations and traceable pair manifests."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from choreographer.student_generation import generate
from choreographer.choreography import score_program
from choreographer.score import SceneScore,ScoreRuntime
from choreographer.render import frame,render

p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',default='data/actions-v3/manifest.json');p.add_argument('--plans',default='data/choreography-v3');p.add_argument('--output',default='artifacts/choreography-v3');p.add_argument('--fresh',action='store_true');p.add_argument('--render',action='store_true');p.add_argument('--limit',type=int);p.add_argument('--device',default='cuda');a=p.parse_args()
rows=json.loads(Path(a.manifest).read_text(encoding='utf-8'));root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
oldtest={r['audio_hash'] for r in json.loads(Path('data/actions-v1/manifest.json').read_text(encoding='utf-8')) if r['split']=='test'}
chosen=[r for r in rows if r['split']=='test' and ((r['audio_hash'] not in oldtest) if a.fresh else (r['audio_hash'] in oldtest))]
if a.limit:chosen=chosen[:a.limit]
records=[];web=Path('preview/public/comparison');web.mkdir(parents=True,exist_ok=True)
for r in chosen:
 ident=r['chart_sha256'][:16];plan=Path(a.plans)/ident/'plan-0.json'
 if not plan.exists():print('PENDING PLAN',ident,flush=True);continue
 features=r.get('features') or str(Path(a.manifest).parent/(r['audio_hash'][:16]+'.features.npz'))
 item=dict(song_id=r['song_id'],chart_sha256=r['chart_sha256'],audio=r['audio'],features=features,variants={})
 for source in ('model','llm'):
  out=root/(ident+'-'+source+'.json')
  result=generate(a.checkpoint,features,out,source,plan if source=='llm' else None,r['audio'],a.device)
  item['variants'][source]=dict(score=str(out),**result);print(json.dumps(result),flush=True)
  if a.render:render(str(out),str(root/(ident+'-'+source+'.mp4')),960,540,30,r['audio'])
  if not a.fresh:shutil.copyfile(out,web/out.name)
  runtime=ScoreRuntime(SceneScore.load(out));grid=__import__('PIL.Image',fromlist=['Image']).new('RGB',(960,540))
  for i,at in enumerate(np.linspace(runtime.score.duration*.15,runtime.score.duration*.8,4)):
   im=frame(runtime,float(at),480,270);grid.paste(im,((i%2)*480,(i//2)*270))
  grid.save(root/(ident+'-'+source+'-contact.png'))
 for condition in ('zero','shuffle'):
  out=root/(ident+'-llm-'+condition+'.json')
  generate(a.checkpoint,features,out,'llm',plan,r['audio'],a.device,condition=condition)
 # Swap the visual plan while preserving this track's observed boundaries and audio.
 available=[x for x in chosen if x['audio_hash']!=r['audio_hash'] and (Path(a.plans)/x['chart_sha256'][:16]/'plan-0.json').exists()]
 def tempo(x):
  analysis=json.loads((Path(a.plans)/x['chart_sha256'][:16]/'analysis.json').read_text(encoding='utf-8'))
  from choreographer.score import TempoMap
  tm=TempoMap(analysis['tempo_map']);return (tm.time_to_beat(analysis['duration'])-tm.time_to_beat(0))*60/analysis['duration']
 other=min(available,key=lambda x:abs(tempo(x)-tempo(r))) if available else None
 if other:
  target=json.loads(plan.read_text(encoding='utf-8'));foreign=json.loads((Path(a.plans)/other['chart_sha256'][:16]/'plan-0.json').read_text(encoding='utf-8'))
  for i,s in enumerate(target['sections']):
   donor_index=min(len(foreign['sections'])-1,int(i/len(target['sections'])*len(foreign['sections'])))
   donor=foreign['sections'][donor_index]
   for key in ('motif_id','materials','verbs','role','layout','density','intensity','scale','hue','anchor','variation','contrast'):s[key]=donor[key]
  target['source_plan_audio_sha256']=foreign['audio_sha256'];target['source']='declared_plan_transfer_ablation'
  generate(a.checkpoint,features,root/(ident+'-plan-swap.json'),'llm',target,r['audio'],a.device)
  other_features=other.get('features') or str(Path(a.manifest).parent/(other['audio_hash'][:16]+'.features.npz'))
  generate(a.checkpoint,features,root/(ident+'-audio-swap.json'),'llm',plan,r['audio'],a.device,condition='swap',conditioning_features=other_features)
 records.append(item)
 (root/'manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
(root/'manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
if not a.fresh:
 (web/'manifest.json').write_text(json.dumps([dict(song_id=r['song_id'],model='/comparison/'+Path(r['variants']['model']['score']).name,llm='/comparison/'+Path(r['variants']['llm']['score']).name) for r in records],ensure_ascii=False,indent=2),encoding='utf-8')
print('GENERATED',len(records),'whole-song pairs',flush=True)
