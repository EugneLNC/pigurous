"""Independent fixed-settings renders and anonymized full-song blind materials."""
import argparse,json,subprocess,concurrent.futures,hashlib,shutil
from pathlib import Path
from choreographer.render import render

def verify(path,duration):
 if not path.exists():return False
 try:
  d=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)]))
  if not (abs(float(d['format']['duration'])-duration)<.15 and {s['codec_type'] for s in d['streams']}>={'video','audio'}):return False
  for seek in [[],['-sseof','-0.15']]:
   result=subprocess.run(['ffmpeg','-v','error',*seek,'-i',str(path),'-frames:v','2','-f','null','-'],capture_output=True)
   if result.returncode:return False
  return True
 except Exception:return False
def work(job):
 score,video,audio,width,height,fps=job;duration=json.loads(score.read_text(encoding='utf-8'))['duration']
 if verify(video,duration):return str(video)
 partial=video.with_name(video.stem+'.partial.mp4')
 if verify(partial,duration):partial.replace(video);print('RECOVERED',video.name,flush=True);return str(video)
 render(str(score),str(partial),width,height,fps,audio,preset='veryfast',threads=2)
 if not verify(partial,duration):raise RuntimeError('Incomplete full-song video')
 partial.replace(video);print('RENDERED',video.name,flush=True);return str(video)

def main():
 p=argparse.ArgumentParser();p.add_argument('manifest');p.add_argument('--workers',type=int,default=3);p.add_argument('--blind');p.add_argument('--width',type=int,default=960);p.add_argument('--height',type=int,default=540);p.add_argument('--fps',type=int,default=30);a=p.parse_args()
 rows=json.loads(Path(a.manifest).read_text(encoding='utf-8'));jobs=[]
 for r in rows:
  for variant in ('model','llm'):
   score=Path(r['variants'][variant]['score']);jobs.append((score,Path(str(score).replace('.json','.mp4')),r['audio'],a.width,a.height,a.fps))
 with concurrent.futures.ProcessPoolExecutor(max_workers=min(3,a.workers)) as ex:
  for result in ex.map(work,jobs):pass
 if a.blind:
  import numpy as np
  root=Path(a.blind);public=root/'participants';public.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(42);trials=[];key=[]
  order=rng.permutation(len(rows))
  for trial,index in enumerate(order):
   r=rows[int(index)];swap=bool(rng.integers(2));variants=['llm','model'] if swap else ['model','llm'];paths=[]
   for label,variant in zip(['A','B'],variants):
    source=Path(r['variants'][variant]['score']).with_suffix('.mp4');name=f'{trial+1:02d}-{label}.mp4';shutil.copyfile(source,public/name);paths.append(name)
   trials.append(dict(trial=trial+1,A=paths[0],B=paths[1],criteria=['timing','composition','coherence','diversity','watch_again']))
   key.append(dict(trial=trial+1,song_id=r['song_id'],llm_preferred_label='A' if swap else 'B'))
  (public/'trials.json').write_text(json.dumps(trials,indent=2),encoding='utf-8');(root/'researcher_key.json').write_text(json.dumps(key,ensure_ascii=False,indent=2),encoding='utf-8')
  with (public/'votes-template.csv').open('w',encoding='utf-8',newline='') as f:
   import csv
   writer=csv.writer(f);writer.writerow(['participant','trial','preference','timing','composition','coherence','diversity','watch_again'])
   for t in trials:writer.writerow(['',t['trial'],'','','','','',''])
  (public/'README.txt').write_text('Watch each full A/B pair with audio. Rate timing, composition, coherence, diversity and willingness to watch again. Record A/B/tie. Source labels and answer keys are not included. No ratings have been fabricated.',encoding='utf-8')
  import sys
  subprocess.run([sys.executable,'scripts/make_blind_page.py',str(public)],check=True)
  print('BLIND MATERIALS',len(trials),'full-song pairs',flush=True)

if __name__=="__main__":main()
