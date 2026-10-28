"""Render two twenty-second demonstrations per training song for actual human review."""
import json,subprocess,math,shutil,argparse
from pathlib import Path
import numpy as np
from choreographer.choreography_data import demonstration
from choreographer.choreography import score_program
from choreographer.score import ScoreRuntime
from choreographer.render import frame
from choreographer.importer import NativeChart

p=argparse.ArgumentParser();p.add_argument('--round',type=int,choices=[0,1,2],default=0);p.add_argument('--indices',default='1,2,3,4,5,6');args=p.parse_args()
root=Path('data/choreography-v3');original=Path('preview/public/reviews');dest=original/('round'+str(args.round)) if args.round else original;dest.mkdir(parents=True,exist_ok=True)
previous=json.loads((original/'manifest.json').read_text(encoding='utf-8'))['items'] if (original/'manifest.json').exists() else []
rows=json.loads((root/'review-selection.json').read_text(encoding='utf-8'));items=[]
for index,r in enumerate(rows):
 if index+1 not in {int(i) for i in args.indices.split(',')}:continue
 d=root/r['chart_sha256'][:16]
 if not (d/'analysis.json').exists():continue
 z=dict(np.load(Path('data/actions-v1')/r['tensor'],allow_pickle=False))
 analysis=json.loads((d/'analysis.json').read_text(encoding='utf-8'));length=min(20.,analysis['duration'])
 starts=np.arange(0,max(1,analysis['duration']-length),10.)
 variability=[]
 for start in starts:
  mask=(z['times']>=start)&(z['times']<start+length);variability.append(float(z['motion'][mask,1:7].std(0).mean()))
 old=next((i for i in previous if i['index']==index+1),None)
 start=old['start_seconds'] if old else float(starts[int(np.argmax(variability))]);item=dict(index=index+1,round=args.round,song_id=r['song_id'],chart_sha256=r['chart_sha256'],
     start_seconds=start,end_seconds=start+length,candidates=[])
 for candidate in range(args.round*2,args.round*2+2):
  path=d/('plan-'+str(candidate)+'.json')
  if not path.exists():continue
  plan=json.loads(path.read_text(encoding='utf-8'));program=demonstration(plan,analysis,z['motion'],z['times'],z['accent'])
  native=NativeChart(json.loads(Path(r['chart']).read_text(encoding='utf-8-sig')),lines_only=True)
  program['hits']=[[native.tempo.beat_to_time(n['startTime']),1.] for _,n in native.notes if not n.get('isFake',0) and 0<=native.tempo.beat_to_time(n['startTime'])<=program['duration']]
  program['hits'].sort()
  name=str(index+1)+('R'+str(args.round) if args.round else '')+('A' if candidate%2==0 else 'B');out=dest/(name+'.mp4')
  score=score_program(program,dict(origin='llm_demonstration',title=r['song_id']+' / candidate '+str(candidate)),plan)
  score.save(dest/(name+'.json'))
  if not out.exists():
   print('RENDER',name,'start',start,'duration',length,flush=True)
   cmd=['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','960x540','-r','24','-i','-',
        '-ss',str(start),'-i',r['audio'],'-map','0:v','-map','1:a:0','-c:a','aac','-c:v','libx264','-preset','veryfast','-threads','2','-pix_fmt','yuv420p','-t',str(length),'-movflags','+faststart',str(out)]
   rt=ScoreRuntime(score)
   with subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as process:
    for j in range(math.ceil(length*24)):process.stdin.write(frame(rt,start+j/24,960,540).tobytes())
    process.stdin.close();error=process.stderr.read();code=process.wait()
    if code:raise RuntimeError(error.decode(errors='replace'))
   frame(rt,start+length/2,960,540).save(dest/(name+'.png'))
  probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(out)]))
  if {s['codec_type'] for s in probe['streams']}!={'video','audio'} or abs(float(probe['format']['duration'])-length)>.15:raise ValueError('Invalid candidate video')
  url='/reviews/'+('round'+str(args.round)+'/' if args.round else '')+name
  item['candidates'].append(dict(id=name,candidate=candidate,video=url+'.mp4',image=url+'.png',plan_sha256=__import__('hashlib').sha256(path.read_bytes()).hexdigest()))
 items.append(item)
(dest/'manifest.json').write_text(json.dumps(dict(source='LLM macro plan + authored expansion; preference demonstrations, not trained student results',round=args.round,items=items),ensure_ascii=False,indent=2),encoding='utf-8')
print('READY',len(items),'pairs',sum(len(i['candidates']) for i in items),'clips',flush=True)
