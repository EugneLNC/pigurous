from pathlib import Path
import json,math
from PIL import Image,ImageDraw
from choreographer.score import SceneScore,ScoreRuntime,Entity,Event,TempoMap
from choreographer.render import frame
from choreographer.choreography import score_program, DEFAULT, MATERIALS, PROFILE, ACTION_VERSION
from make_demo import make

root=Path(__file__).resolve().parents[1]
public=root/'preview'/'public';media=root/'docs'/'media'
public.mkdir(parents=True,exist_ok=True);media.mkdir(parents=True,exist_ok=True)
orbit=make();orbit.save(public/'demo.json')
controls=[0.8,0.4,0.3,0.2,0,12,1,0,0]
p=dict(profile='mathematical-v2',duration=8,seed=42,hue=.54,times=[0,8],controls=[controls,controls],poses=[[0,0,0,1],[0,0,0,1]],hits=[])
mathscore=SceneScore(8,TempoMap.from_bpm([(0,120)]).to_list(),metadata=dict(title='Harmonic Rose / Authored execution demo',origin='authored_primitive_gallery',render_profile='mathematical-v2',presentation=p))
for i in range(5):mathscore.entities.append(Entity(i,'circle',width=1,height=1));mathscore.events.append(Event(0,'create',i))
mathscore.save(public/'mathematical-demo.json')
program=dict(profile=PROFILE,version=ACTION_VERSION,duration=8,tempo_map=TempoMap.from_bpm([(0,120)]).to_list(),seed=42,groups=[dict(id=i,name=material) for i,material in enumerate(MATERIALS)],calls=[],hits=[])
for i,material in enumerate(MATERIALS):
    a=list(DEFAULT);b=list(DEFAULT)
    b[0]=.2+(i%3)*.3;b[1]=.2+(i//3)*.3;b[2]=.12;b[4]=.55;b[5]=.48+i*.025;b[8]=.02;b[9]=0
    a[:]=b;a[4]=0
    program['calls'].append(dict(object=i,material=material,verb='enter',beat=0,duration=1,**{'from':a,'to':b}))
    c=b.copy();c[3]=.8
    program['calls'].append(dict(object=i,material=material,verb='rotate',beat=2,duration=10,**{'from':b,'to':c}))
program['calls'].sort(key=lambda c:c['beat'])
choreo=score_program(program,dict(origin='authored_action_fixture',title='Nine materials / Authored action gallery'))
choreo.save(public/'choreography-demo.json')
examples=[('ORBIT / EVENT TIMELINE',orbit,6),('HARMONIC ROSE / EXECUTOR',mathscore,3),('NINE MATERIALS / ACTIONS',choreo,2.4)]
gallery=Image.new('RGB',(1440,510),(8,12,22))
for i,(name,score,t) in enumerate(examples):
 tile=frame(ScoreRuntime(score),t,480,450);d=ImageDraw.Draw(tile);d.rectangle((0,0,480,35),fill=(8,12,22));d.text((18,13),name,fill=(185,230,220));gallery.paste(tile,(i*480,0))
d=ImageDraw.Draw(gallery);d.text((20,475),'AUTHORED RENDERER EXAMPLES - NOT TRAINED MODEL RESULTS - NO THIRD-PARTY AUDIO',fill=(132,170,188));gallery.save(media/'renderer-gallery.png')
rt=ScoreRuntime(mathscore);frames=[]
for i in range(48):
 image=frame(rt,.25+i/8,720,405);d=ImageDraw.Draw(image);d.rectangle((0,0,720,25),fill=(8,12,22));d.text((14,8),'AUTHORED MATHEMATICAL EXECUTOR / SILENT / NOT MODEL OUTPUT',fill=(170,218,218));frames.append(image.quantize(colors=128))
frames[0].save(media/'authored-mathematical-demo.gif',save_all=True,append_images=frames[1:],duration=125,loop=0,optimize=True)
print('Created three authored SceneScore examples, gallery and animated GIF.')
