import json
from pathlib import Path
from choreographer.choreography import DEFAULT,PROFILE,ACTION_VERSION,primitives,states,score_program
from choreographer.score import TempoMap
p=dict(profile=PROFILE,version=ACTION_VERSION,duration=4,tempo_map=TempoMap.from_bpm([(0,120)]).to_list(),seed=42,
       groups=[dict(id=i,name='geometry-'+str(i)) for i in range(9)],calls=[],hits=[[2,.8]])
for i,material in enumerate(['stroke','orbit','ribbon','tunnel','spark','stars','rose','lissajous','hypotrochoid']):
 a=list(DEFAULT);b=list(DEFAULT);b[0]=.3+i*.05;b[4]=.6;b[6]=.35;b[8]=.2;b[9]=.15
 p['calls'].append(dict(object=i,material=material,verb='enter',beat=0,duration=2,**{'from':a,'to':b}))
 a=b.copy();b=b.copy();b[0]=.8;b[7]=.8
 p['calls'].append(dict(object=i,material=material,verb='translate',beat=3,duration=3,**{'from':a,'to':b}))
p['calls'].sort(key=lambda c:c['beat'])
source=states(p,2.25)[6]['values'].copy();target=source.copy();target[6]=.8
p['calls'].append(dict(object=6,material='lissajous',verb='morph',beat=4.5,duration=1,**{'from':source,'to':target}))
source=states(p,2.25)[1]['values'].copy();target=source.copy();target[6]=.8
p['calls'].append(dict(object=1,material='rose',verb='enter',beat=4.5,duration=1,blend_material=True,**{'from':source,'to':target}))
p['calls'].sort(key=lambda c:c['beat'])
fixture=dict(program=p,samples=[dict(time=t,states=states(p,t),primitives=primitives(p,t)) for t in [0,.6,1.7,2.1,2.25,2.5,3.8]])
Path('tests/fixtures/choreography_parity.json').write_text(json.dumps(fixture),encoding='utf-8')
score_program(p,dict(origin='authored_action_fixture',title='Nine independent material actions')).save('artifacts/choreography-gallery.json')
