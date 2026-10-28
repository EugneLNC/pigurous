import copy,json
import numpy as np
import pytest
from choreographer.choreography import validate_program,primitives,states,validate_plan,plan_vectors,vectors_plan,score_program
from choreographer.llm_planning import Budget,create_plan

def program():
    return json.loads(open('tests/fixtures/choreography_parity.json',encoding='utf-8').read())['program']

def test_group_motion_seek_invalid_reference_and_time():
    p=program(); before=primitives(p,1.7);primitives(p,3.9);assert primitives(p,1.7)==before
    assert len({v[0] for v in before})==9
    assert states(p,2.0)[0]['values'][0]!=states(p,2.0)[8]['values'][0]
    bad=copy.deepcopy(p);bad['calls'][0]['object']=100
    with pytest.raises(ValueError,match='reference'):validate_program(bad)
    bad=copy.deepcopy(p);bad['calls'][-1]['duration']=100
    with pytest.raises(ValueError,match='outside'):validate_program(bad)
    score_program(p).validate()

def test_step_change_and_early_accents():
    p=program();assert not primitives(p,0)
    p['calls'][0]['curve']='step'
    assert states(p,.9)[0]['values'][4]==0
    assert states(p,1.0)[0]['values'][4]>.5

def test_different_channels_animate_in_parallel_and_hold_preserves_state():
    p=program();initial=p['calls'][0]['to'].copy();target=initial.copy();target[2]=.9
    p['calls'].append(dict(object=0,material='stroke',verb='scale',beat=3.5,duration=2,**{'from':initial,'to':target}))
    p['calls'].append(dict(object=0,material='stroke',verb='hold',beat=4,duration=1,**{'from':initial,'to':initial}))
    p['calls'].sort(key=lambda c:c['beat'])
    s=states(p,2.3)[0]['values']
    assert s[0]>initial[0] and s[2]>initial[2]

def test_cross_material_morph_crossfades_without_instant_replacement():
    p=program();s=states(p,2.5)[6]
    assert s['morph_from']=='rose' and abs(s['morph_progress']-.5)<1e-7
    plain=copy.deepcopy(p);plain['calls']=[c for c in plain['calls'] if c['verb']!='morph']
    assert len([v for v in primitives(p,2.5) if v[0]==6])==2*len([v for v in primitives(plain,2.5) if v[0]==6])

def test_optional_enter_blend_preserves_legacy_enter_semantics():
    p=program();s=states(p,2.5)[1]
    assert s['morph_from']=='orbit' and abs(s['morph_progress']-.5)<1e-7
    legacy=copy.deepcopy(p)
    for c in legacy['calls']:c.pop('blend_material',None)
    assert 'morph_from' not in states(legacy,2.5)[1]

def test_zero_duration_action_applies_at_its_exact_start():
    p=program();p['calls'][0]['duration']=0
    assert states(p,0)[0]['values'][4]==p['calls'][0]['to'][4]

def example():
    analysis=dict(audio_sha256='test',duration=8,sections=[dict(index=0,start_beat=0,end_beat=16)])
    plan=dict(version='choreography-plan-v1',audio_sha256='test',sections=[dict(index=0,start_beat=0,end_beat=16,motif_id='a',materials=['ribbon'],verbs=['draw','gather'],role='build',layout='diagonal',density=.5,intensity=.6,scale=.6,hue=.3,anchor=[.2,-.1],variation=.4,contrast=.7)])
    return analysis,plan

def test_plan_cache_no_network_and_boundaries(tmp_path):
    analysis,plan=example();calls=[]
    def request(payload):
        calls.append(payload);return dict(text=json.dumps(plan),id='request',usage=dict(input_tokens=100,output_tokens=20,output_tokens_details=dict(reasoning_tokens=10)))
    create_plan(analysis,tmp_path/'p.json',cache=tmp_path/'cache',ledger=tmp_path/'ledger',request=request)
    create_plan(analysis,tmp_path/'q.json',cache=tmp_path/'cache',ledger=tmp_path/'ledger',request=lambda p:pytest.fail('cache made API request'))
    assert len(calls)==1 and Budget(tmp_path/'ledger').summary()['used_or_reserved']==120
    np.testing.assert_allclose(plan_vectors(vectors_plan(plan_vectors(plan),analysis))[:,30:38],plan_vectors(plan)[:,30:38],atol=1e-6)
    plan['sections'][0]['end_beat']=17
    with pytest.raises(ValueError,match='boundaries'):validate_plan(plan,analysis)

def test_budget_reserves_unknown_usage_and_retries(tmp_path):
    b=Budget(tmp_path/'l',limit=100);ident=b.reserve(70);b.finish(ident,None)
    with pytest.raises(ValueError,match='exhausted'):b.reserve(31)
    b.finish(ident,dict(total_tokens=20));b.reserve(80)
    assert b.summary()['used_or_reserved']==100
    analysis,_=example();calls=[]
    def fail(p):calls.append(p);raise RuntimeError('network failed')
    with pytest.raises(RuntimeError,match='three'):create_plan(analysis,tmp_path/'p',cache=tmp_path/'c',ledger=tmp_path/'retry',request=fail)
    assert len(calls)==3 and Budget(tmp_path/'retry').summary()['unknown_usage']==3

def test_invalid_json_types_retry_and_style_changes_cache_key(tmp_path):
    analysis,plan=example();calls=[]
    def request(payload):
        calls.append(payload)
        return dict(text=json.dumps([] if len(calls)==1 else plan),usage=dict(total_tokens=10),id=str(len(calls)))
    first=create_plan(analysis,tmp_path/'a',cache=tmp_path/'c',ledger=tmp_path/'l',request=request)
    assert len(calls)==2 and first['budget']['used_or_reserved']==20
    second=create_plan(analysis,tmp_path/'b',cache=tmp_path/'c',ledger=tmp_path/'l',request=request,style_brief='Smooth natural transitions')
    assert second['cache_key']!=first['cache_key'] and len(calls)==3
