import numpy as np
import torch
from choreographer.student_model import ChoreographyStudent,MAX_ACTIONS,student_loss
from choreographer.choreography import PLAN_SIZE,PARAM_SIZE

def batch():
    return dict(audio=torch.randn(2,12,897),times=torch.arange(12)[None].repeat(2,1).float()/16,
        whole=torch.randn(2,3,768),whole_mask=torch.zeros(2,3,dtype=torch.bool),whole_plan=torch.rand(2,3,PLAN_SIZE),
        indices=torch.zeros(2,12,dtype=torch.long),mask=torch.ones(2,12,dtype=torch.bool),motion=torch.rand(2,12,12),accent=torch.zeros(2,12),
        count=torch.tensor([3,2]),material=torch.zeros(2,MAX_ACTIONS,dtype=torch.long),verb=torch.zeros(2,MAX_ACTIONS,dtype=torch.long),
        object=torch.zeros(2,MAX_ACTIONS,dtype=torch.long),parameters=torch.rand(2,MAX_ACTIONS,PARAM_SIZE+2),parameter_mask=torch.ones(2,MAX_ACTIONS,PARAM_SIZE+2),
        event_mask=torch.arange(MAX_ACTIONS)[None]<torch.tensor([3,2])[:,None],synthetic_weight=torch.tensor([1.,.25]))

def test_student_learns_action_heads_and_plan_changes_output():
    torch.set_num_threads(2);torch.manual_seed(42)
    model=ChoreographyStudent(np.random.default_rng(42).random((64,12)).astype(np.float32))
    b=batch();out=model.forward_actions(b,True);loss=student_loss(out,b);loss.backward()
    assert torch.isfinite(loss)
    for head in [model.material_head,model.verb_head,model.parameter_head,model.structure]:assert head.weight.grad.abs().sum()>0
    model.eval()
    with torch.inference_mode():
        first=model.forward_actions(b,True);b['whole_plan']=torch.zeros_like(b['whole_plan']);second=model.forward_actions(b,True)
        assert (first['parameters']-second['parameters']).abs().mean()>1e-5

def test_missing_aesthetic_labels_do_not_train_action_semantics():
    torch.set_num_threads(2);model=ChoreographyStudent(np.zeros((64,12),np.float32));b=batch();b['synthetic_weight'][:]=0
    loss=student_loss(model.forward_actions(b,False),b);loss.backward()
    assert model.verb_head.weight.grad is None or model.verb_head.weight.grad.abs().sum()==0

def test_windows_scalar_count_is_promoted_to_int64():
    from choreographer.student_training import collate
    b=batch();row={k:v[0].numpy() for k,v in b.items() if k!='whole_mask'}
    row['count']=np.int32(3)
    assert collate([row])['count'].dtype==torch.int64

def test_human_preference_is_scoped_to_watched_events_and_plan_bins(tmp_path):
    import json
    from choreographer.student_training import StudentDataset,collate
    times=np.arange(320,dtype=np.float32)/16
    np.savez(tmp_path/'x.npz',audio=np.zeros((320,897),np.float32),motion=np.zeros((320,12),np.float32),
        accent=np.zeros(320,np.float32),times=times,beats=times*2,plan=np.zeros((1,PLAN_SIZE),np.float32),plan_indices=np.zeros(320,np.int64))
    calls=[dict(beat=b,duration=d,material='ribbon',verb='translate',object=0,to=[.5]*12)
           for b,d in [(2,2),(12,2),(18,4),(26,2)]]
    (tmp_path/'p.json').write_text(json.dumps(dict(tempo_map=[dict(beat=0,time=0,bpm=120)],calls=calls)))
    (tmp_path/'manifest.json').write_text(json.dumps([dict(split='train',tensor='x.npz',program='p.json',
        reviewed_intervals=[dict(start_seconds=5,end_seconds=10)],rejected_intervals=[dict(start_seconds=12,end_seconds=14)])]))
    data=StudentDataset(tmp_path/'manifest.json','train');first=data[0];native=data[1];second=data[2]
    np.testing.assert_allclose(first['event_weights'][:2],[.25,1])
    assert first['synthetic_weight']==.25
    np.testing.assert_allclose(first['whole_weights'][2:5],[.25,1,1])
    assert first['whole_weights'][6]==0
    np.testing.assert_allclose(second['event_weights'][:2],[.25,0])
    assert second['synthetic_weight']==0
    assert not native['event_weights'].any() and not native['whole_weights'].any()
    assert collate([first,second])['whole_weights'].shape==(2,10)
    weights=data.preference_sampling_weights(8)
    assert weights[0]==weights[1]==weights[2]==weights[3]==8
    assert weights[4]==weights[5]==1

def test_student_plan_zero_roundoff_and_partial_final_bin():
    from choreographer.student_generation import section_plan_vectors
    raw=np.arange(4*PLAN_SIZE,dtype=np.float32).reshape(4,PLAN_SIZE)
    result=section_plan_vectors(raw,[dict(start_seconds=-1e-16,end_seconds=3),dict(start_seconds=7.99,end_seconds=8)])
    np.testing.assert_allclose(result[0],raw[:2].mean(0))
    np.testing.assert_array_equal(result[1],raw[-1])
    assert np.isfinite(result).all()

def test_invisible_preparation_keeps_the_model_selected_operation(tmp_path,monkeypatch):
    import json
    import choreographer.student_generation as sg
    from choreographer.action_data import MERT_REVISION
    from choreographer.score import TempoMap
    from choreographer.choreography import VERBS,MATERIALS,states
    tempo=TempoMap.from_bpm([(0,120)])
    class Cache:
        metadata=dict(duration=2,audio_sha256='fixture',backend='mert',revision=MERT_REVISION)
        def __init__(self,path):self.tempo=tempo
        def sample(self,times):return np.zeros((len(times),768),np.float32),np.zeros((len(times),129),np.float32)
    class Model:
        def forward_actions(self,batch,teacher):
            count=torch.zeros(1,MAX_ACTIONS+1);count[0,1]=1
            material=torch.zeros(1,MAX_ACTIONS,len(MATERIALS));material[:,:,MATERIALS.index('ribbon')]=1
            verb=torch.zeros(1,MAX_ACTIONS,len(VERBS));verb[:,:,VERBS.index('draw')]=1
            return dict(count=count,material=material,verb=verb,object=torch.zeros(1,MAX_ACTIONS,8),
                        parameters=torch.full((1,MAX_ACTIONS,PARAM_SIZE+2),.5),accent=torch.zeros(1,batch['audio'].shape[1]))
    section=dict(index=0,start_beat=0,end_beat=4,start_seconds=0,end_seconds=2)
    analysis=dict(audio_sha256='fixture',duration=2,sections=[section],tempo_map=tempo.to_list())
    plan=dict(version='choreography-plan-v1',audio_sha256='fixture',sections=[dict(section,motif_id='a',materials=['ribbon'],verbs=['draw'],
        role='hold',layout='center',density=.5,intensity=.5,scale=.5,hue=.5,anchor=[0,0],variation=.5,contrast=.5)])
    monkeypatch.setattr(sg,'Features',Cache);monkeypatch.setattr(sg,'analyze_features',lambda *a:analysis)
    monkeypatch.setattr(sg,'load_student',lambda *a:(Model(),dict(epoch=0)))
    checkpoint=tmp_path/'weights';checkpoint.write_bytes(b'fixture');output=tmp_path/'scene.json'
    sg.generate(checkpoint,'cache',output,'llm',plan)
    program=json.loads(output.read_text(encoding='utf-8'))['metadata']['presentation']
    assert program['calls'][0]['verb']=='draw'
    assert states(program,1.9)[0]['values'][4]==0
