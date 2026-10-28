import hashlib,json
import pytest
from choreographer.reviews import import_reviews

def test_review_binds_exact_plan_interval_and_displayed_candidates(tmp_path):
    d=tmp_path/'plans'/'chart';d.mkdir(parents=True)
    (d/'plan-0.json').write_text('{}')
    digest=hashlib.sha256((d/'plan-0.json').read_bytes()).hexdigest()
    item=dict(chart_sha256='chart',start_seconds=10,end_seconds=30,candidates=[dict(candidate=0,plan_sha256=digest)])
    (tmp_path/'manifest').write_text(json.dumps(dict(items=[item])))
    vote=dict(chart_sha256='chart',start_seconds=10,end_seconds=30,preferred_candidate=0,source='explicit_user_review')
    (tmp_path/'input').write_text(json.dumps([vote]))
    result=import_reviews(tmp_path/'input',tmp_path/'manifest',tmp_path/'plans',tmp_path/'output')
    assert result[0]['reviewed_candidates']==[0] and result[0]['candidate_plan_hashes']['0']==digest
    (d/'plan-0.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='changed'):import_reviews(tmp_path/'input',tmp_path/'manifest',tmp_path/'plans',tmp_path/'output')

def test_rejected_old_pair_does_not_reject_new_candidate(tmp_path):
    from choreographer.choreography_data import aesthetic_labels
    record=dict(chart_sha256='chart');path=tmp_path/'plan';path.write_text('{}')
    judgments=[dict(chart_sha256='chart',preferred_candidate=None,reviewed_candidates=[0,1],start_seconds=100,end_seconds=120),
               dict(chart_sha256='chart',preferred_candidate=2,reviewed_candidates=[2,3],start_seconds=100,end_seconds=120)]
    assert aesthetic_labels(record,0,path,judgments)['rejected_intervals']
    selected=aesthetic_labels(record,2,path,judgments)
    assert selected['reviewed_intervals'] and not selected['rejected_intervals']
    assert aesthetic_labels(record,4,path,judgments)['aesthetic_source']=='llm_unreviewed'
