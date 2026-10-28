"""Record observed artifacts and explicit acceptance boundaries."""
import hashlib,json
from pathlib import Path
from choreographer.llm_planning import Budget

root=Path('artifacts/student-feedback')
card=dict(name='Music Choreographer v3 research student',format='choreography-student-v1',parameters=11120640,
    checkpoint=str(root/'best.pt'),checkpoint_sha256=hashlib.sha256((root/'best.pt').read_bytes()).hexdigest(),
    architecture=dict(whole_encoder_layers=4,local_encoder_layers=4,compound_query_decoder_layers=4,width=256,heads=8,plan_fields=54,max_queries=64,groups=8,plan_cross_attention=True),
    corpus=dict(records=109,original=56,new_public_candidates_requested=80,new_accepted=53,splits=dict(train=79,val=5,test=25)),
    supervision=dict(native='12-field sampled geometric projection and real non-fake note accents; auxiliary heads',
        new_actions='LLM macro plans plus authored expansion; not direct artist labels',unreviewed_weight=.25,
        human_selected_songs=6,human_selected_seconds=120,complete_selected_events=212,rejected_overlap_events=262,
        watched_interval_weight=1,rejected_interval_weight=0,review_sampling_boost=8,preferences='data/choreography-v3/user-reviews-combined.json'),
    runs=dict(joint=json.loads(Path('artifacts/student-joint/runs/student/summary.json').read_text()),
              preference=json.loads((root/'runs/student/summary.json').read_text())),
    best_preference_epoch=0,server_gpu=dict(physical_index=5,name='RTX 5090',uuid='GPU-e8a4e6e7-8ef5-68b4-4157-f859b237f62a',gpu_7_used=False),
    llm=Budget('artifacts/llm/token-ledger.sqlite').summary(),
    final_outputs=dict(diagnostics=str(root/'predicted-comparison/manifest.json'),fresh=str(root/'predicted-fresh/manifest.json'),blind=str(root/'predicted-blind/participants')),
    validation=dict(aesthetic_acceptance='pending_actual_blind_ratings',native_motion_beats_constant_mean=False,
        preference_stage_improves_held_out_validation=False,improves_selected_training_clip_reconstruction=True),
    limitations=['Parallel action queries, not an autoregressive neural state memory.',
        'Normalized continuous regression; coarse time ticks/residuals are export annotations.',
        'Native aggregate geometry does not reconstruct every entity/parent or label every new primitive.',
        'Authored analytic topology and internal material deformation are execution, not new learned geometry.',
        'Repeated test diagnostics are exploratory, not pristine independent confirmation.',
        'Music/plan ablation sensitivity is not evidence of beneficial musical alignment.',
        'MERT weights carry the upstream non-commercial label; data availability is not an audio redistribution license.'])
(root/'model-card.json').write_text(json.dumps(card,ensure_ascii=False,indent=2),encoding='utf-8')
print('Saved model card, artifact paths and acceptance limits')
