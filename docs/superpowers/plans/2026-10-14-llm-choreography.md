# Learned choreography and LLM planning implementation

Approved scope: restore rich visuals, nine independent geometry materials, twelve composable actions, versioned choreography-v3, timestamped music analysis, DeepSeek whole-song plans, a plan-conditioned student with a learned plan path, retraining, same-checkpoint A/B and six paired training excerpts for human selection.

- [x] Executable action program and Python/TypeScript parity.
- [x] AudioAnalysis, ChoreographyPlan, encrypted local API bridge, persistent 100,000,000-token reservation ledger and cache.
- [x] Compound action student, native geometric/accent auxiliary supervision, separately marked synthetic action supervision, learned/external plan paths and resume.
- [x] Collect up to 80 additional public candidates, freeze 20 new song families before experimentation.
- [x] Generate 6 x 2 twenty-second training candidates, record actual interval preferences, regenerate rejected song 6 once and record 6R1A.
- [x] Single permitted GPU training, export weights and reproducible experiment evidence. Joint: 1,408 updates; preference: 780 updates; preference GPU 5.
- [ ] Full-song A/B for five diagnostic tracks, ablations, nearest-neighbor diagnostics and fresh blinded materials.
- [ ] Rich default preview, synchronized comparison, editable plans/actions, video verification and documentation.

No raw audio, private keys or other project files go to the LLM. API usage counts input plus output once, including reasoning within output. Unknown usage remains reserved. Human-selected labels only cover watched intervals, never entire songs. No inferred or fabricated human votes. API cap is an upper bound, not a consumption target. GPU 7 and its known UUID are always excluded. Existing server work stays untouched; use a new subdirectory of the approved project directory.

Research acceptance remains open: selected-clip imitation improved, but held-out validation did not improve and native motion reconstruction still loses to the constant-mean baseline. Twenty families are excluded from training, but repeated diagnostic reads make the current evaluation exploratory. Blind votes have not been supplied. The code uses a parallel query decoder and continuous parameter regression; complete native entity/parent reconstruction and learned state memory are future work. See `docs/LLM_CHOREOGRAPHY.md` for evidence and deviations.
