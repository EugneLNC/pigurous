# Learned visual actions implementation plan

**Goal:** Train an audio-conditioned action director on additional human performance charts and render its predictions without the procedural audio director.

**Architecture:** Sample supported native line geometry and note timings into bounded motion descriptors. Fit a train-only 64-word motion codebook; train a whole-song planner and local temporal network to predict words, descriptor residuals and chart accents from frozen MERT and transient features. An explicit, documented geometry-to-effect adapter executes the predicted descriptors with orbit, ribbon, tunnel and spark primitives. This adapter is a hand-designed visual vocabulary, not learned aesthetic supervision.

**Tech stack:** Python/PyTorch, frozen MERT, NumPy/SciPy, SceneScore, TypeScript/Three.js.

- [x] Expand `scripts/fetch_phira.py` with pagination; collect pages 2–3 into a new directory. Keep package hashes, quarantine unsupported charts and never alter the original corpus.
- [x] Add `action_data.py`: sample native geometry at 16 Hz, retain unsupported-feature exclusions, group by song identity and audio hash, preserve previous held-out splits, fit normalization/codebook only on training songs. Cache MERT at timestamped sample positions.
- [x] Add `action_model.py` and `action_training.py`: whole-song attention, local sequence conditioning, discrete motion words plus residuals, sparse accent loss, checkpoint/resume, validation early stopping and single-GPU memory cap. New output heads require a new action branch; retain original event checkpoints.
- [x] Add `actions.py`: versioned executable action tracks, deterministic model inference and score export. Render only predicted controls; no calls to `compile_presentation`, energy quantiles or audio peak detection during action inference/rendering.
- [x] Extend Python and browser renderers with explicit layer gates and model origins. Verify learned score preview, seeking and output export.
- [x] Add tests for train-only codebook fitting/group isolation, empty action gates, reproducibility, checkpoint restore and swapped/zero conditioning.
- [x] Upload only this project's new code, labels and cached features. Train with `scripts/shared_gpu.py`, excluding GPU 7 and reserving 2 GiB from current free memory.
- [x] Generate all held-out songs, record correct/zero/shuffled/matched-audio controls, token diversity and reconstruction errors. Render two actual held-out examples and state limitations without claiming aesthetic success from training loss.

Validation commands: `.venv/Scripts/python.exe -B -m pytest -q`; `npm.cmd run build` and `npm.cmd test` inside `preview`; use `choreo actions prepare/train/generate/evaluate` for the new branch. Real training results and supported corpus counts belong in `docs/LEARNED_ACTIONS.md` after measurement.

The user explicitly approved the expanded payload after the initial automatic review rejection. Upload hash verification, an isolated server checkout, GPU 0 training for 15 epochs / 900 updates, real checkpoint restoration, all-five-song conditioning diagnostics and result download are complete. Local and server experiments remain separately recorded. Final architecture adds independently learned visibility and explicit effect activation heads; see `docs/LEARNED_ACTIONS.md`.
