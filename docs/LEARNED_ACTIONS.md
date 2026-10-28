# Learned action director

The `actions` branch learns time-varying motion from human charts. It does not use `polish` output as training labels. `polish` remains the explicit procedural comparator.

## What is learned and what is authored

The frozen MERT representation and 10 ms transient features condition a four-layer whole-song planner and four-layer local Transformer. The model predicts a 64-word motion vocabulary, 12 continuous residual parameters, an accent probability track, continuous density/visibility and explicit orbit/ribbon/tunnel activation weights. The vocabulary is fitted on training charts only. Original model `global_in` and compatible `planner` tensors can initialize this branch; the original event decoder is retained separately.

Human supervision consists of native chart line density, weighted center, spread, translation speed, unwrapped rotation speed, scale change, opacity, angular alignment, scale and mean orientation. Non-fake note timestamps supervise accents. Sampling is 16 Hz, giving 62.5 ms spacing, not the original event decoder's 1/192-beat precision. Fast changes below this sampling rate can be lost. Notes teach hit timing; line geometry teaches movement. This is a geometric projection of chart authorship, not complete reconstruction of a chart's visual design.

The explicit effect head learns weak labels derived from human geometry: rotation supports orbit contours, coherent translation supports ribbons, expansion and spread support depth rings. The final checkpoint predicts the three activations directly. `actions.execute_motion` validates and executes those weights; it does not classify the predicted geometry into effects. Legacy checkpoints retain a clearly marked `legacy_geometry_adapter` mode, while the final output carries `effect_selection=model_head`.

Predicted accent peaks trigger sparks; predicted orientation, center and size transform the scene. Palette and primitive shapes remain authored. This correspondence is weak supervision: charts do not contain hand-labelled orbit/ribbon/tunnel choices. The renderer interpolates and executes the resulting controls without inspecting audio, selecting modes by loudness, finding spectral onsets, or falling back to the old procedural director. The discrete words and residuals determine motion; separate heads predict persistent visibility and effect choice. The accent peak decoder uses a fixed 0.5 probability threshold and 125 ms minimum separation on model predictions, not audio flux.

Inference uses 16-second, 50% overlapping local windows with whole-song context pooled every 2 seconds and weighted overlap/add. It is an offline temporal predictor, not the original autoregressive event decoder. Windows do not recreate entity IDs. Exported 8-second section labels summarize predictions for inspection; they are not inferred musical phrase boundaries. Decoding is greedy and deterministic; seed changes the procedural particle layout and palette, not the action token sequence. No claim is made that it has already learned good long-range narrative, 3D composition or professional aesthetic quality.

## Corpus

`data/phira-expanded` contains the newly collected public listing pages 2–3 (40 packages requested, 32 unambiguous chart/audio pairs, 8 quarantined). It complements the previous 16 Phira pairs and 20 official pairs. Package and chart hashes and public URLs are retained. Public availability does not grant redistribution permission; raw charts/music are excluded from the new server bundle and code release.

`data/actions-v1/manifest.json` contains 56 accepted chart records: 18 official plus 38 Phira. Twelve of the 68 candidate pairs are quarantined for unsupported channels, special textures/UI or invalid timing. Sampling native supported trajectories bypasses curve compression; it does not relax the separate strict SceneScore reconstruction audit. No unsupported chart is silently presented as full supervision.

Grouped split: 47 training chart records, 4 validation, 5 test. Song title aliases and identical audio hashes are grouped together; previous held-out official songs keep their split. Different encodings or completely different title aliases can still evade this conservative grouping, so this is not an acoustic fingerprint deduplication guarantee. The same song can have multiple accepted charts; record count is not unique-song count. Both normalization units and the geometric adapter are fixed before evaluation; vocabulary fitting sees training records only.

## Commands

```powershell
$env:CHOREO_MERT_PATH=(Resolve-Path data/models/MERT-v1-95M).Path
choreo actions prepare data/prepared-mert/manifest.json data/phira/sources.json data/phira-expanded/sources.json -o data/actions-v1 --device cuda
choreo actions train data/actions-v1/manifest.json -o runs/actions-v1 --initialize artifacts/server-5090/inference.pt --epochs 40 --device cuda
choreo actions train data/actions-v1/manifest.json -o runs/actions-v1 --resume runs/actions-v1/last.pt --epochs 60 --device cuda
choreo actions generate --checkpoint runs/actions-v1/best.pt --features path/to/mert-cache.npz --audio song.ogg -o learned.json
choreo render learned.json --audio song.ogg -o learned.mp4 --width 1280 --height 720 --fps 30
choreo actions evaluate --checkpoint runs/actions-v1/best.pt --manifest data/actions-v1/manifest.json -o artifacts/learned-actions
```

On the shared server every GPU command must run under `scripts/shared_gpu.py`. It selects one currently available GPU by UUID, excludes physical GPU 7 and its known UUID, leaves at least 2 GiB free, and caps allocation at the lesser of available memory minus headroom and 30 GiB. No other users' jobs are stopped.

## Evaluation boundary

Tests check train-only vocabulary, song/audio group isolation, strict layer disabling, deterministic seeking, conditioning sensitivity and model checkpoint restoration. Python/TypeScript parity includes both procedural controls and learned poses. These are correctness checks, not listening tests.

Held-out evaluation generates every test song and compares correct, zero, shuffled and tempo/energy-matched other-song audio. It records motion MSE, accent Brier score, word diversity, section variation and output change. The train-mean motion baseline helps detect collapse. A change after swapping audio alone does not prove that the correct result is musically better. Five test records are insufficient for the planned 20-song aesthetic acceptance study; no preference-rate or generalization claim should be inferred.

## Executed experiments (2026-10-13)

Automatic approval initially rejected the expanded bundle because earlier permission covered only 20 songs. The user subsequently approved this exact new 453.5 MB code/feature/label bundle. It has now been uploaded and all 84 payload files verified. The three experiments below predate that upload and ran locally on one RTX 3070 Ti Laptop GPU with a 3400 MiB allocator cap. The later server run is recorded separately below.

| Run | Actual epochs / updates | Selected checkpoint | Notes |
|---|---:|---|---|
| `runs/actions-local` | 16 / 3,792 | epoch 5 (zero based) | 50 planner tensors transferred from the existing 5090 event model; first 2 epochs followed by a real optimizer/RNG resume |
| `runs/actions-visibility` | 11 / 2,607 | epoch 0 | Initialized from previous best; learned density/visibility head fixes discrete-word-induced blackouts |
| `runs/actions-direct` | 11 / 2,607 | epoch 0 | Initialized from visibility best; adds explicit learned effect activation head |

Total work: **9,006 updates across separate experiments**, not one 38-epoch checkpoint. The final checkpoint's lineage uses the selected best at each stage (6 + 1 + 1 epochs), rather than all terminal checkpoints. Every run stopped after 10 epochs without validation improvement. The final model has **6,768,990 trainable parameters**; frozen MERT is cached separately. Measured PyTorch peak allocated memory was approximately 0.145 GiB for these batch-2, 256-frame windows; this excludes cached CPU tensors, the frozen encoder and CUDA/driver overhead and is not a full VRAM usage estimate.

Final weights: `runs/actions-direct/best.pt`. Final five complete SceneScores, the evaluation JSON and example videos: `artifacts/learned-actions-direct/`. The old experiments remain intact. `preview/public/learned.json` is the actual NWAD output from this checkpoint; the preview offers both model and procedural examples.

Mean over the five songs (lower error is better):

| Audio condition | Motion MSE | Accent Brier |
|---|---:|---:|
| Correct | 0.06394 | 0.16983 |
| Zero | 0.06424 | 0.19038 |
| Shuffled 2-second blocks | 0.07024 | 0.26047 |
| Tempo/energy-matched other song | 0.05590 | 0.25873 |

The train-mean motion baseline is **0.04271**, better than the learned motion result. Correct audio improves accent prediction versus shuffled/swapped audio, but it has no consistent motion reconstruction advantage: swapped music is better by that metric. The model is not accepted as successful spatial choreography. The five songs use respectively 29, 16, 15, 20 and 5 motion words. Swapping audio changes 67.47% of words on average and changes effect weights by mean absolute 0.1159. Such sensitivity is not proof of aesthetic correspondence. All five have zero measured false-hidden frames under the diagnostic `(predicted opacity < .05) and (chart opacity > .5)`; this is not a guarantee against all dark frames.

The first held-out outputs revealed the visibility issue and informed subsequent implementation. Therefore the final five-song results are **exploratory diagnostics, not a pristine final test or an independent acceptance study**. No test labels entered optimization or vocabulary fitting, but a new locked set is needed for final claims. Next evidence needed is direct human visual-effect annotation or suitable animation demonstrations, stronger song-level validation and at least 20 new blind-evaluation songs. More GPU capacity alone does not resolve the observed generalization gap.

Validation: 20 Python tests, 3 TypeScript parity tests and browser checks for model provenance, seek reproducibility, changing frames, layer editing/export, audio playback and mobile layout. The two demonstration videos use the first official test song and first Phira test song in manifest order, with fixed 8-second excerpts starting at 10 seconds, 45% and 80% of each track. They are examples, not selected winners or complete-song quality acceptance.

## Server action run (2026-10-13)

The approved archive is 453,512,082 bytes with SHA256 `358942d0072bfb7c9f4d7ca2d5d72080c0d9a6decbedaee604c564c28043188a`. It was unpacked into a new `/path/to/music-choreographer/research-actions-v1` directory. Existing source and experiment folders were preserved. Both upload and download receipts are in `artifacts/actions-server`.

The run used **physical GPU 0, RTX 5090**, UUID `GPU-931eae62-3c33-2587-1e87-b70886c58ff4`. Physical GPU 7 and its UUID were excluded by `shared_gpu.py`. At launch, allocation was capped at 10,462 MiB based on existing shared usage, leaving 2,048 MiB headroom. Training and generation ran serially. No other users' jobs were stopped.

This is a separate action training run initialized from the original server event model's 50 compatible planner tensors, not an upload/resume of the local action weights. Batch size was 8. Two initial epochs were followed by actual optimizer/RNG checkpoint restoration and continuation to a 40-epoch maximum. Validation early stopping ended at **15 epochs / 900 optimizer updates**, selecting **epoch 4 (zero based)**. The final model has 6,768,990 trainable parameters. Peak PyTorch allocated memory was 0.2551 GiB; as with the local runs, this excludes driver/context overhead, other users' allocations and CPU feature caches.

Server environment: Python 3.12.3, PyTorch 2.7.0+cu128, CUDA build 12.8, NumPy 2.5.4. Six uploaded action tests passed on CPU. The nested checkout requires `PYTHONPATH=src`; pytest must use an independent configuration (`-c /dev/null -p no:cacheprovider`) to avoid inheriting the parent's source path.

| Audio condition | Mean motion MSE | Mean accent Brier |
|---|---:|---:|
| Correct | 0.05678 | 0.14843 |
| Zero | 0.08445 | 0.18135 |
| Shuffled | 0.06209 | 0.20930 |
| Matched other song | 0.06506 | 0.23089 |

The train-mean motion baseline remains **0.04271**, so spatial reconstruction has not passed that comparator. Correct audio improves the aggregate diagnostics versus audio ablations, while individual songs can still favour swapped audio. Word counts are 14, 5, 4, 13 and 3. A matched audio swap changes an average 44.41% of words. These are exploratory results from the same five development test songs, not an independent generalization or aesthetic test. The server output is retained separately rather than silently replacing the local preview based on test results.

Local server weights: `artifacts/actions-server/best.pt`; SHA256 `d30533270600d346a0e10ccae41a550c317cc69f5e1f1b8c959df3ac6b507b35`. Execution evidence, model card, metrics and all five complete SceneScores are beside the weights. Two fixed-protocol 24-second video montages are `example-1.mp4` (NWAD) and `example-2.mp4` (太阳系 Disco).

To continue on the server, keep using the isolated directory and guard:

```bash
cd /path/to/music-choreographer/research-actions-v1
PYTHONPATH=src ../.venv/bin/python -B scripts/shared_gpu.py --required-mb 8192 -- ../.venv/bin/python -B -m choreographer.cli actions train data/actions-v1/manifest.json -o runs/actions-5090 --resume runs/actions-5090/last.pt --epochs 60 --batch-size 8 --device cuda
```

The completed run stopped for validation stagnation; further epochs alone are not assumed to improve its spatial choreography.
