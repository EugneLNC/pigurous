# Audio-directed performance output Implementation Plan

> Implementation proceeds inline in the existing research workspace under the user's request to improve output. No additional execution approval is required.

**Goal:** Replace illegible raw-line presentation with rhythmically explicit, layered, section-aware output, while retaining the original learned score and identifying procedural contributions.

**Architecture:** A `polish` compiler reads timestamped audio features and the generated score, detects actual transient peaks and phrase energy, and stores a deterministic presentation extension in SceneScore metadata. Five editable layer entities control the composition. Python/Canvas renderers share the same analytic geometry and timestamped control contract; arbitrary seeking never depends on previous frames.

**Tech Stack:** NumPy/SciPy, Pillow/FFmpeg, TypeScript/Canvas2D, existing Three.js preview.

## Tasks

- [x] `src/choreographer/presentation.py`: compile 30 Hz robustly normalized bass/mid/high/energy controls; onsets use spectral flux peaks with >=160 ms separation, causal attack/decay pulses; 16-beat descriptors blend three compositions (orbit, ribbons, tunnel), with gradual phrase transitions. Silence gates effects. Bound geometry inside frame. Store source hash and `origin=audio_directed_presentation`; preserve source score as a separate input. `polish(score, audio, features, output, seed=42)` validates audio hashes and duration.
- [x] `src/choreographer/performance_render.py`: pure `primitives(presentation, time)` emits paths and points; deterministic orbit contours, mirrored ribbons, depth rings, onset bursts and seeded star field. `performance_frame` draws glow plus sharp cores. Route only the explicit presentation profile through it in `render.py`; old SceneScore rendering is preserved.
- [x] `preview/src/performance.ts`: equivalent sampled controls and analytic geometry rendered to CanvasTexture. `main.ts` recognizes the profile, exposes layer transforms, and correctly records the composed canvas. Report procedural origin in the UI.
- [x] `src/choreographer/cli.py`: expose `polish SCORE --audio AUDIO --features CACHE -o OUTPUT --seed 42`. No training/model claim for procedural output.
- [x] `tests/test_presentation.py`: deterministic replay, silent-audio gating, beat onset alignment, wrong-audio rejection, bounded coordinates and meaningful response to audio substitution. Run `python -B -m pytest -q -p no:cacheprovider`.
- [x] Real validation: polish NWAD and a second held-out song; render complete videos and multiple stills across sections, inspect frames, run browser import/seek/play/export with no errors. Run `npm test` and `npm run build` in `preview`. Save exact command/examples and limitations in `docs/PRESENTATION.md`.

## References consulted

- MilkDrop: https://www.geisswerks.com/milkdrop/ — waveform-based scenes and beat-triggered effects.
- Butterchurn: https://github.com/jberg/butterchurn — WebGL MilkDrop implementation and preset blending.
- Synesthesia audio uniforms: https://synesthesia.live/docs/ssf/audio_uniforms.html — separate frequency-band levels and transient controls.

Original implementation; no vendor presets, assets or shader source copied. Audio responsiveness and image improvements must not be reported as a successful retraining result.
