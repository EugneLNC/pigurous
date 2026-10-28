# Mathematical action presentation implementation plan

**Goal:** Execute existing model actions with graceful, closed mathematical curves and bounded continuous movement, retaining the raw score as a comparison.

**Architecture:** A local compiler converts the existing predicted control/pose tracks to a versioned `mathematical-v2` presentation. The three learned effect channels drive a harmonic rose, a 3:2 Lissajous curve and a rational hypotrochoid. Continuous controls receive offline smoothing and acceleration/velocity constraints; accent timestamps are preserved exactly. This changes the authored action library and renderer, not trained weights or aesthetic supervision.

**Tech stack:** NumPy/SciPy, PIL supersampling, TypeScript/Canvas/Three.js, SceneScore.

- [x] Add `math_presentation.py`: safe compilation, finite input validation, circular orientation smoothing, bounded dynamics, unchanged accent timing and source hashes.
- [x] Add `math_geometry.py` and matching TypeScript geometry: closed rational curves, coherent palette, one dominant form, subtle complementary echoes and ordered accent particles. Keep arbitrary-time seeking deterministic.
- [x] Extend Python/TypeScript rendering for the new profile, use supersampled strokes and reduced bloom, retain old profiles.
- [x] Add `choreo math` and preview examples with explicit mathematical-action-library provenance. Maintain playback, seek, editing and export.
- [x] Test curve closure/bounds, movement rate limits, exact accent preservation, cross-language geometry and seek determinism.
- [x] Compile all five server outputs; render the same fixed two-song excerpts and a side-by-side comparison. Create a separately labelled authored action gallery, inspect the actual images and verify encoded audio/video.
- [x] Record raw versus projected motion diagnostics and actual limitations in `docs/MATHEMATICAL_ACTIONS.md`; preserve original weights, scores and reports.

Primary equation references: Wolfram MathWorld Lissajous Curve, Rose Curve and Hypotrochoid. Frequency ratios remain rational and fixed within each primitive so parameter changes do not break closure. No new audio rules, fixed song-to-template assignments or fabricated human shape-selection labels are introduced.
