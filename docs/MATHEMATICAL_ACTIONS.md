# Mathematical action execution (2026-10-14)

The `mathematical-v2` profile gives the existing learned controls a closed, coherent geometric vocabulary and bounded movement. It is an action-library/rendering upgrade. No weights were retrained, no new mathematical shape labels were fabricated, and the original model scores remain intact.

## Diagnosis and change

The server NWAD output had pose rotation speeds of 47.15 / 279.05 / 1,062.12 / 2,494.46 degrees per second at the 50th / 90th / 99th / maximum percentiles. Discrete word changes and uncertain orientation estimates caused abrupt whole-scene movement. Multiple unrelated waves, contours and spark rays compounded the visual clutter.

`math_presentation.py` smooths continuous controls, estimates orientation through confidence-weighted sine/cosine vectors and executes a damped movement response at 60 Hz. Pose angular speed is bounded at 0.24 rad/s (13.75 degrees/s), angular acceleration at 0.48 rad/s². Translation and scale have separate speed/acceleration bounds. These limits describe the scene pose, not every point's internal parametric velocity. The compiler operates offline and can anticipate changes through centred smoothing; it is not a causal real-time filter.

Predicted hit timestamps and strengths are copied exactly, and the hit sampler remains causal. This preserves model timing, not proof that all predicted hits match the music correctly. Continuous motion smoothing and all display constraints are authored actuator behaviour.

## Mathematical vocabulary

The three existing effect channels become basis weights for:

- **Harmonic rose:** a fivefold rose/rosette family based on `r = a cos(5θ)`, with bounded radial deformation.
- **Lissajous:** `x = A sin(3θ + δ)`, `y = B sin(2θ)`. The fixed 3:2 ratio closes after `2π`. The phase stays away from the approximately `π/4` retracing degeneracy, preserving an open-looking weave rather than a nearly retraced arc.
- **Hypotrochoid:** `x = (R-r) cos(t) + d cos((R-r)t/r)`, `y = (R-r) sin(t) - d sin((R-r)t/r)`. Here `R/r=5/2`, and the path closes over `4π`.

The renderer linearly mixes the closed basis paths into **one main contour**, normalizes it to a legible viewport scale, and adds a faint echo. This avoids drawing three competing motifs across the same screen. Palette changes run smoothly along the contour. Background points use a golden-angle arrangement; accent particles appear in a short symmetrical halo. Twofold supersampling reduces staircase edges; restrained bloom keeps fine lines readable.

The five editable layers represent output roles: main motif, echo, orbit guide, accent halo and phyllotaxis. They are not new independently trained action heads. Existing predicted channels, motion and accent timing drive this execution profile. Mathematical topology, petal counts, frequency ratios, palette and layout constraints remain authored. The model has **not** learned independent selection of those new mathematical properties.

Primary equation references: [Wolfram MathWorld: Lissajous Curve](https://mathworld.wolfram.com/LissajousCurve.html), [Rose Curve](https://mathworld.wolfram.com/RoseCurve.html), [Hypotrochoid](https://mathworld.wolfram.com/Hypotrochoid.html). Geometry is implemented independently; no reference images or third-party code were copied.

## Use and outputs

```bash
choreo math artifacts/actions-server/dcef4cf813abf326.json -o mathematical.json
choreo render mathematical.json --audio song.ogg -o mathematical.mp4 --width 1280 --height 720 --fps 30
python -B scripts/math_demo.py --render
```

`choreo math` requires a complete existing `performance-v1` control track and refuses an output path identical to its input. It stores the original source hash, preserves origin/checkpoint metadata and leaves raw `action_track` data available for inspection. No audio or feature extraction is performed by this compiler.

All five server scores have mathematical counterparts in `artifacts/mathematical-v2`; `manifest.json` records source paths and raw/projected movement diagnostics. All five maximum pose rotation speeds are bounded at approximately 13.75 degrees/s, compared with original maxima between 2,403 and 2,774 degrees/s. These movement metrics do not measure aesthetic quality or music correspondence.

The two 24-second musical examples retain the previous fixed protocol: first official and first Phira development test songs, excerpts starting at 10 seconds, 45% and 80% of each track. `before-after.mp4` shows the same NWAD model prediction through the previous and mathematical executors. The gallery `action-gallery.png` separately displays **authored isolated primitives**, explicitly labelled as such, rather than model choreography.

The preview defaults to the mathematical server example. “原始模型” opens the exact underlying server score and “规则对照” opens the earlier procedural comparator. Source labels, deterministic seeking, layer edits and exports are preserved.

## Validation and limitation

Validation covers bounded continuous rotation, circular angle handling, unchanged accent events, silence, curve closure without an endpoint seam, finite viewport geometry and stateless seeking. Python/TypeScript fixtures check controls and primitive geometry at arbitrary seek times. Rendering uses independent rasterizers; the fixture is not a claim of pixel-identical browser/offline output.

Browser checks cover the mathematical default, comparison buttons, repeated seeking, layer editing/saving, associated audio playback, mobile layout and WebM export. Original rendering profiles remain supported.

Executed checks: **25 Python tests, 4 TypeScript fixture tests**, production preview build and the mathematical browser test passed. MP4s were verified as H264/AAC, 30 fps, approximately 24.02 seconds (960×540 examples; 1280×360 comparison). Browser export is checked with ffprobe for a real decodable video stream, audio stream and at least 1.5 seconds of duration. A stale time-slider value during recording startup could previously let a test stop before capturing frames; recording now redraws time zero before capturing the stream, and the browser test verifies decoded media instead of download alone.

This adds an attractive, constrained action space for the existing model to execute. It does not establish learned aesthetic choreography: the model still often favours the former ribbon channel, now interpreted as the Lissajous basis. No new blind rating, genre generalization or mathematical-action training success is claimed. Independent shape selection and long-form mathematical motifs need suitable authored animation demonstrations or direct annotations, followed by a new locked evaluation set.
