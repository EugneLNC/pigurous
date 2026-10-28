"""Compile existing learned controls into a constrained mathematical action library."""
import copy
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from .audio import sha256
from .score import SceneScore

PROFILE = "mathematical-v2"
RATE = 60
LAYERS = (
    "Harmonic motif",
    "Echo filament",
    "Orbit guide",
    "Accent halo",
    "Phyllotaxis",
)


def inertial_track(target, times, speed, acceleration):
    result = np.empty_like(target, dtype=float)
    result[0] = target[0]
    velocity = 0.0
    for i in range(1, len(target)):
        dt = times[i] - times[i - 1]
        force = np.clip(
            16 * (target[i] - result[i - 1]) - 8 * velocity, -acceleration, acceleration
        )
        velocity = np.clip(velocity + force * dt, -speed, speed)
        result[i] = result[i - 1] + velocity * dt
    return result


def angular_diagnostics(poses, times):
    velocity = np.abs(np.diff(poses[:, 2]) / np.diff(times)) * 180 / np.pi
    return dict(
        rotation_speed_deg_sec_quantiles=np.quantile(
            velocity, [0.5, 0.9, 0.99, 1]
        ).tolist()
    )


def compile_math(p, motion=None):
    source_times = np.asarray(p["times"], float)
    source_controls = np.asarray(p["controls"], float)
    duration = float(p["duration"])
    if len(source_times) < 2 or source_controls.shape != (len(source_times), 9):
        raise ValueError("Expected a complete nine-field presentation track")
    if (
        not np.isfinite(source_times).all()
        or not np.isfinite(source_controls).all()
        or np.any(np.diff(source_times) <= 0)
    ):
        raise ValueError("Non-finite or non-monotonic presentation")
    if abs(source_times[0]) > 1e-6 or abs(source_times[-1] - duration) > 1e-5:
        raise ValueError("Presentation must cover the complete song")
    hits = np.asarray(p.get("hits", []), float)
    if hits.size and (
        hits.ndim != 2 or hits.shape[1] != 2 or not np.isfinite(hits).all()
    ):
        raise ValueError("Invalid predicted accent events")
    times = np.r_[np.arange(0, duration, 1 / RATE), duration]
    controls = np.stack(
        [np.interp(times, source_times, source_controls[:, j]) for j in range(9)], 1
    )
    for j in (0, 1, 2, 3, 6, 7, 8):
        seconds = 0.18 if j == 0 else 0.35 if j < 4 else 0.8
        controls[:, j] = gaussian_filter1d(
            controls[:, j], seconds * RATE, mode="nearest"
        )
        controls[:, j] = np.clip(controls[:, j], 0, 1)
    controls[:, 4] = 0  # Exact predicted hits are sampled causally by the renderer.
    if np.any(np.diff(controls[:, 5]) < -1e-5):
        raise ValueError("Action phase must not run backwards")
    poses = np.asarray(
        p.get("poses", np.tile([0, 0, 0, 1], (len(source_times), 1))), float
    )
    if poses.shape != (len(source_times), 4) or not np.isfinite(poses).all():
        raise ValueError("Invalid source pose track")
    confidence = np.ones(len(source_times))
    if motion is not None:
        motion = np.asarray(motion, float)
        if motion.shape == (len(source_times) - 1, 12):
            confidence = np.linalg.norm(motion[:, 10:12] * 2 - 1, axis=1)
            confidence = np.r_[confidence, confidence[-1]]
    # Smooth orientation as a direction, avoiding angle-wrap and uncertain atan2 jumps.
    sine = np.interp(times, source_times, np.sin(poses[:, 2]) * confidence)
    cosine = np.interp(times, source_times, np.cos(poses[:, 2]) * confidence)
    sine = gaussian_filter1d(sine, 0.6 * RATE, mode="nearest")
    cosine = gaussian_filter1d(cosine, 0.6 * RATE, mode="nearest")
    angle = np.arctan2(sine, cosine)
    for i in range(1, len(angle)):
        if np.hypot(sine[i], cosine[i]) < 0.08:
            angle[i] = angle[i - 1]
    angle = np.unwrap(angle)
    targets = [
        np.clip(np.interp(times, source_times, poses[:, j]), -0.1, 0.1) for j in (0, 1)
    ]
    targets += [angle, np.clip(np.interp(times, source_times, poses[:, 3]), 0.84, 1.02)]
    limits = [(0.12, 0.30), (0.12, 0.30), (0.24, 0.48), (0.08, 0.22)]
    projected = np.stack(
        [
            inertial_track(target, times, *limit)
            for target, limit in zip(targets, limits)
        ],
        1,
    )
    result = copy.deepcopy(p)
    result.update(
        profile=PROFILE,
        action_library="mathematical-actions-v1",
        sample_rate=RATE,
        times=times.tolist(),
        controls=controls.round(7).tolist(),
        poses=projected.round(7).tolist(),
        mathematical_actions=dict(
            slots=["harmonic_rose", "lissajous_3_2", "hypotrochoid_5_2"],
            selection="existing learned effect channels; no newly trained shape selector",
            motion_limits=dict(
                rotation_speed_rad_sec=0.24,
                rotation_acceleration_rad_sec2=0.48,
                translation_speed=0.12,
                translation_acceleration=0.30,
            ),
            raw=angular_diagnostics(poses, source_times),
            projected=angular_diagnostics(projected, times),
            smoothing="continuous controls and pose only; predicted accent timestamps unchanged",
        ),
    )
    return result


def mathematical_score(score, output):
    if Path(score).resolve() == Path(output).resolve():
        raise ValueError("Preserve the original: use a separate mathematical output")
    result = SceneScore.load(score)
    if result.metadata.get("render_profile") != "performance-v1":
        raise ValueError(
            "Expected a performance-v1 score with existing predicted controls"
        )
    p = result.metadata["presentation"]
    raw_motion = result.metadata.get("action_track", {}).get("motion")
    result.metadata["presentation"] = compile_math(p, raw_motion)
    result.metadata.update(
        render_profile=PROFILE,
        title="Music Choreographer / Mathematical actions",
        source_score=str(Path(score).resolve()),
        source_score_sha256=sha256(score),
        execution_method="Existing predicted controls executed by a mathematical action library with bounded motion; weights unchanged",
    )
    for entity in result.entities:
        if 0 <= entity.id < len(LAYERS):
            entity.source["presentation_layer"] = LAYERS[entity.id]
    result.save(output)
    return dict(
        output=str(output),
        profile=PROFILE,
        diagnostics=result.metadata["presentation"]["mathematical_actions"],
        predicted_accent_events=len(p["hits"]),
        origin=result.metadata.get("origin"),
    )
