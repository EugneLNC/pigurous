"""Executable predicted actions. The fixed adapter defines geometry, never reads audio."""
import json
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
from .action_data import RATE, FIELDS, MERT_REVISION
from .action_training import load_model, predict
from .audio import Features, sha256
from .score import SceneScore, Entity, Event, TempoMap
from .presentation import LAYERS


def execute_motion(motion, accent, tokens, duration, seed=42, effects=None):
    """A documented geometric vocabulary, weakly supervised by native chart motion.

    Rotation drives orbit visibility, coherent translation drives ribbons, and
    expansion/spread drives depth rings. These fixed actuator semantics do not
    infer timing, sections or style from audio. All inputs are model predictions.
    """
    m = np.clip(np.asarray(motion, float), 0, 1)
    if (
        m.ndim != 2
        or m.shape[1] != len(FIELDS)
        or len(m) < 2
        or not np.isfinite(m).all()
    ):
        raise ValueError("Invalid predicted motion track")
    accent = np.clip(np.asarray(accent, float), 0, 1)
    if accent.shape != (len(m),) or not np.isfinite(accent).all():
        raise ValueError("Invalid accents")
    times = np.arange(len(m)) / RATE
    (
        density,
        x,
        y,
        spread,
        travel,
        spin,
        zoom,
        opacity,
        alignment,
        scale,
        sine,
        cosine,
    ) = m.T
    if effects is None:
        # Compatibility with the earlier geometry-only pilot. New checkpoints
        # predict these actuator weights directly; no adapter chooses them.
        activations = np.stack(
            [
                spin + 0.25 * (1 - alignment),
                travel + 0.25 * alignment,
                zoom + 0.5 * spread,
            ],
            1,
        )
        activations /= np.maximum(activations.sum(1, keepdims=True), 1e-6)
    else:
        activations = np.asarray(effects, float)
        if (
            activations.shape != (len(m), 3)
            or not np.isfinite(activations).all()
            or np.any(activations < 0)
            or np.any(activations > 1)
        ):
            raise ValueError("Invalid predicted effect activations")
    # No energy-dependent template selection, spectral peaks or beat-trigger fallback.
    phase = np.cumsum(0.5 + 3 * travel + 2 * spin) / RATE
    controls = np.c_[
        opacity * (0.35 + 0.65 * density),
        zoom,
        travel,
        spin,
        np.zeros(len(m)),
        phase,
        activations,
    ]
    peaks, _ = find_peaks(accent, height=0.5, distance=2, prominence=0.05)
    hits = [[round(float(times[i]), 5), round(float(accent[i]), 5)] for i in peaks]
    angles = np.unwrap(np.arctan2(sine * 2 - 1, cosine * 2 - 1))
    poses = np.c_[(x - 0.5) * 0.6, (y - 0.5) * 0.6, angles, 0.7 + 0.6 * scale]
    sections = []
    for start in range(0, len(m), RATE * 8):
        end = min(start + RATE * 8, len(m))
        mix = activations[start:end].mean(0)
        sections.append(
            dict(
                start=float(times[start]),
                end=min(duration, end / RATE),
                mode=["orbit", "ribbons", "tunnel"][int(mix.argmax())],
                weights=mix.tolist(),
                words=sorted(set(map(int, tokens[start:end]))),
            )
        )
    # Endpoint duplication ensures interpolation remains defined to exact audio end.
    return dict(
        profile="performance-v1",
        action_version="motion-actions-v1",
        effect_selection="model_head"
        if effects is not None
        else "legacy_geometry_adapter",
        strict_gates=True,
        duration=duration,
        seed=seed,
        hue=0.5 + (seed % 17) * 0.005,
        sample_rate=RATE,
        times=np.r_[times, duration].tolist(),
        controls=np.r_[controls, controls[-1:]].round(6).tolist(),
        poses=np.r_[poses, poses[-1:]].round(6).tolist(),
        hits=hits,
        sections=sections,
        columns=[
            "visibility",
            "expansion",
            "travel",
            "spin",
            "accent",
            "phase",
            "orbit",
            "ribbons",
            "tunnel",
        ],
    )


def score_from_prediction(
    pred, metadata, tempo, checkpoint, seed=42, condition="correct"
):
    duration = float(metadata["duration"])
    p = execute_motion(
        pred["motion"],
        pred["accent"],
        pred["tokens"],
        duration,
        seed,
        pred.get("effects"),
    )
    boundaries = np.r_[
        0, np.flatnonzero(np.diff(pred["tokens"])) + 1, len(pred["tokens"])
    ]
    calls = [
        dict(
            word=int(pred["tokens"][a]),
            start_seconds=float(a / RATE),
            duration_seconds=float(min(duration, b / RATE) - a / RATE),
            parameter_frames=[int(a), int(b)],
            actuators=["orbit", "ribbons", "tunnel", "sparks", "stars"],
        )
        for a, b in zip(boundaries[:-1], boundaries[1:])
    ]
    score = SceneScore(
        duration,
        tempo,
        metadata=dict(
            origin="learned_actions",
            render_profile="performance-v1",
            title="Music Choreographer / Learned actions",
            audio=metadata,
            presentation=p,
            action_schema=dict(
                version="motion-actions-v1",
                fields=FIELDS,
                rate=RATE,
                supervision="human chart geometry projection; hand-designed effect actuator semantics",
            ),
            action_track=dict(
                calls=calls,
                tokens=pred["tokens"].tolist(),
                motion=pred["motion"].round(6).tolist(),
                accent=pred["accent"].round(6).tolist(),
                effects=pred["effects"].round(6).tolist()
                if "effects" in pred
                else None,
            ),
            checkpoint_sha256=sha256(checkpoint),
            condition=condition,
            complete_generation=True,
            model_geometry_used=True,
        ),
    )
    tm = TempoMap(tempo)
    for i, name in enumerate(LAYERS):
        score.entities.append(
            Entity(i, "circle", width=1, height=1, source={"presentation_layer": name})
        )
        score.events.append(Event(tm.time_to_beat(0), "create", i))
    for s in p["sections"]:
        score.segments.append(
            dict(
                start=tm.time_to_beat(s["start"]),
                end=tm.time_to_beat(s["end"]),
                mode=s["mode"],
                words=s["words"],
            )
        )
    return score


def generate(
    checkpoint, features, output, device="cpu", seed=42, condition="correct", audio=None
):
    f = Features(features)
    if f.metadata["backend"] != "mert":
        raise ValueError("Action checkpoint requires MERT")
    if f.metadata.get("revision") != MERT_REVISION:
        raise ValueError(
            "Action checkpoint requires the pinned MERT revision used for training"
        )
    if audio and f.metadata["audio_sha256"] != sha256(audio):
        raise ValueError("Audio/cache mismatch")
    times = np.arange(0, f.metadata["duration"], 1 / RATE)
    a, t = f.sample(times)
    model, _ = load_model(checkpoint, device)
    pred = predict(model, np.c_[a, t], device, condition)
    score = score_from_prediction(
        pred, f.metadata, f.tempo.to_list(), checkpoint, seed, condition
    )
    score.save(output)
    return dict(
        output=output,
        words=len(set(pred["tokens"])),
        accents=len(score.metadata["presentation"]["hits"]),
    )


def evaluate(checkpoint, manifest, output, device="cpu"):
    root = Path(manifest).parent
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    records = json.loads(Path(manifest).read_text(encoding="utf-8"))
    tests = [r for r in records if r["split"] == "test"]
    model, ck = load_model(checkpoint, device)
    train = [
        np.load(root / r["tensor"])["motion"] for r in records if r["split"] == "train"
    ]
    mean = np.concatenate(train).mean(0)
    report = []
    for r in tests:
        z = np.load(root / r["tensor"])
        actual = z["motion"]
        n = len(actual)
        predictions = {
            c: predict(model, z["audio"], device, c)
            for c in ("correct", "zero", "shuffle")
        }
        # Match other-song negatives by average tempo and log-band energy, without labels.
        bpm = lambda v: float(np.median(np.diff(v["beats"])) * RATE * 60)
        energy = lambda v: float(np.mean(v["audio"][:, 768:896]))
        candidates = [
            v
            for v in records
            if v["audio_hash"] != r["audio_hash"] and v["song_id"] != r["song_id"]
        ]
        candidates = [(v, dict(np.load(root / v["tensor"]))) for v in candidates]
        other, oz = min(
            candidates,
            key=lambda v: abs(np.log(max(1, bpm(v[1])) / max(1, bpm(z))))
            + abs(energy(v[1]) - energy(z)),
        )
        indices = np.arange(n) % len(oz["audio"])
        predictions["matched_swap"] = predict(model, oz["audio"][indices], device)
        correct = predictions["correct"]
        metrics = {
            c: dict(
                motion_mse=float(((p["motion"] - actual) ** 2).mean()),
                accent_brier=float(((p["accent"] - z["accent"]) ** 2).mean()),
                output_change=float(np.abs(p["motion"] - correct["motion"]).mean()),
                token_change=float(np.mean(p["tokens"] != correct["tokens"])),
            )
            for c, p in predictions.items()
        }
        if "effects" in correct:
            for c, prediction in predictions.items():
                metrics[c]["effect_weight_change"] = float(
                    np.abs(prediction["effects"] - correct["effects"]).mean()
                )
        counts = np.bincount(correct["tokens"], minlength=len(model.vocabulary))
        prob = counts[counts > 0] / n
        windows = [
            correct["motion"][i : i + RATE * 8].mean(0) for i in range(0, n, RATE * 8)
        ]
        row = dict(
            song_id=r["song_id"],
            audio=r["audio"],
            tensor=r["tensor"],
            metrics=metrics,
            constant_train_mean_mse=float(((actual - mean) ** 2).mean()),
            distinct_words=int((counts > 0).sum()),
            token_entropy=float(-(prob * np.log2(prob)).sum()),
            section_std=float(np.std(windows, axis=0).mean()),
            matched_song=other["song_id"],
            bpm=bpm(z),
            matched_bpm=bpm(oz),
            falsely_hidden_fraction=float(
                np.mean((correct["motion"][:, 7] < 0.05) & (actual[:, 7] > 0.5))
            ),
            predicted_effect_means=correct["effects"].mean(0).tolist()
            if "effects" in correct
            else None,
            predicted_effect_std=correct["effects"].std(0).tolist()
            if "effects" in correct
            else None,
        )
        score = score_from_prediction(
            correct,
            json.loads(str(z["metadata"])),
            json.loads(str(z["tempo"])),
            checkpoint,
        )
        score.save(out / (Path(r["tensor"]).stem + ".json"))
        report.append(row)
        print(json.dumps(row, ensure_ascii=True), flush=True)
    summary = dict(
        test_songs=len(report),
        checkpoint_epoch=ck["epoch"],
        results=report,
        limitation="Proxy motion reconstruction and conditioning diagnostics; not an aesthetic preference study",
    )
    (out / "evaluation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
