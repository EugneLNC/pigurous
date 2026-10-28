"""Audio-directed presentation compiler; procedural output, not learned supervision."""
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from .audio import Features, sha256
from .score import SceneScore, Entity, Event

PROFILE = "performance-v1"
LAYERS = (
    "Orbit contours",
    "Melodic ribbons",
    "Depth tunnel",
    "Onset sparks",
    "Star field",
)


def normalized(x):
    x = np.asarray(x, dtype=float)
    hi = float(np.quantile(x, 0.95))
    return np.clip(x / max(hi, 1e-5), 0, 1)


def compile_presentation(features, seed=42):
    f = Features(features)
    duration = float(f.metadata["duration"])
    t = np.arange(0, duration, 1 / 30)
    t = np.r_[t, duration]
    raw = np.maximum(0, f.transient[:, :128])
    energy = raw.mean(1)
    # Absolute silence stays silent even when the file contains little variation.
    gate = np.clip(energy / 0.012, 0, 1)
    channels = [normalized(energy)]
    channels += [
        normalized(raw[:, a:b].mean(1)) for a, b in [(0, 35), (35, 85), (85, 128)]
    ]
    tracks = np.stack(
        [
            np.interp(t, f.transient_times, gaussian_filter1d(x * gate, 1.0))
            for x in channels
        ],
        1,
    )
    flux = np.maximum(f.transient[:, 128], 0)
    step = max(1e-4, float(np.median(np.diff(f.transient_times))))
    peaks, _ = find_peaks(
        flux,
        distance=max(1, int(0.16 / step)),
        height=max(1e-6, float(np.quantile(flux, 0.65))),
        prominence=max(1e-6, float(np.quantile(flux, 0.80)) * 0.3),
    )
    strengths = normalized(flux)
    hits = [
        [round(float(f.transient_times[i]), 5), round(float(strengths[i]), 5)]
        for i in peaks
        if f.transient_times[i] < duration and gate[i] > 0.2
    ]
    pulse = np.zeros_like(t)
    for at, strength in hits:
        age = t - at
        pulse = np.maximum(
            pulse,
            np.where(
                (age >= 0) & (age < 0.7),
                strength * np.exp(-np.maximum(age, 0) / 0.13),
                0,
            ),
        )
    beats = np.array([f.tempo.time_to_beat(float(v)) for v in t])
    anchors = np.arange(np.floor(beats[0] / 16) * 16, beats[-1] + 16, 16)
    sections = []
    for anchor in anchors:
        a, b = max(0, f.tempo.beat_to_time(anchor)), min(
            duration, f.tempo.beat_to_time(anchor + 16)
        )
        use = (t >= a) & (t <= b)
        if not use.any() or b <= a:
            continue
        e = tracks[use, 0].mean()
        sections.append(
            dict(
                start=round(a, 5),
                end=round(b, 5),
                energy=float(e),
            )
        )
    # Compare phrase energy inside this track rather than using a global loudness threshold.
    energies = np.array([s["energy"] for s in sections])
    lo, hi = np.quantile(energies, [0.1, 0.9])
    for section in sections:
        level = np.clip((section["energy"] - lo) / max(0.015, hi - lo), 0, 1)
        logits = -12 * (level - np.array([0.5, 0.0, 1.0])) ** 2
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        section["weights"] = weights.tolist()
        section["mode"] = ["orbit", "ribbons", "tunnel"][int(weights.argmax())]
    # Smooth transitions across phrase centres, rather than hard preset cuts.
    centres = [(s["start"] + s["end"]) / 2 for s in sections]
    mixes = np.stack(
        [np.interp(t, centres, [s["weights"][i] for s in sections]) for i in range(3)],
        1,
    )
    # Track-wide spectral balance influences the palette; no genre-to-template label.
    balance = raw.mean(0)
    centroid = float(np.dot(balance, np.linspace(0, 1, 128)) / max(1e-8, balance.sum()))
    hue = (0.46 + 0.08 * centroid + (seed % 31) * 0.001) % 1
    return dict(
        profile=PROFILE,
        sample_rate=30,
        duration=duration,
        seed=int(seed),
        hue=hue,
        columns=[
            "energy",
            "bass",
            "mid",
            "high",
            "pulse",
            "beat",
            "orbit",
            "ribbons",
            "tunnel",
        ],
        times=np.round(t, 5).tolist(),
        controls=np.round(np.c_[tracks, pulse, beats, mixes], 5).tolist(),
        hits=hits,
        sections=sections,
    )


def polish(score, audio, features, output, seed=42):
    if Path(score).resolve() == Path(output).resolve():
        raise ValueError("Use a separate output path to preserve the original score")
    original = SceneScore.load(score)
    f = Features(features)
    digest = sha256(audio)
    if f.metadata["audio_sha256"] != digest:
        raise ValueError("Features do not belong to supplied audio")
    source_hash = original.metadata.get("audio", {}).get("audio_sha256")
    if source_hash is not None and source_hash != digest:
        raise ValueError("Source score belongs to another audio file")
    if abs(original.duration - f.metadata["duration"]) > 0.05:
        raise ValueError("Score/audio duration mismatch")
    p = compile_presentation(features, seed)
    result = SceneScore(
        original.duration,
        f.tempo.to_list(),
        metadata={
            "origin": "audio_directed_presentation",
            "render_profile": PROFILE,
            "title": "Music Choreographer / 音乐演出",
            "audio": f.metadata,
            "source_score": str(Path(score).resolve()),
            "source_score_sha256": sha256(score),
            "model_geometry_used": False,
            "method": "Procedural audio-directed output; not evidence of learned choreography",
            "presentation": p,
        },
    )
    begin = f.tempo.time_to_beat(0)
    for i, name in enumerate(LAYERS):
        result.entities.append(
            Entity(i, "circle", width=1, height=1, source={"presentation_layer": name})
        )
        result.events.append(Event(begin, "create", i))
    for section in p["sections"]:
        a, b = [f.tempo.time_to_beat(section[k]) for k in ("start", "end")]
        result.segments.append(
            dict(start=a, end=b, mode=section["mode"], energy=section["energy"])
        )
    result.save(output)
    return result
