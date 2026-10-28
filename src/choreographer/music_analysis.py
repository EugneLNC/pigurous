"""Observed whole-track structure, not instrument/emotion transcription."""
import json
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
from .audio import Features, sha256

VERSION = "audio-analysis-v1"


def analyze_features(features, audio=None):
    f = Features(features)
    if audio and sha256(audio) != f.metadata["audio_sha256"]:
        raise ValueError("Audio/cache mismatch")
    duration = float(f.metadata["duration"])
    start, end = f.tempo.time_to_beat(0), f.tempo.time_to_beat(duration)
    beats = np.r_[start, np.arange(np.ceil(start + 1e-7), end), end]
    times = np.array([f.tempo.beat_to_time(float(b)) for b in beats])
    sem, tr = [], []
    for a, b in zip(times[:-1], times[1:]):
        sm = (f.semantic_times >= a) & (f.semantic_times < b)
        tm = (f.transient_times >= a) & (f.transient_times < b)
        sem.append(f.semantic[sm].mean(0) if sm.any() else f.sample([(a+b)/2])[0][0])
        tr.append(f.transient[tm].mean(0) if tm.any() else f.sample([(a+b)/2])[1][0])
    sem, tr = np.asarray(sem), np.asarray(tr)
    energy = tr[:, :128].mean(1)
    reference = max(float(np.quantile(energy, .95)), 1e-6)
    energy = np.clip(energy / reference, 0, 1)
    unit = sem / np.maximum(np.linalg.norm(sem, axis=1, keepdims=True), 1e-8)
    novelty = np.r_[0., np.maximum(0., 1 - (unit[1:] * unit[:-1]).sum(1))]
    novelty += .15 * np.r_[0., np.abs(np.diff(energy))]
    peaks, _ = find_peaks(novelty, distance=8, prominence=.025)
    cuts = [0]
    while len(sem) - cuts[-1] > 24:
        candidates = [int(p) for p in peaks if cuts[-1]+8 <= p <= cuts[-1]+24]
        cut = max(candidates, key=lambda p: novelty[p]) if candidates else cuts[-1]+16
        cuts.append(cut)
    cuts.append(len(sem))
    sections, descriptors = [], []
    for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
        descriptors.append(sem[a:b].mean(0))
        sections.append(dict(index=i, start_beat=float(beats[a]), end_beat=float(beats[b]),
            start_seconds=float(times[a]), end_seconds=float(times[b]),
            energy=round(float(energy[a:b].mean()), 4),
            energy_change=round(float(energy[b-1]-energy[a]), 4),
            onset_activity=round(float(tr[a:b, -1].mean()), 5),
            bands=[round(float(v), 4) for v in [tr[a:b, :35].mean(), tr[a:b, 35:85].mean(), tr[a:b, 85:128].mean()]],
            boundary_confidence=round(float(min(1, novelty[a]*4)), 4),
            boundary_source="feature_novelty" if a in peaks else "bounded_analysis_window"))
    d = np.asarray(descriptors)
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-8)
    similarity = d @ d.T
    repeats = [dict(a=i, b=j, similarity=round(float(similarity[i,j]), 4))
               for i in range(len(d)) for j in range(i+2, len(d)) if similarity[i,j] >= .85]
    repeats.sort(key=lambda x: -x["similarity"])
    return dict(version=VERSION, audio_sha256=f.metadata["audio_sha256"], duration=duration,
        tempo_map=f.tempo.to_list(), feature_revision=f.metadata.get("revision"),
        sections=sections, similar_sections=repeats[:64],
        observations={"bands_are_not_instruments":True, "section_labels":None,
                      "instrument_transcription":None, "emotion_labels":None})


def analyze(audio, features, output):
    result = analyze_features(features, audio)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
