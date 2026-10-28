"""Timestamped, frozen audio features; synthetic features are explicitly marked."""
from pathlib import Path
import hashlib
import json, os
import subprocess
import numpy as np
from scipy.signal import stft
from .score import TempoMap


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def decode(path, sr=24000):
    p = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(sr),
            "-",
        ],
        capture_output=True,
        check=True,
    )
    x = np.frombuffer(p.stdout, dtype="<f4").copy()
    if not len(x):
        raise ValueError("Empty audio")
    return x


def extract(path, output, device="cpu", backend="mert", bpm=None, revision="main"):
    x = decode(path)
    duration = len(x) / 24000
    _, times, z = stft(x, fs=24000, nperseg=1024, noverlap=784, boundary="zeros")
    magnitude = np.abs(z)
    # 128 logarithmic frequency bands, 10 ms hop, plus positive spectral flux.
    edges = np.unique(np.round(np.geomspace(1, 513, 129)).astype(int))
    bands = np.stack(
        [np.log1p(magnitude[a:b].mean(0) * 100) for a, b in zip(edges, edges[1:])], 1
    )
    bands = np.stack(
        [
            np.interp(np.linspace(0, 1, 128), np.linspace(0, 1, bands.shape[1]), v)
            for v in bands
        ]
    )
    flux = np.maximum(0, np.diff(bands, axis=0, prepend=bands[:1])).mean(1)
    transient = np.c_[bands, flux].astype("float32")
    if backend == "mert":
        import torch
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        name = os.environ.get("CHOREO_MERT_PATH", "m-a-p/MERT-v1-95M")
        processor = Wav2Vec2FeatureExtractor.from_pretrained(name, revision=revision)
        encoder = (
            AutoModel.from_pretrained(name, revision=revision, trust_remote_code=True)
            .to(device)
            .eval()
        )
        encoder.requires_grad_(False)
        vectors = []
        stamps = []
        with torch.inference_mode():
            for start in range(0, len(x), 4 * 24000):
                window = x[start : start + 5 * 24000]
                if len(window) < 400:
                    window = np.pad(window, (0, 400 - len(window)))
                inputs = processor(window, sampling_rate=24000, return_tensors="pt")
                out = encoder(
                    **{k: v.to(device) for k, v in inputs.items()},
                    output_hidden_states=True
                )
                v = torch.stack(out.hidden_states[-4:]).mean(0)[0].float().cpu().numpy()
                # Valid-convolution receptive field is 400 samples; stride 320.
                t = start / 24000 + (np.arange(len(v)) * 320 + 199.5) / 24000
                vectors.append(v)
                stamps.append(t)
        t = np.concatenate(stamps)
        v = np.concatenate(vectors)
        # Windows align on the same 75 Hz lattice; average coincident frames.
        keys = np.rint((t - 199.5 / 24000) * 75).astype(int)
        unique, inv = np.unique(keys, return_inverse=True)
        semantic = np.zeros((len(unique), v.shape[1]), np.float32)
        np.add.at(semantic, inv, v)
        semantic /= np.bincount(inv)[:, None]
        semantic_times = unique / 75 + 199.5 / 24000
        resolved = getattr(encoder.config, "_commit_hash", None) or revision
        local_provenance = Path(name) / "provenance.json"
        if local_provenance.is_file():
            resolved = json.loads(local_provenance.read_text())["revision"]
        del encoder
    elif backend == "spectral":
        # A transparent low-cost baseline/smoke mode, never labelled MERT.
        semantic = transient[:, :128]
        semantic_times = times
        resolved = "spectral-v1"
    else:
        raise ValueError("Unknown audio backend")
    if bpm is None:
        from beat_this.inference import File2Beats

        beats, downbeats = File2Beats(
            checkpoint_path="final0", device=device, dbn=False
        )(str(path))
        tempo = TempoMap.from_beats(beats)
        beat_source = "beat_this:final0"
    else:
        tempo = TempoMap.from_bpm([(0, float(bpm))])
        beats = np.arange(0, duration, 60 / float(bpm))
        downbeats = beats[::4]
        beat_source = "explicit_bpm"
    metadata = dict(
        audio_sha256=sha256(path),
        backend=backend,
        revision=resolved,
        duration=duration,
        sample_rate=24000,
        beat_source=beat_source,
        license="CC-BY-NC-4.0" if backend == "mert" else "code license",
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        semantic=semantic,
        semantic_times=semantic_times,
        transient=transient,
        transient_times=times,
        beats=beats,
        downbeats=downbeats,
        tempo_json=json.dumps(tempo.to_list()),
        metadata_json=json.dumps(metadata),
    )
    return metadata


class Features:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as z:
            self.semantic = z["semantic"]
            self.semantic_times = z["semantic_times"]
            self.transient = z["transient"]
            self.transient_times = z["transient_times"]
            self.metadata = json.loads(str(z["metadata_json"]))
            self.tempo = TempoMap(json.loads(str(z["tempo_json"])))

    def sample(self, times):
        def interp(v, t):
            return np.stack(
                [
                    np.interp(times, t, v[:, i], left=0, right=0)
                    for i in range(v.shape[1])
                ],
                1,
            ).astype("float32")

        return interp(self.semantic, self.semantic_times), interp(
            self.transient, self.transient_times
        )

    def context(self, tempo, start, end, step=None):
        if step is None:
            times = np.arange(tempo.beat_to_time(start), tempo.beat_to_time(end), 0.01)
            beats = np.array([tempo.time_to_beat(t) for t in times])
        else:
            beats = np.arange(start, end, step)
            times = np.array([tempo.beat_to_time(b) for b in beats])
        a, t = self.sample(times)
        return a, t, beats
