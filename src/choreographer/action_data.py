"""Human-chart motion supervision. No audio-to-visual rule labels are used."""
import hashlib
import json
import unicodedata
from pathlib import Path
import numpy as np
from .audio import Features, extract, sha256, decode
from .importer import NativeChart

RATE = 16
MERT_REVISION = "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"
FIELDS = [
    "density",
    "center_x",
    "center_y",
    "spread",
    "travel",
    "spin",
    "zoom",
    "opacity",
    "alignment",
    "scale",
    "tilt_sin",
    "tilt_cos",
]


def chart_motion(native, times):
    """Geometric projection, not a claim of full chart or texture reconstruction."""
    result, accents = [], np.zeros(len(times), np.float32)
    previous = None
    for time in times:
        beat = native.tempo.time_to_beat(float(time))
        states = np.array(
            [
                [s[k] for k in ("x", "y", "rotation", "scale_x", "alpha")]
                for s in [native.line_state(i, beat) for i in range(len(native.lines))]
            ]
        )
        visible = (states[:, 4] > 0.02) & (np.abs(states[:, :2]).max(1) < 2)
        weights = states[:, 4] * visible
        count, denom = visible.sum(), max(weights.sum(), 1e-8)
        center = (states[:, :2] * weights[:, None]).sum(0) / denom
        spread = np.sqrt(
            (((states[:, :2] - center) ** 2).sum(1) * weights).sum() / denom
        )
        angle = np.deg2rad(states[:, 2])
        sin, cos = (np.sin(angle) * weights).sum() / denom, (
            np.cos(angle) * weights
        ).sum() / denom
        align = abs((np.exp(2j * angle) * weights).sum() / denom)
        travel = spin = zoom = 0.0
        if previous is not None:
            # Unwrapped native rotations preserve intentional full revolutions.
            travel = (
                (
                    np.linalg.norm(states[:, :2] - previous[:, :2], axis=1) * weights
                ).sum()
                / denom
                * RATE
            )
            spin = (
                (np.abs(states[:, 2] - previous[:, 2]) * weights).sum() / denom * RATE
            )
            zoom = (
                (np.abs(states[:, 3] - previous[:, 3]) * weights).sum() / denom * RATE
            )
        result.append(
            [
                min(1, count / 12),
                *(np.clip(center, -1, 1) * 0.5 + 0.5),
                min(1, spread / 1.5),
                np.tanh(travel / 3),
                np.tanh(spin / 360),
                np.tanh(zoom / 3),
                weights.sum() / max(1, count),
                align,
                min(1, (np.abs(states[:, 3]) * weights).sum() / denom / 3),
                sin * 0.5 + 0.5,
                cos * 0.5 + 0.5,
            ]
        )
        previous = states
    for _, note in native.notes:
        if note.get("isFake", 0):
            continue
        at = native.tempo.beat_to_time(note["startTime"])
        index = int(round(at * RATE))
        if 0 <= index < len(times):
            accents[index] = 1
    values = np.asarray(result, np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite native motion")
    native.line_state.cache_clear()
    return np.clip(values, 0, 1), accents


def grouped_splits(records):
    """Union song aliases and exact audio duplicates before splitting."""
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, r in enumerate(records):
        title = r["song_id"].split("|")[0]
        if "phira" not in r.get("sources", {}) and title.endswith(".0"):
            title = title.rsplit(".", 2)[0]
        title = "".join(
            c for c in unicodedata.normalize("NFKC", title).casefold() if c.isalnum()
        )
        for key in (("song", title), ("audio", r["audio_hash"])):
            if key in seen:
                parent[find(i)] = find(seen[key])
            seen[key] = i
    groups = {}
    for i, r in enumerate(records):
        groups.setdefault(find(i), []).append(r)
    for group in groups.values():
        explicit = {r["split"] for r in group if r.get("split")}
        if len(explicit) > 1:
            raise ValueError("Conflicting pre-existing held-out song splits")
        key = min(r["song_id"].casefold() for r in group)
        bucket = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 10
        split = (
            next(iter(explicit))
            if explicit
            else ("test" if bucket == 0 else "val" if bucket == 1 else "train")
        )
        for r in group:
            r["split"] = split
    return records


def fit_vocabulary(records, root, words=64):
    train = [
        np.load(root / r["tensor"])["motion"] for r in records if r["split"] == "train"
    ]
    data = np.concatenate(train)
    rng = np.random.default_rng(42)
    points = data[rng.choice(len(data), min(40000, len(data)), replace=False)]
    # Each dimension already has documented fixed physical units; no held-out stats.
    centers = points[rng.choice(len(points), words, replace=False)].copy()
    # NumPy elementwise Lloyd updates avoid loading a second Windows OpenMP runtime.
    for _ in range(30):
        labels = np.concatenate(
            [
                ((p[:, None] - centers[None]) ** 2).sum(-1).argmin(1)
                for p in np.array_split(points, max(1, len(points) // 1000))
            ]
        )
        updated = np.stack(
            [
                points[labels == i].mean(0) if np.any(labels == i) else centers[i]
                for i in range(words)
            ]
        )
        if np.max(np.abs(updated - centers)) < 1e-5:
            break
        centers = updated
    np.save(root / "vocabulary.npy", centers.astype(np.float32))
    return centers


def prepare(sources, output, device="cuda"):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    records, errors = [], []
    all_records = []
    for source in sources:
        all_records.extend(json.loads(Path(source).read_text(encoding="utf-8")))
    seen = set()
    for r in all_records:
        chart = Path(r["chart"])
        digest = sha256(chart)
        if digest in seen:
            continue
        seen.add(digest)
        try:
            native = NativeChart(
                json.loads(chart.read_text(encoding="utf-8-sig")), lines_only=True
            )
            audio_hash = sha256(r["audio"])
            target = root / (digest[:16] + ".npz")
            if not target.exists():
                cache = Path(
                    r.get("features") or root / (audio_hash[:16] + ".features.npz")
                )
                if not cache.exists():
                    extract(
                        r["audio"],
                        cache,
                        device,
                        "mert",
                        revision="12af15fef9d0ac838c3f475bfbbf26d2060dd4f5",
                    )
                f = Features(cache)
                if (
                    f.metadata["audio_sha256"] != audio_hash
                    or f.metadata["backend"] != "mert"
                    or f.metadata.get("revision") != MERT_REVISION
                ):
                    raise ValueError("MERT cache provenance mismatch")
                times = np.arange(0, f.metadata["duration"], 1 / RATE)
                motion, accents = chart_motion(native, times)
                semantic, transient = f.sample(times)
                beats = np.array(
                    [f.tempo.time_to_beat(float(t)) for t in times], np.float32
                )
                np.savez_compressed(
                    target,
                    audio=np.c_[semantic, transient].astype(np.float32),
                    motion=motion,
                    accent=accents,
                    times=times,
                    beats=beats,
                    metadata=json.dumps(f.metadata),
                    tempo=json.dumps(f.tempo.to_list()),
                )
            z = np.load(target)
            cached_metadata = json.loads(str(z["metadata"]))
            if (
                cached_metadata["audio_sha256"] != audio_hash
                or z["audio"].shape[1] != 897
                or cached_metadata.get("revision") != MERT_REVISION
            ):
                raise ValueError("Action tensor provenance or feature shape mismatch")
            movement = float(np.abs(np.diff(z["motion"][:, 2:7], axis=0)).mean())
            records.append(
                dict(
                    r,
                    tensor=target.name,
                    chart_sha256=digest,
                    audio_hash=audio_hash,
                    frames=len(z["times"]),
                    movement=movement,
                    line_count=len(native.lines),
                    supervision="native_line_geometry_and_nonfake_note_times; no texture reconstruction",
                )
            )
        except Exception as exc:
            errors.append(dict(song_id=r["song_id"], chart=str(chart), reason=str(exc)))
        (root / "quarantine.json").write_text(
            json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (root / "progress.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"Action data: {len(records)} accepted, {len(errors)} quarantined; {r['song_id']}",
            flush=True,
        )
    grouped_splits(records)
    centers = fit_vocabulary(records, root)
    for r in records:
        motion = np.load(root / r["tensor"])["motion"]
        distance = ((motion[:, None] - centers[None]) ** 2).sum(-1).min(1)
        r["codebook_rmse"] = float(np.sqrt(np.mean(distance) / len(FIELDS)))
    (root / "manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (root / "schema.json").write_text(
        json.dumps(
            dict(
                version="motion-actions-v1",
                fields=FIELDS,
                rate=RATE,
                words=len(centers),
                vocabulary_fitted_on="train only",
                feature_backend="MERT-v1-95M + transient",
            ),
            indent=2,
        )
    )
    return dict(
        accepted=len(records),
        quarantined=len(errors),
        splits={
            s: sum(r["split"] == s for r in records) for s in ("train", "val", "test")
        },
    )
