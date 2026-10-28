from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from .score import SceneScore, TempoMap
from .audio import Features
from .representation import blocks, descriptor, grouped_split


def prepare(
    sources, output, lines_only=False, backend="spectral", device="cpu", bpm=None
):
    from .audio import decode, extract, sha256
    from .importer import convert_chart, reconstruction_audit

    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    records = grouped_split(json.loads(Path(sources).read_text(encoding="utf-8")))
    result = []
    quarantine = []
    for i, r in enumerate(records):
        sid = sha256(r["audio"])[:16]
        dest = root / sid
        dest.mkdir(exist_ok=True)
        try:
            duration = len(decode(r["audio"])) / 24000
            score, native = convert_chart(
                r["chart"], duration, r["song_id"], lines_only=lines_only
            )
            audit = reconstruction_audit(score, native, frames=120)
            if not audit["passed"]:
                raise ValueError(f"Reconstruction audit failed: {audit}")
            score.metadata["source_hashes"] = r.get("sources", {})
            score.metadata["audit"] = audit
            score.save(dest / "score.json")
            cache = dest / "features.npz"
            if not cache.exists():
                extract(r["audio"], cache, device, backend, bpm)
            feature = Features(cache)
            if (
                feature.metadata["audio_sha256"] != sha256(r["audio"])
                or feature.metadata["backend"] != backend
            ):
                raise ValueError("Cache provenance mismatch")
            # Automatic and authoritative chart tempos are both retained.
            result.append(
                dict(
                    r,
                    score=str((dest / "score.json").resolve()),
                    features=str(cache.resolve()),
                    audio_hash=sha256(r["audio"]),
                )
            )
        except Exception as exc:
            quarantine.append(
                dict(
                    song_id=r["song_id"],
                    reason=f"{type(exc).__name__}: {exc}",
                    chart=r["chart"],
                )
            )
        (root / "manifest.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (root / "quarantine.json").write_text(
            json.dumps(quarantine, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"Prepared {i+1}/{len(records)}; accepted={len(result)}, quarantined={len(quarantine)}",
            flush=True,
        )
    return result, quarantine


def fit_codes(descriptors, k=64, seed=42):
    x = np.asarray(descriptors, np.float32)
    rng = np.random.default_rng(seed)
    mean = x.mean(0)
    std = x.std(0).clip(0.01)
    z = (x - mean) / std
    centers = z[rng.choice(len(z), min(k, len(z)), replace=False)].copy()
    for _ in range(30):
        labels = ((z[:, None] - centers) ** 2).sum(-1).argmin(-1)
        newer = np.stack(
            [
                z[labels == i].mean(0) if np.any(labels == i) else c
                for i, c in enumerate(centers)
            ]
        )
        if np.allclose(newer, centers):
            break
        centers = newer
    return {"centers": centers, "mean": mean, "std": std}


def code_labels(desc, codebook):
    z = (desc - codebook["mean"]) / codebook["std"]
    return ((z[:, None] - codebook["centers"]) ** 2).sum(-1).argmin(-1)


class SceneDataset(Dataset):
    def __init__(
        self,
        manifest,
        split="train",
        max_events=2048,
        max_entities=128,
        codebook=None,
        condition="correct",
    ):
        records = json.loads(Path(manifest).read_text(encoding="utf-8"))
        self.items = []
        self.songs = []
        self.quarantine = []
        # Detect split leakage by both declared family and exact audio bytes.
        for key in ("song_id", "audio_hash"):
            seen = {}
            for r in records:
                if key not in r:
                    continue
                if r[key] in seen and seen[r[key]] != r["split"]:
                    raise ValueError(f"{key} leaks across splits")
                seen[r[key]] = r["split"]
        for r in records:
            if r["split"] != split:
                continue
            score = SceneScore.load(r["score"])
            feature = Features(r["features"])
            tempo = TempoMap(score.tempo_map)
            begin = tempo.time_to_beat(0)
            end = tempo.time_to_beat(score.duration)
            anchors = np.arange(begin, end, 16)
            whole = feature.context(tempo, begin, end, step=1)[0]
            if condition == "zero":
                whole[:] = 0
            elif condition == "shuffle":
                np.random.default_rng(42).shuffle(whole)
            plans = np.stack([descriptor(score, a, min(a + 16, end)) for a in anchors])
            try:
                chunks = list(blocks(score, max_events, max_entities))
            except ValueError as e:
                self.quarantine.append({"song_id": r["song_id"], "reason": str(e)})
                continue
            previous = None
            for chunk in chunks:
                audio, transient, _ = feature.context(
                    tempo, chunk["anchor"] - 4, chunk["end"] + 4
                )
                if condition == "zero":
                    audio[:] = 0
                    transient[:] = 0
                elif condition == "shuffle":
                    rng = np.random.default_rng(42)
                    rng.shuffle(audio)
                    rng.shuffle(transient)
                chunk.update(
                    audio=audio,
                    transient=transient,
                    whole=whole,
                    plan=plans,
                    plan_index=int((chunk["anchor"] - begin) // 16),
                    song_id=r["song_id"],
                    bpm=tempo.points[0].bpm,
                    energy=float(transient[:, :128].mean()),
                )
                chunk["index"] = len(self.items)
                chunk["previous"] = previous
                previous = chunk["index"]
                self.items.append(chunk)
            self.songs.append(r)
            print(
                f"{split}: {len(self.songs)} songs, {len(self.items)} blocks",
                flush=True,
            )
        if not self.items:
            raise ValueError(
                f"No usable {split} samples. Quarantine: {self.quarantine}"
            )
        self.codebook = codebook or fit_codes(
            np.concatenate([item["plan"] for item in self.items])
        )
        for item in self.items:
            item["codes"] = code_labels(item["plan"], self.codebook)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate(items):
    out = {}
    for key in (
        "cats",
        "bins",
        "residual",
        "audio",
        "transient",
        "whole",
        "plan",
        "codes",
        "state",
    ):
        arrays = [torch.as_tensor(i[key]) for i in items]
        fill = float("nan") if key == "plan" else 0
        out[key] = torch.nn.utils.rnn.pad_sequence(
            arrays, batch_first=True, padding_value=fill
        )
    out["plan_index"] = torch.tensor([i["plan_index"] for i in items])
    out["indices"] = torch.tensor([i["index"] for i in items])
    for key in ("audio", "whole"):
        out[key + "_mask"] = (
            torch.arange(out[key].shape[1])[None]
            >= torch.tensor([len(i[key]) for i in items])[:, None]
        )
    return out


class HardNegativeBatches:
    """Group different songs with nearby BPM/energy, without duplicating indices."""

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batches = list(self._make())

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)

    def _make(self):
        remaining = set(np.random.permutation(len(self.dataset)).tolist())
        while remaining:
            anchor = int(np.random.choice(list(remaining)))
            remaining.remove(anchor)
            batch = [anchor]
            item = self.dataset[anchor]
            songs = {item["song_id"]}
            while remaining and len(batch) < self.batch_size:
                candidates = [
                    i for i in remaining if self.dataset[i]["song_id"] not in songs
                ]
                if not candidates:
                    break
                other = min(
                    candidates,
                    key=lambda i: abs(np.log(self.dataset[i]["bpm"] / item["bpm"]))
                    + abs(self.dataset[i]["energy"] - item["energy"]),
                )
                batch.append(other)
                songs.add(self.dataset[other]["song_id"])
                remaining.remove(other)
            yield batch
