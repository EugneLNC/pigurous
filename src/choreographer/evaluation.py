import csv, json
from pathlib import Path
import numpy as np
from .score import SceneScore, TempoMap
from .representation import descriptor


def evaluate(scores, output, votes=None, training_scores=None, seed=42):
    records = []
    features = []
    for path in scores:
        score = SceneScore.load(path)
        tempo = TempoMap(score.tempo_map)
        a = tempo.time_to_beat(0)
        b = tempo.time_to_beat(score.duration)
        ds = np.stack(
            [descriptor(score, t, min(t + 16, b)) for t in np.arange(a, b, 16)]
        )
        records.append(
            dict(
                path=str(path),
                song_id=score.metadata.get(
                    "song_id",
                    score.metadata.get("audio", {}).get("audio_sha256", str(path)),
                ),
                valid=True,
                entities=len(score.entities),
                events=len(score.events),
                duration=score.duration,
                degenerate_empty=not bool(score.entities),
                complete_generation=score.metadata.get("complete_generation", True),
            )
        )
        features.append(ds.mean(0))
    report = {
        "scores": records,
        "claim": "feasibility only; no aesthetic or generalization claim without held-out blind ratings",
    }
    if training_scores:
        train = []
        for path in training_scores:
            s = SceneScore.load(path)
            tm = TempoMap(s.tempo_map)
            train.append(descriptor(s, tm.time_to_beat(0), tm.time_to_beat(s.duration)))
        distances = np.linalg.norm(
            np.array(features)[:, None] - np.array(train)[None], axis=-1
        )
        report["descriptor_nearest_neighbors"] = [
            {
                "source": str(scores[i]),
                "neighbor": str(training_scores[j]),
                "distance": float(distances[i, j]),
            }
            for i, j in enumerate(distances.argmin(1))
        ]
        report[
            "retrieval_limit"
        ] = "Descriptor retrieval is a screening signal, not proof of trajectory originality."
    if votes:
        with open(votes, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        grouped = {}
        for r in rows:
            value = float(r["full_preferred"])
            if value not in (0, 0.5, 1):
                raise ValueError("full_preferred must be 0, 0.5, or 1")
            grouped.setdefault(r["song_id"], []).append(value)
        means = np.array([np.mean(v) for v in grouped.values()])
        rng = np.random.default_rng(seed)
        boot = rng.choice(means, (10000, len(means)), replace=True).mean(1)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        rate = float(means.mean())
        report["blind_preference"] = {
            "songs": len(means),
            "rate": rate,
            "song_bootstrap_ci95": [float(lo), float(hi)],
            "acceptance_met": len(means) >= 20 and rate >= 0.6 and lo > 0.5,
        }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def blind_materials(pairs, output, seed=42):
    rng = np.random.default_rng(seed)
    public = []
    key = []
    for i, p in enumerate(pairs):
        swap = bool(rng.integers(2))
        a, b = (p["full"], p["local"]) if not swap else (p["local"], p["full"])
        public.append(
            {
                "trial": i,
                "song_id": p["song_id"],
                "A": a,
                "B": b,
                "criteria": [
                    "timing",
                    "composition",
                    "coherence",
                    "diversity",
                    "watch_again",
                ],
            }
        )
        key.append({"trial": i, "full": "B" if swap else "A"})
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    (root / "trials.json").write_text(json.dumps(public, indent=2), encoding="utf-8")
    (root / "researcher_key.json").write_text(
        json.dumps(key, indent=2), encoding="utf-8"
    )
