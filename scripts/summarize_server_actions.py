"""Validate downloaded server scores and record measured training provenance."""
import json
from pathlib import Path
import numpy as np
from choreographer.audio import sha256
from choreographer.score import SceneScore, ScoreRuntime

root = Path("artifacts/actions-server")
receipt = json.loads((root / "upload-receipt.json").read_text())
download = json.loads((root / "download-receipt.json").read_text())
execution = json.loads((root / "execution.json").read_text())
summary = json.loads((root / "summary.json").read_text())
evaluation = json.loads((root / "evaluation.json").read_text(encoding="utf-8"))
checkpoint_hash = sha256(root / "best.pt")
validation = []
for row in evaluation["results"]:
    score_file = root / (Path(row["tensor"]).stem + ".json")
    score = SceneScore.load(score_file)
    p = score.metadata["presentation"]
    assert score.metadata["checkpoint_sha256"] == checkpoint_hash
    assert score.metadata["complete_generation"]
    assert p["effect_selection"] == "model_head"
    assert len(p["times"]) == len(p["controls"]) == len(p["poses"])
    assert abs(p["times"][-1] - score.duration) < 1e-6
    assert np.isfinite(p["controls"]).all() and np.isfinite(p["poses"]).all()
    runtime = ScoreRuntime(score)
    for time in (0, score.duration * 0.25, score.duration * 0.5, score.duration * 0.75):
        assert len(runtime.state(time)) == 5
    validation.append(
        dict(
            song_id=row["song_id"],
            score=str(score_file),
            duration=score.duration,
            entities=len(score.entities),
            words=row["distinct_words"],
            passed=True,
        )
    )
rows = evaluation["results"]
means = {
    c: {
        m: float(np.mean([r["metrics"][c][m] for r in rows]))
        for m in ("motion_mse", "accent_brier", "token_change", "effect_weight_change")
    }
    for c in ("correct", "zero", "shuffle", "matched_swap")
}
baseline = float(np.mean([r["constant_train_mean_mse"] for r in rows]))
card = dict(
    format="motion-actions-v1",
    checkpoint=str(root / "best.pt"),
    checkpoint_sha256=checkpoint_hash,
    upload=receipt,
    download=download,
    execution=execution,
    training=summary,
    mean_metrics=means,
    mean_motion_baseline=baseline,
    spatial_reconstruction_beats_constant_baseline=means["correct"]["motion_mse"]
    < baseline,
    scores=validation,
    supervision="Human-chart geometry and note timing; geometry-derived weak effect labels",
    evaluation_status="Five exploratory development test songs; not independent aesthetic acceptance",
    runtime_verified="All five scores load, have finite executable model controls, cover complete tracks, and retain five stable entities",
    license="MERT CC-BY-NC-4.0; original chart/music rights retained",
)
(root / "model-card.json").write_text(
    json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(
    json.dumps(
        dict(
            checkpoint_sha256=checkpoint_hash,
            epochs=execution["epochs"],
            steps=execution["optimizer_steps"],
            valid_scores=len(validation),
            mean_metrics=means,
        )
    )
)
