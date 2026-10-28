"""Record measured provenance for the local action experiments."""
import json, hashlib
from pathlib import Path
import numpy as np
from choreographer.audio import sha256

root = Path("artifacts/learned-actions-direct")
records = json.loads(Path("data/actions-v1/manifest.json").read_text(encoding="utf-8"))
evaluation = json.loads((root / "evaluation.json").read_text(encoding="utf-8"))
runs = {
    name: json.loads(Path("runs", name, "summary.json").read_text())
    for name in ("actions-local", "actions-visibility", "actions-direct")
}
rows = evaluation["results"]
server_receipt_path = Path("artifacts/actions-server/upload-receipt.json")
server_upload = (
    dict(
        status="verified after user approval",
        receipt=json.loads(server_receipt_path.read_text()),
    )
    if server_receipt_path.exists()
    else "not performed; explicit payload approval pending"
)
card = dict(
    format="motion-actions-v1",
    checkpoint="runs/actions-direct/best.pt",
    checkpoint_sha256=sha256("runs/actions-direct/best.pt"),
    manifest_sha256=sha256("data/actions-v1/manifest.json"),
    vocabulary_sha256=sha256("data/actions-v1/vocabulary.npy"),
    parameters=runs["actions-direct"]["parameters"],
    actual_device="one local RTX 3070 Ti Laptop GPU; 3400 MiB allocator cap",
    expanded_server_upload=server_upload,
    runs=runs,
    total_experiment_updates=sum(r["steps"] for r in runs.values()),
    splits={
        s: dict(
            charts=sum(r["split"] == s for r in records),
            unique_song_ids=len({r["song_id"] for r in records if r["split"] == s}),
        )
        for s in ("train", "val", "test")
    },
    supervision="Native human chart geometry and note timing; effects use geometry-derived weak labels",
    effect_selection="explicit model head, renderer executes predicted weights",
    evaluation_status="Exploratory after test-output-driven visibility debugging; no blind aesthetic acceptance",
    mean_metrics={
        c: {
            m: float(np.mean([r["metrics"][c][m] for r in rows]))
            for m in (
                "motion_mse",
                "accent_brier",
                "token_change",
                "effect_weight_change",
            )
        }
        for c in ("correct", "zero", "shuffle", "matched_swap")
    },
    mean_motion_baseline=float(np.mean([r["constant_train_mean_mse"] for r in rows])),
    validations=dict(
        python_tests=20,
        typescript_tests=3,
        browser="provenance, seeking, editing, audio playback, mobile",
        examples="two fixed 24-second montages, H264/AAC, 960x540, 30fps",
    ),
    license="MERT CC-BY-NC-4.0; chart/audio creators retain rights; research-only provenance does not grant redistribution",
)
(root / "model-card.json").write_text(
    json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(
    json.dumps(
        dict(
            checkpoint_sha256=card["checkpoint_sha256"],
            updates=card["total_experiment_updates"],
        )
    )
)
