"""Teacher-forced conditional diagnostic; not a substitute for animation judging."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from choreographer.dataset import SceneDataset, collate
from choreographer.model import ModelConfig, Choreographer, loss

p = argparse.ArgumentParser()
p.add_argument("manifest")
p.add_argument("--checkpoint", required=True)
p.add_argument("--output", required=True)
p.add_argument("--device", default="cpu")
a = p.parse_args()
ck = torch.load(a.checkpoint, map_location=a.device, weights_only=False)
cfg = ModelConfig(**ck["config"])
model = Choreographer(cfg).to(a.device).eval()
model.load_state_dict(ck["model"])
results = {}
for condition in ["correct", "zero", "shuffle", "no_global"]:
    data = SceneDataset(
        a.manifest,
        "test",
        max_entities=cfg.max_entities,
        codebook=ck["codebook"],
        condition=condition if condition != "no_global" else "correct",
    )
    by_song = {}
    with torch.inference_mode():
        for item in data:
            b = {k: v.to(a.device) for k, v in collate([item]).items()}
            stage = 1 if condition == "no_global" else ck["stage"]
            out = model(
                b["cats"][:, :-1],
                b["bins"][:, :-1],
                b["residual"][:, :-1],
                b["audio"],
                b["transient"],
                b["whole"],
                b["plan_index"],
                b["state"],
                stage,
                b["audio_mask"],
                b["whole_mask"],
                next_cats=b["cats"][:, 1:],
            )
            # Compare the same event loss, excluding planner/contrastive auxiliaries.
            value, _ = loss(
                out, b["cats"][:, 1:], b["bins"][:, 1:], b["residual"][:, 1:], stage=1
            )
            by_song.setdefault(item["song_id"], []).append(float(value))
    results[condition] = {k: float(np.mean(v)) for k, v in by_song.items()}
report = {
    "metric": "teacher-forced compound event loss, per-song mean",
    "results": results,
    "limitations": "No preference, perceptual quality, free-running robustness or 20-song generalization claim.",
}
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
Path(a.output).write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(report, ensure_ascii=False))
