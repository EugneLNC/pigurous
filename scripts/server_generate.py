"""One held-out whole-song check of the newly trained server checkpoint."""
import json
import os
from pathlib import Path
import time
import torch
from choreographer.audio import sha256
from choreographer.generation import generate
from choreographer.evaluation import evaluate

root = Path(__file__).resolve().parents[1]
if not (root / ".choreographer-owned").is_file():
    raise RuntimeError("Missing isolated-project ownership marker")
if "CHOREO_SHARED_MEMORY_MB" not in os.environ:
    raise RuntimeError("Launch through shared_gpu.py")
torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(
    int(os.environ["CHOREO_SHARED_MEMORY_MB"])
    * 1024**2
    / torch.cuda.get_device_properties(0).total_memory
)
records = json.loads((root / "data/prepared-mert/manifest.json").read_text())
record = next(r for r in records if r["split"] == "test")
assert sha256(record["audio"]) == record["audio_hash"]
output = root / "runs/server-5090"
destination = output / "generated-test.json"
if destination.exists():
    raise RuntimeError("Preserve previous generation")
begun = time.time()
score = generate(
    output / "inference.pt",
    record["features"],
    destination,
    device="cuda",
    seed=42,
    max_events=2048,
)
score.validate()
report = evaluate([destination], output / "generation-evaluation.json")
report.update(
    song_id=record["song_id"],
    seed=42,
    seconds=time.time() - begun,
    peak_memory_gb=torch.cuda.max_memory_allocated() / 1024**3,
    gpu_uuid=os.environ["CUDA_VISIBLE_DEVICES"],
)
(output / "generation-evaluation.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
if not score.metadata["complete_generation"]:
    raise SystemExit(2)
