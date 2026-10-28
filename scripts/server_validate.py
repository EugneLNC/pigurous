"""Bounded validation in our isolated server directory; no global modifications."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
if not (root / ".choreographer-owned").is_file():
    raise RuntimeError("Missing isolated-project ownership marker")
os.chdir(root)
os.environ.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")
# CPU checks see no GPU; shared_gpu.py exposes one permitted UUID per GPU task.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
output = root / "runs/server-5090"
if output.exists():
    raise RuntimeError("Validation output already exists; preserve prior evidence")
output.mkdir(parents=True)
report = {"root": str(root), "started_unix": time.time(), "runs": []}


def run(label, command, gpu=False):
    if gpu:
        command = [
            sys.executable,
            "scripts/shared_gpu.py",
            "--required-mb",
            "8192",
            "--",
        ] + command
    log = output / f"{label}.log"
    begun = time.time()
    print(f"Starting {label}; log: {log}", flush=True)
    with log.open("w") as stream:
        process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    record = {
        "name": label,
        "exit_code": process.returncode,
        "seconds": time.time() - begun,
        "command": command,
        "log": str(log),
    }
    report["runs"].append(record)
    (output / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(record), flush=True)
    if process.returncode:
        raise RuntimeError(f"{label} failed; see {log}")


run("tests", [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider"])
run("full-model-smoke", [sys.executable, "-B", "scripts/gpu_smoke.py"], gpu=True)
base = [
    sys.executable,
    "-B",
    "-m",
    "choreographer.cli",
    "train",
    "data/prepared-mert/manifest.json",
    "--output",
    str(output),
    "--config",
    "configs/research.json",
    "--device",
    "cuda",
    "--batch-size",
    "2",
    "--accumulate",
    "16",
    "--memory-limit-gb",
    "30",
]
for label, stage, epochs in [
    ("local", 1, 5),
    ("planner", 2, 8),
    ("alignment", 3, 9),
    ("resume", 3, 10),
]:
    command = base + ["--stage", str(stage), "--epochs", str(epochs)]
    if stage != 1:
        command += ["--resume", str(output / "last.pt")]
    run(label, command, gpu=True)
run(
    "export",
    [
        sys.executable,
        "-B",
        "scripts/export_checkpoint.py",
        str(output / "last.pt"),
        str(output / "inference.pt"),
        "--expected-stage",
        "3",
    ],
)
report["completed_unix"] = time.time()
(output / "validation.json").write_text(json.dumps(report, indent=2))
print("Server validation completed", flush=True)
