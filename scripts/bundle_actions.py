"""Inventory-bound upload: code and downsampled research tensors, no raw music."""
from pathlib import Path
import json, hashlib, zipfile

root = Path.cwd()
data = root / "data/actions-v1"
records = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
files = [
    Path("src/choreographer") / p.name
    for p in (root / "src/choreographer").glob("*.py")
]
files += [Path("scripts/shared_gpu.py")]
files += [Path("tests/test_actions.py")]
files += [Path("data/actions-v1") / r["tensor"] for r in records]
files += [
    Path("data/actions-v1") / name
    for name in ("manifest.json", "quarantine.json", "vocabulary.npy", "schema.json")
]
files = sorted(set(files))
manifest = [
    dict(
        path=p.as_posix(),
        bytes=(root / p).stat().st_size,
        sha256=hashlib.sha256((root / p).read_bytes()).hexdigest(),
    )
    for p in files
]
output = root / "artifacts/actions-research.zip"
with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
    for p in files:
        archive.write(root / p, p.as_posix())
    archive.writestr("actions-upload-inventory.json", json.dumps(manifest, indent=2))
(root / "artifacts/actions-upload-inventory.json").write_text(
    json.dumps(manifest, indent=2)
)
print(
    json.dumps(
        dict(
            files=len(files),
            bytes=output.stat().st_size,
            sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
            chart_records=len(records),
        )
    )
)
