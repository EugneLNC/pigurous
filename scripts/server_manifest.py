"""Remap our copied manifests only, never discover or modify other directories."""
from pathlib import Path, PureWindowsPath
import json

root = Path(__file__).resolve().parents[1]
if not (root / ".choreographer-owned").exists():
    raise RuntimeError("Missing isolated-project marker")
for rel in [
    "data/pilot/sources.json",
    "data/prepared/manifest.json",
    "data/prepared-mert/manifest.json",
]:
    path = root / rel
    if not path.exists():
        continue
    records = json.loads(path.read_text())
    for r in records:
        for key in ["chart", "audio", "score", "features"]:
            if key in r:
                p = PureWindowsPath(r[key])
                base = "pilot" if key in ("chart", "audio") else "prepared"
                target = root / "data" / base / p.parent.name / p.name
                if key == "features" and p.parent.name == "prepared-mert":
                    target = root / "data/prepared-mert" / p.name
                if not target.is_file():
                    raise FileNotFoundError(target)
                r[key] = str(target)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2))
print("Only copied project manifests remapped:", root)
