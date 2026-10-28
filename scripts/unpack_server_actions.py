"""Verify a server result archive and retain its generated files locally."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument(
    "--archive", default="artifacts/actions-server/server-actions-result.zip"
)
parser.add_argument("--sha256", required=True)
parser.add_argument("--output", default="artifacts/actions-server")
args = parser.parse_args()
root = Path(args.output).resolve()
workspace = Path.cwd().resolve()
if not root.is_relative_to(workspace):
    raise ValueError("Output must stay in this workspace")
archive = Path(args.archive)
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
if digest != args.sha256:
    raise ValueError("Server archive hash mismatch")
with zipfile.ZipFile(archive) as z:
    inventory = json.loads(z.read("result-inventory.json"))
    if set(z.namelist()) != {r["path"] for r in inventory} | {"result-inventory.json"}:
        raise ValueError("Unexpected archive contents")
    names = [Path(r["path"]).name for r in inventory]
    if len(set(names)) != len(names):
        raise ValueError("Flattened result filenames collide")
    for r in inventory:
        data = z.read(r["path"])
        if len(data) != r["bytes"] or hashlib.sha256(data).hexdigest() != r["sha256"]:
            raise ValueError("Result file hash mismatch: " + r["path"])
        target = root / Path(r["path"]).name
        if not target.resolve().is_relative_to(root):
            raise ValueError("Unsafe result destination")
        if (
            target.exists()
            and hashlib.sha256(target.read_bytes()).hexdigest() != r["sha256"]
        ):
            raise ValueError(
                "Refuse to overwrite different previous evidence: " + str(target)
            )
    root.mkdir(parents=True, exist_ok=True)
    for r in inventory:
        target = root / Path(r["path"]).name
        if not target.exists():
            target.write_bytes(z.read(r["path"]))
    (root / "result-inventory.json").write_bytes(z.read("result-inventory.json"))
receipt = dict(archive_sha256=digest, verified_files=len(inventory), output=str(root))
(root / "download-receipt.json").write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt))
