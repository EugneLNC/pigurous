"""Pinned upstream weights/code, useful when Hugging Face cache redirects stall."""
from pathlib import Path
import hashlib, json, requests

REVISION = "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"
WEIGHT_SHA = "a2b8b747f72c06e0595aeae41ae5473f4364938c6b39b2c58be38c48e6bd3fcd"
root = Path("data/models/MERT-v1-95M")
root.mkdir(parents=True, exist_ok=True)
manifest = {
    "repository": "m-a-p/MERT-v1-95M",
    "revision": REVISION,
    "license": "CC-BY-NC-4.0",
    "files": {},
}
for name in [
    "config.json",
    "preprocessor_config.json",
    "configuration_MERT.py",
    "modeling_MERT.py",
    "pytorch_model.bin",
]:
    target = root / name
    url = f"https://huggingface.co/m-a-p/MERT-v1-95M/resolve/{REVISION}/{name}?download=true"
    if not target.exists():
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with target.with_suffix(target.suffix + ".part").open("wb") as f:
                count = 0
                for block in r.iter_content(65536):
                    f.write(block)
                    count += len(block)
                    if count % (16 * 1024 * 1024) == 0:
                        print(f"{name}: {count//1024//1024} MiB", flush=True)
        target.with_suffix(target.suffix + ".part").replace(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if name == "pytorch_model.bin" and digest != WEIGHT_SHA:
        raise ValueError("Weight SHA256 mismatch")
    manifest["files"][name] = digest
    print(name, digest, flush=True)
(root / "provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
