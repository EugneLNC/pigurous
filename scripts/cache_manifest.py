import argparse, json
from pathlib import Path
from choreographer.audio import extract, Features

p = argparse.ArgumentParser()
p.add_argument("--source", default="data/prepared/manifest.json")
p.add_argument("--output", default="data/prepared-mert")
p.add_argument("--device", default="cuda")
a = p.parse_args()
root = Path(a.output)
root.mkdir(parents=True, exist_ok=True)
records = json.loads(Path(a.source).read_text(encoding="utf-8"))
accepted = []
errors = []
for r in records:
    cache = root / (r["audio_hash"][:16] + ".npz")
    try:
        if not cache.exists():
            extract(
                r["audio"],
                cache,
                a.device,
                "mert",
                None,
                "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5",
            )
        f = Features(cache)
        if (
            f.metadata["audio_sha256"] != r["audio_hash"]
            or f.metadata["backend"] != "mert"
            or f.metadata["beat_source"] != "beat_this:final0"
        ):
            raise ValueError("Mismatched feature provenance")
        accepted.append(dict(r, features=str(cache.resolve())))
    except Exception as exc:
        errors.append({"song_id": r["song_id"], "reason": str(exc)})
    (root / "manifest.json").write_text(
        json.dumps(accepted, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (root / "quarantine.json").write_text(
        json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"MERT + Beat This: {len(accepted)} accepted, {len(errors)} failed / {len(records)}",
        flush=True,
    )
