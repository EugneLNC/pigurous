"""Download an opt-in research pilot. No redistribution rights are implied."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import urllib.parse

REPO = "7aGiven/Phigros_Resource"


def fetch(url):
    req = urllib.request.Request(
        url, headers={"User-Agent": "music-choreographer-research"}
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        return r.read()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/pilot")
    p.add_argument("--songs", type=int, default=20)
    args = p.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    trees = {
        b: json.loads(
            fetch(f"https://api.github.com/repos/{REPO}/git/trees/{b}?recursive=1")
        )
        for b in ("chart", "music")
    }
    music = {e["path"]: e for e in trees["music"]["tree"] if e["path"].endswith(".ogg")}
    candidates = [
        e
        for e in trees["chart"]["tree"]
        if e["path"].endswith("/EZ.json") and e.get("size", 0) < 3000000
    ]
    # Deterministic selection independent of title ordering, pinned branch commits.
    candidates.sort(key=lambda e: hashlib.sha256(e["path"].encode()).hexdigest())
    manifest = []
    for e in candidates:
        folder = e["path"].split("/")[0]
        name = folder.rsplit(".", 1)[0] + ".ogg"
        if name not in music:
            continue
        sid = hashlib.sha256(folder.encode()).hexdigest()[:16]
        dest = root / sid
        dest.mkdir(exist_ok=True)
        family = (
            folder.rsplit(".", 1)[0] if folder.rsplit(".", 1)[-1].isdigit() else folder
        )
        record = {
            "song_id": family,
            "rights": "public mirror; original chart/audio rights retained; research only; do not redistribute",
            "sources": {},
        }
        for branch, remote, local in [
            ("chart", e["path"], "chart.json"),
            ("music", name, "audio.ogg"),
        ]:
            url = f"https://raw.githubusercontent.com/{REPO}/{trees[branch]['sha']}/{urllib.parse.quote(remote)}"
            target = dest / local
            if not target.exists():
                target.write_bytes(fetch(url))
            record[branch if branch == "chart" else "audio"] = str(target.resolve())
            record["sources"][branch] = {
                "url": url,
                "commit": trees[branch]["sha"],
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        manifest.append(record)
        print(f"{len(manifest)}/{args.songs}: {folder}", flush=True)
        (root / "sources.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if len(manifest) >= args.songs:
            break
    if len(manifest) < args.songs:
        raise RuntimeError("Not enough matched chart/audio pairs")


if __name__ == "__main__":
    main()
