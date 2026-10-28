"""Public Phira chart packages, bounded and provenance-preserving; no login."""
import argparse, hashlib, json, re, stat, zipfile
from pathlib import Path
import requests

p = argparse.ArgumentParser()
p.add_argument("--charts", type=int, default=20)
p.add_argument("--output", default="data/phira")
p.add_argument("--start-page", type=int, default=1)
a = p.parse_args()
root = Path(a.output)
root.mkdir(parents=True, exist_ok=True)
session = requests.Session()
session.headers["User-Agent"] = "MusicChoreographer-research/0.1"
listing = session.get(
    "https://phira.5wyxi.com/chart",
    params={
        "pageNum": 20,
        "page": a.start_page,
        "order": "-rating",
        "division": "regular",
        "type": 0,
    },
    timeout=30,
)
listing.raise_for_status()
items = listing.json()["results"]
for page in range(a.start_page + 1, a.start_page + (a.charts + 19) // 20):
    response = session.get(
        "https://phira.5wyxi.com/chart",
        params={
            "pageNum": 20,
            "page": page,
            "order": "-rating",
            "division": "regular",
            "type": 0,
        },
        timeout=30,
    )
    response.raise_for_status()
    items.extend(response.json()["results"])
records = []
errors = []
for meta in items[: a.charts]:
    dest = root / str(meta["id"])
    dest.mkdir(exist_ok=True)
    archive = dest / "source.zip"
    try:
        if not archive.exists():
            with session.get(meta["file"], stream=True, timeout=30) as response:
                response.raise_for_status()
                count = 0
                if int(response.headers.get("Content-Length", 0)) > 64 * 1024**2:
                    raise ValueError("Package exceeds 64 MiB pilot limit")
                with (dest / "source.part").open("wb") as f:
                    for block in response.iter_content(65536):
                        count += len(block)
                        if count > 64 * 1024**2:
                            raise ValueError("Package exceeds pilot limit")
                        f.write(block)
            (dest / "source.part").replace(archive)
        with zipfile.ZipFile(archive) as z:
            if sum(i.file_size for i in z.infolist()) > 256 * 1024**2:
                raise ValueError("Unpacked size exceeds pilot limit")
            for info in z.infolist():
                if info.is_dir():
                    continue
                target = (dest / info.filename).resolve()
                if not target.is_relative_to(dest.resolve()) or stat.S_ISLNK(
                    info.external_attr >> 16
                ):
                    raise ValueError("Unsafe ZIP member")
                if target.suffix.lower() not in (
                    ".json",
                    ".ogg",
                    ".mp3",
                    ".wav",
                    ".yml",
                    ".yaml",
                    ".txt",
                ):
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as source:
                    target.write_bytes(source.read())
        charts = []
        for candidate in dest.rglob("*.json"):
            try:
                raw = json.loads(candidate.read_text(encoding="utf-8-sig"))
                if "judgeLineList" in raw:
                    charts.append(candidate)
            except (ValueError, UnicodeError):
                pass
        audio = [
            f for f in dest.rglob("*") if f.suffix.lower() in (".ogg", ".mp3", ".wav")
        ]
        if len(charts) != 1 or len(audio) != 1:
            raise ValueError(
                f"Ambiguous package: {len(charts)} charts, {len(audio)} audio files; requires explicit selection"
            )
        family = (
            re.sub(r"\s+", "", meta["name"].casefold())
            + "|"
            + meta["composer"].casefold()
        )
        records.append(
            {
                "song_id": family,
                "chart": str(charts[0].resolve()),
                "audio": str(audio[0].resolve()),
                "rights": "Public fan chart; original creators retain rights; research provenance only, not redistribution permission",
                "sources": {
                    "phira": {
                        "id": meta["id"],
                        "url": f"https://phira.5wyxi.com/chart/{meta['id']}",
                        "package_url": meta["file"],
                        "package_sha256": hashlib.sha256(
                            archive.read_bytes()
                        ).hexdigest(),
                        "chart_updated": meta["chartUpdated"],
                    }
                },
            }
        )
        (dest / "phira-metadata.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as exc:
        errors.append({"id": meta["id"], "reason": str(exc)})
    (root / "sources.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (root / "quarantine.json").write_text(
        json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Phira {meta['id']}: {len(records)} accepted packages, {len(errors)} failures",
        flush=True,
    )
