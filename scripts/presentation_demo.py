"""Rebuild actual-audio presentation examples and inspection stills."""
import argparse
import json
from pathlib import Path
from PIL import Image
from choreographer.presentation import polish
from choreographer.score import ScoreRuntime
from choreographer.render import frame, render

p = argparse.ArgumentParser()
p.add_argument("--render", action="store_true")
a = p.parse_args()
root = Path("artifacts/presentation")
root.mkdir(exist_ok=True, parents=True)
records = json.loads(
    Path("data/prepared-mert/manifest.json").read_text(encoding="utf-8")
)
for index, record in enumerate(r for r in records if r["split"] == "test"):
    source = (
        "artifacts/server-5090/generated-test.json" if index == 0 else record["score"]
    )
    name = "nwad" if index == 0 else "verruckt"
    path = root / f"{name}.json"
    score = polish(source, record["audio"], record["features"], path)
    score.metadata["title"] = record["song_id"] + " / 音乐演出"
    score.save(path)
    rt = ScoreRuntime(score)
    times = [score.duration * x for x in (0.1, 0.3, 0.5, 0.8)]
    sheet = Image.new("RGB", (1280, 720))
    for i, t in enumerate(times):
        still = frame(rt, t, 640, 360)
        sheet.paste(still, (i % 2 * 640, i // 2 * 360))
    sheet.save(root / f"{name}-contact.png")
    print(
        json.dumps(
            {
                "song": record["song_id"],
                "onsets": len(score.metadata["presentation"]["hits"]),
                "modes": [s["mode"] for s in score.segments],
                "output": str(path),
            }
        ),
        flush=True,
    )
    if a.render:
        print(
            render(score, root / f"{name}.mp4", 1280, 720, 30, record["audio"]),
            flush=True,
        )
