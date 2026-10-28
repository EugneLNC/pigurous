"""Fixed excerpt protocol: first official and first Phira held-out song, no cherry-picking."""
import argparse, json, subprocess
from pathlib import Path
from PIL import Image, ImageDraw
from choreographer.score import SceneScore, ScoreRuntime
from choreographer.render import frame

parser = argparse.ArgumentParser()
parser.add_argument("--directory", default="artifacts/learned-actions")
root = Path(parser.parse_args().directory)
rows = json.loads((root / "evaluation.json").read_text(encoding="utf-8"))["results"]
selected = [rows[0], next(r for r in rows if "phira" in r["audio"])]
manifest = []
for index, row in enumerate(selected):
    score = SceneScore.load(root / (Path(row["tensor"]).stem + ".json"))
    rt = ScoreRuntime(score)
    starts = [10.0, score.duration * 0.45, score.duration * 0.8]
    montage = Image.new("RGB", (960, 810))
    parts = []
    for j, start in enumerate(starts):
        for k, delta in enumerate([1.0, 5.0]):
            im = frame(rt, start + delta, 480, 270)
            ImageDraw.Draw(im).text(
                (12, 12), f"MODEL / {start+delta:.1f}s", fill="white"
            )
            montage.paste(im, (k * 480, j * 270))
        still = frame(rt, start + 2, 960, 540)
        still.save(root / f"example-{index+1}-section-{j+1}.png")
        part = root / f"example-{index+1}-part-{j+1}.mp4"
        parts.append(part)
        cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "960x540",
            "-r",
            "30",
            "-i",
            "-",
            "-ss",
            str(start),
            "-i",
            row["audio"],
            "-map",
            "0:v",
            "-map",
            "1:a:0",
            "-t",
            "8",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(part),
        ]
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            for i in range(240):
                im = frame(rt, start + i / 30, 960, 540)
                ImageDraw.Draw(im).text(
                    (16, 16),
                    f"LEARNED ACTIONS / SONG {index+1} / {start+i/30:.2f}s",
                    fill=(175, 210, 225),
                )
                process.stdin.write(im.tobytes())
        finally:
            process.stdin.close()
        if process.wait():
            raise RuntimeError("FFmpeg excerpt failed")
        print(f"Rendered song {index+1}, section {j+1}", flush=True)
    montage.save(root / f"example-{index+1}-contact.png")
    concat = root / f"example-{index+1}-concat.txt"
    concat.write_text(
        "\n".join("file '" + p.name + "'" for p in parts), encoding="utf-8"
    )
    video = root / f"example-{index+1}.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(video),
        ],
        check=True,
    )
    manifest.append(
        dict(
            song_id=row["song_id"],
            audio=row["audio"],
            score=str(root / (Path(row["tensor"]).stem + ".json")),
            video=str(video),
            excerpts=[dict(start=s, seconds=8) for s in starts],
        )
    )
(root / "examples.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
)
