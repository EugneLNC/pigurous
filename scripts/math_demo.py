"""Mathematical execution of fixed server predictions; separate authored gallery."""
import argparse
import json
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw
from choreographer.math_presentation import mathematical_score, PROFILE
from choreographer.score import SceneScore, ScoreRuntime, Entity, Event, TempoMap
from choreographer.render import frame
from choreographer.audio import sha256

parser = argparse.ArgumentParser()
parser.add_argument("--render", action="store_true")
args = parser.parse_args()
root = Path("artifacts/mathematical-v2")
root.mkdir(parents=True, exist_ok=True)
reference = Path("artifacts/actions-server")
rows = json.loads((reference / "evaluation.json").read_text(encoding="utf-8"))[
    "results"
]
records = []
for row in rows:
    source = reference / (Path(row["tensor"]).stem + ".json")
    output = root / (Path(row["tensor"]).stem + ".json")
    report = mathematical_score(source, output)
    records.append(
        dict(
            song_id=row["song_id"],
            audio=row["audio"],
            source=str(source),
            score=str(output),
            report=report,
        )
    )
(root / "manifest.json").write_text(
    json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
)
Path("preview/public/mathematical.json").write_bytes(
    Path(records[0]["score"]).read_bytes()
)
Path("preview/public/model-reference.json").write_bytes(
    Path(records[0]["source"]).read_bytes()
)

gallery = Image.new("RGB", (1440, 480), (3, 6, 15))
for family, name in enumerate(
    ["HARMONIC ROSE", "LISSAJOUS / 3:2", "HYPOTROCHOID / 5:2"]
):
    controls = [0.8, 0.4, 0.3, 0.2, 0, 12, *[int(i == family) for i in range(3)]]
    p = dict(
        profile=PROFILE,
        duration=8,
        seed=42,
        hue=0.54,
        times=[0, 8],
        controls=[controls, controls],
        poses=[[0, 0, 0, 1], [0, 0, 0, 1]],
        hits=[],
    )
    score = SceneScore(
        8,
        TempoMap.from_bpm([(0, 120)]).to_list(),
        metadata=dict(
            origin="authored_primitive_gallery", render_profile=PROFILE, presentation=p
        ),
    )
    for i in range(5):
        score.entities.append(Entity(i, "circle", width=1, height=1))
        score.events.append(Event(0, "create", i))
    image = frame(ScoreRuntime(score), 3, 480, 480)
    draw = ImageDraw.Draw(image)
    draw.text((18, 18), name, fill=(185, 215, 230))
    draw.text(
        (18, 456), "AUTHORED ACTION / NOT MODEL CHOREOGRAPHY", fill=(110, 140, 160)
    )
    gallery.paste(image, (family * 480, 0))
gallery.save(root / "action-gallery.png")
selected = [records[0], next(r for r in records if "phira" in r["audio"])]


def render_clip(runtime, start, output, audio, comparison=None):
    width, height = (1280, 360) if comparison else (960, 540)
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            "30",
            "-i",
            "-",
            "-ss",
            str(start),
            "-i",
            audio,
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
            str(output),
        ],
        stdin=subprocess.PIPE,
    )
    try:
        for i in range(240):
            time = start + i / 30
            if comparison:
                image = Image.new("RGB", (width, height))
                image.paste(frame(comparison, time, 640, 360), (0, 0))
                image.paste(frame(runtime, time, 640, 360), (640, 0))
                draw = ImageDraw.Draw(image)
                draw.text((14, 14), "PREVIOUS / SAME MODEL CONTROLS", fill="white")
                draw.text(
                    (654, 14), "MATHEMATICAL / CONSTRAINED EXECUTION", fill="white"
                )
            else:
                image = frame(runtime, time, width, height)
                ImageDraw.Draw(image).text(
                    (16, 16),
                    f"MATHEMATICAL ACTIONS / {time:.2f}s",
                    fill=(170, 205, 225),
                )
            process.stdin.write(image.tobytes())
    finally:
        process.stdin.close()
    if process.wait():
        raise RuntimeError("Mathematical video encoding failed")


def concatenate(parts, output):
    listing = output.with_suffix(".concat.txt")
    listing.write_text(
        "\n".join("file '" + p.name + "'" for p in parts), encoding="utf-8"
    )
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
            str(listing),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )


examples = []
for index, record in enumerate(selected):
    score = SceneScore.load(record["score"])
    runtime = ScoreRuntime(score)
    starts = [10.0, score.duration * 0.45, score.duration * 0.8]
    contact = Image.new("RGB", (960, 810))
    for j, start in enumerate(starts):
        for k, delta in enumerate([1.0, 5.0]):
            im = frame(runtime, start + delta, 480, 270)
            ImageDraw.Draw(im).text(
                (12, 12), f"{start+delta:.1f}s", fill=(185, 210, 225)
            )
            contact.paste(im, (k * 480, j * 270))
        frame(runtime, start + 2, 960, 540).save(
            root / f"example-{index+1}-section-{j+1}.png"
        )
    contact.save(root / f"example-{index+1}-contact.png")
    if args.render:
        parts = []
        for j, start in enumerate(starts):
            part = root / f"example-{index+1}-part-{j+1}.mp4"
            parts.append(part)
            render_clip(runtime, start, part, record["audio"])
            print(f"Math video {index+1}, section {j+1}", flush=True)
        concatenate(parts, root / f"example-{index+1}.mp4")
        if index == 0:
            before = ScoreRuntime(SceneScore.load(record["source"]))
            parts = []
            for j, start in enumerate(starts):
                part = root / f"comparison-part-{j+1}.mp4"
                parts.append(part)
                render_clip(runtime, start, part, record["audio"], before)
            concatenate(parts, root / "before-after.mp4")
    examples.append(
        dict(
            song_id=record["song_id"],
            score=record["score"],
            source=record["source"],
            starts=starts,
            video=str(root / f"example-{index+1}.mp4"),
        )
    )
(root / "examples.json").write_text(
    json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(
    json.dumps(
        dict(
            compiled=len(records),
            gallery=str(root / "action-gallery.png"),
            source_checkpoint=sha256("artifacts/actions-server/best.pt"),
            weights_changed=False,
        )
    )
)
