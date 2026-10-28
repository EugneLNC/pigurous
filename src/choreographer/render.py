from pathlib import Path
import math, subprocess
import numpy as np
from PIL import Image, ImageDraw
from .score import ScoreRuntime, SceneScore


def frame(runtime, time, width=1920, height=1080):
    if runtime.score.metadata.get("render_profile") in ("performance-v1", "mathematical-v2", "choreography-v3"):
        from .performance_render import performance_frame

        return performance_frame(runtime, time, width, height)
    im = Image.new("RGB", (width, height), (8, 12, 22))
    draw = ImageDraw.Draw(im, "RGBA")
    states = runtime.state(time)
    for ent in runtime.score.entities:
        s = states.get(ent.id)
        if not s or s["alpha"] <= 0:
            continue
        x = (s["x"] + 1) * width / 2
        y = (1 - s["y"]) * height / 2
        w = ent.width * s["scale_x"] * width / 2
        h = ent.height * s["scale_y"] * height / 2
        w *= 1 + 0.15 * s.get("pulse", 0)
        h *= 1 + 0.15 * s.get("pulse", 0)
        if not all(math.isfinite(v) for v in [x, y, w, h]):
            continue
        color = tuple(
            int(np.clip(s[k], 0, 1) * 255) for k in ("red", "green", "blue")
        ) + (int(np.clip(s["alpha"], 0, 1) * 255),)
        angle = -math.radians(s["rotation"])
        c, sn = math.cos(angle), math.sin(angle)
        if ent.kind == "circle":
            draw.ellipse(
                (x - abs(w) / 2, y - abs(h) / 2, x + abs(w) / 2, y + abs(h) / 2),
                fill=color,
            )
        else:
            points = [
                (x + dx * c - dy * sn, y + dx * sn + dy * c)
                for dx, dy in [
                    (-w / 2, -h / 2),
                    (w / 2, -h / 2),
                    (w / 2, h / 2),
                    (-w / 2, h / 2),
                ]
            ]
            draw.polygon(points, fill=color)
    return im


def render(score, output, width=1920, height=1080, fps=60, audio=None, preset="medium", threads=None):
    if width < 2 or height < 2 or width % 2 or height % 2 or fps <= 0:
        raise ValueError("Video needs even positive dimensions and positive fps")
    score = SceneScore.load(score) if isinstance(score, (str, Path)) else score
    rt = ScoreRuntime(score)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
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
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
    ]
    if audio:
        cmd += ["-i", str(audio), "-map", "0:v", "-map", "1:a:0", "-c:a", "aac"]
    cmd += [
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-pix_fmt",
        "yuv420p",
        "-t",
        str(score.duration),
        "-movflags",
        "+faststart",
        str(output),
    ]
    if threads is not None:
        cmd[-1:-1] = ["-threads", str(int(threads))]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as p:
        try:
            for i in range(math.ceil(score.duration * fps)):
                p.stdin.write(frame(rt, i / fps, width, height).tobytes())
            p.stdin.close()
            error = p.stderr.read()
            code = p.wait()
        except BaseException:
            p.kill()
            p.wait()
            raise
        if code:
            raise RuntimeError(error.decode(errors="replace"))
    return {
        "frames": math.ceil(score.duration * fps),
        "width": width,
        "height": height,
        "fps": fps,
        "path": str(output),
    }
