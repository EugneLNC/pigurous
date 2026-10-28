"""Seekable procedural geometry shared by the offline and browser presentation."""
import colorsys
import math
from bisect import bisect_right
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops, ImageEnhance

TAU = math.tau


def sample(p, time):
    i = max(0, min(len(p["times"]) - 2, bisect_right(p["times"], time) - 1))
    a, b = p["times"][i : i + 2]
    u = max(0, min(1, (time - a) / max(1e-8, b - a)))
    value = (
        np.asarray(p["controls"][i]) * (1 - u) + np.asarray(p["controls"][i + 1]) * u
    )
    value[4] = max(
        (
            strength * math.exp(-(time - at) / 0.13)
            for at, strength in p["hits"]
            if 0 <= time - at < 0.7
        ),
        default=0,
    )
    return value


def primitives(p, time):
    e, bass, mid, high, pulse, beat, orbit, ribbons, tunnel = sample(p, time)
    fade = min(1, max(0, time / 1.2), max(0, (p["duration"] - time) / 2))
    gain = fade * min(1, e * 2.5)
    phase = beat * 0.09
    hue = p["hue"]
    out = []
    strict = bool(p.get("strict_gates", False))
    pose = None
    if "poses" in p:
        pose = [
            np.interp(time, p["times"], np.asarray(p["poses"])[:, j]) for j in range(4)
        ]

    def add(layer, points, color, alpha, width=1.5, dot=False):
        if strict and layer == 4:
            alpha *= high
        if pose is not None:
            x, y, angle, size = pose
            c, s = math.cos(angle), math.sin(angle)
            points = [
                (x + size * (px * c - py * s), y + size * (px * s + py * c))
                for px, py in points
            ]
        if alpha * gain > 0.002:
            out.append(
                (
                    layer,
                    points,
                    colorsys.hsv_to_rgb(color % 1, 0.55, 1),
                    float(alpha * gain),
                    width,
                    dot,
                )
            )

    def noise(i):
        return (math.sin(i * 127.1 + p["seed"] * 31.7) * 43758.5453) % 1

    # Coordinates use half-frame-height as the unit; frame aspect is preserved.
    for i in range(84):
        x = (noise(i + 1) * 2 - 1) * 1.7
        y = (noise(i + 201) * 2 - 1) * 0.92
        y += 0.025 * math.sin(phase + noise(i + 401) * TAU)
        add(
            4,
            [(x, y)],
            hue + 0.08,
            0.10 + 0.28 * high * noise(i + 501),
            0.8 + noise(i + 601) * 1.4,
            True,
        )
    # Perspective rings: sparse in quiet phrases, expanded during dense phrases.
    for i in range(10):
        z = (i / 10 + beat * 0.018) % 1
        radius = 0.12 + z * z * 1.38
        angle = phase * 0.28 + z * 0.8
        pts = []
        for j in range(97):
            theta = j / 96 * TAU
            r = (
                radius
                * (0.85 + 0.15 * abs(math.sin(theta * 2)))
                * (1 + 0.035 * mid * math.sin(theta * 6 + phase))
            )
            pts.append(
                (r * math.cos(theta + angle), r * 0.64 * math.sin(theta + angle))
            )
        add(
            2,
            pts,
            hue + 0.16 * z,
            ((0 if strict else 0.08) + 0.66 * tunnel) * (1 - z) * (0.45 + 0.45 * bass),
            1.1,
        )
    # Luminous closed contours, with harmonic lobes responding to mids/highs.
    for k in range(5):
        pts = []
        for j in range(193):
            theta = j / 192 * TAU
            r = (0.40 + k * 0.038) * (1 + 0.28 * pulse)
            r += (0.018 + 0.055 * mid) * math.sin(
                theta * (3 + k % 2) + phase * (1 if k % 2 else -1) + k * 0.35
            )
            r += 0.018 * high * math.sin(theta * 15 - phase * 3)
            pts.append(
                (r * math.cos(theta + phase * 0.22), r * math.sin(theta + phase * 0.22))
            )
        add(
            0,
            pts,
            hue + k * 0.027,
            ((0 if strict else 0.08) + 1.0 * orbit) * (1 - k * 0.08),
            2.0 + (k == 0) * 1.0,
        )
    # Bilateral wave ribbons: distinct from radial contours; expressive at low energy.
    for k in range(6):
        pts = []
        for j in range(161):
            x = (j / 160 * 2 - 1) * 1.52
            env = max(0, 1 - (x / 1.55) ** 2)
            y = math.sin(x * 3.3 + phase + k * 0.17) * (0.12 + 0.26 * mid) * env
            y += math.sin(x * 8 - phase * 1.3 + k * 0.25) * (0.018 + 0.055 * high) * env
            y += (k - 2.5) * 0.046
            pts.append((x, y))
        add(
            1,
            pts,
            hue + 0.38 + k * 0.012,
            (0 if strict else 0.04) + 0.96 * ribbons,
            2.0,
        )
    # Onset-timed shock rings and short radial sparks. No free-running fake beats.
    for index, (at, strength) in enumerate(p["hits"]):
        age = time - at
        if age < 0 or age > 0.65:
            continue
        radius = 0.36 + 0.65 * age
        alpha = strength * (1 - age / 0.65) ** 2
        pts = [
            (radius * math.cos(j / 96 * TAU), radius * math.sin(j / 96 * TAU))
            for j in range(97)
        ]
        add(3, pts, hue + 0.02, alpha * 0.8, 2.0)
        for j in range(14):
            theta = j / 14 * TAU + index * 0.31
            r = 0.4 + age * (0.65 + noise(j + index) * 0.7)
            pts = [
                (r * math.cos(theta), r * math.sin(theta)),
                (
                    (r + 0.025 + age * 0.04) * math.cos(theta),
                    (r + 0.025 + age * 0.04) * math.sin(theta),
                ),
            ]
            add(3, pts, hue + 0.40 + 0.18 * noise(j + index + 50), alpha, 2.0)
    return out


def performance_frame(runtime, time, width, height):
    p = runtime.score.metadata["presentation"]
    states = runtime.state(time)
    mathematical = p.get("profile") == "mathematical-v2"
    geometry = primitives
    if p.get("profile") == "choreography-v3":
        from .choreography import primitives as geometry
    output_width, output_height = width, height
    if mathematical:
        from .math_geometry import primitives as geometry

        width, height = width * 2, height * 2
    ink = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(ink)
    scale = height / 2
    for layer, points, rgb, alpha, stroke, dot in geometry(p, time):
        s = states.get(layer)
        if not s:
            continue
        angle = math.radians(s["rotation"])
        c, sn = math.cos(angle), math.sin(angle)
        xy = []
        for x, y in points:
            x, y = x * s["scale_x"], y * s["scale_y"]
            xy.append(
                (
                    width / 2 + s["x"] * width / 2 + (x * c - y * sn) * scale,
                    height / 2 - s["y"] * height / 2 - (x * sn + y * c) * scale,
                )
            )
        color = tuple(
            int(max(0, min(255, v * alpha * s["alpha"] * s[key] * 255)))
            for v, key in zip(rgb, ("red", "green", "blue"))
        )
        if dot:
            x, y = xy[0]
            r = max(0.7, stroke * height / 1080)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
        else:
            draw.line(
                xy,
                fill=color,
                width=max(1, round(stroke * height / 540)),
                joint="curve",
            )
    if mathematical:
        width, height = output_width, output_height
        ink = ink.resize((width, height), Image.Resampling.LANCZOS)
    background = Image.new("RGB", (width, height), (3, 6, 15))
    glow = ink.filter(
        ImageFilter.GaussianBlur(max(1, height * (0.004 if mathematical else 0.008)))
    )
    halo = (
        ink.resize((max(2, width // 4), max(2, height // 4)))
        .filter(ImageFilter.GaussianBlur(5))
        .resize((width, height))
    )
    return ImageChops.add(
        ImageChops.add(background, ink),
        ImageChops.add(
            ImageEnhance.Brightness(glow).enhance(2.1 if mathematical else 3.5),
            ImageEnhance.Brightness(halo).enhance(0.65 if mathematical else 3),
        ),
    )
