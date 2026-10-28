"""Closed parametric actions, ordered composition and seekable mathematical geometry."""
import colorsys
import math
from bisect import bisect_right

TAU = math.tau


def curve(family, travel, expansion, spin, phase, pulse=0, points=721):
    out = []
    for j in range(points):
        u = j / (points - 1)
        t = u * TAU
        if family == 0:
            morph = 0.72 + 0.20 * travel
            radius = 0.78 * ((1 - morph) + morph * math.cos(5 * t))
            radius *= 1 + 0.025 * pulse
            x, y = radius * math.cos(t), radius * math.sin(t)
        elif family == 1:
            # Avoid the 3:2 phase degeneracy near pi/4, where the path retraces itself.
            delta = 0.08 + 0.14 * math.sin(phase * 0.24) + 0.12 * travel
            x = 0.70 * math.sin(3 * t + delta)
            y = (0.45 + 0.025 * spin) * math.sin(2 * t)
        elif family == 2:
            t *= 2  # R/r=5/2: both frequencies close after 4 pi.
            d = 2.35 + 0.55 * expansion + 0.18 * math.sin(phase * 0.19)
            norm = 0.78 / (3 + d)
            x = norm * (3 * math.cos(t) + d * math.cos(1.5 * t))
            y = norm * (3 * math.sin(t) - d * math.sin(1.5 * t))
        else:
            raise ValueError("Unknown mathematical action")
        out.append((x, y))
    out[-1] = out[0]
    return out


def sample(p, time):
    i = max(0, min(len(p["times"]) - 2, bisect_right(p["times"], time) - 1))
    a, b = p["times"][i : i + 2]
    u = max(0, min(1, (time - a) / max(1e-8, b - a)))
    controls = [
        x * (1 - u) + y * u for x, y in zip(p["controls"][i], p["controls"][i + 1])
    ]
    pose = [x * (1 - u) + y * u for x, y in zip(p["poses"][i], p["poses"][i + 1])]
    controls[4] = max(
        (
            s * math.exp(-(time - at) / 0.13)
            for at, s in p["hits"]
            if 0 <= time - at < 0.7
        ),
        default=0,
    )
    return controls, pose


def primitives(p, time):
    (visibility, expansion, travel, spin, pulse, clock, *weights), pose = sample(
        p, time
    )
    fade = min(1, max(0, time / 1.8), max(0, (p["duration"] - time) / 2))
    gain = fade * min(1, visibility * 2)
    phase = clock * 0.11
    emphasized = [max(0, w) for w in weights]
    denom = max(1e-9, sum(emphasized))
    mixes = [v / denom for v in emphasized]
    x, y, angle, size = pose
    c, s = math.cos(angle), math.sin(angle)
    out = []
    hue = p["hue"]

    def add(layer, pts, color, alpha, width=1.4, dot=False, posed=True):
        if alpha * gain < 0.004:
            return
        if posed:
            pts = [
                (x + size * (px * c - py * s), y + size * (px * s + py * c))
                for px, py in pts
            ]
        out.append(
            (
                layer,
                pts,
                colorsys.hsv_to_rgb(color % 1, 0.42, 1),
                alpha * gain,
                width,
                dot,
            )
        )

    # Golden-angle phyllotaxis produces an ordered, sparse frame around the motif.
    for j in range(72):
        theta = j * 2.399963229728653 + phase * 0.025
        radius = 0.86 + 0.12 * math.sqrt((j + 0.5) / 72)
        add(
            4,
            [(radius * math.cos(theta), radius * math.sin(theta))],
            hue + 0.04,
            (0.025 + 0.04 * spin) * sum(weights),
            0.8,
            True,
            False,
        )
    if sum(emphasized) > 1e-8:
        bases = [
            curve(family, travel, expansion, spin, phase, pulse) for family in range(3)
        ]
        # Blend closed bases into ONE motif instead of crossing three unrelated layers.
        pts = [
            (
                sum(mixes[f] * bases[f][j][0] for f in range(3)),
                sum(mixes[f] * bases[f][j][1] for f in range(3)),
            )
            for j in range(721)
        ]
        # Keep the harmonic mixture legible when its bases partially cancel.
        norm = (
            0.76
            * (1 + 0.025 * pulse)
            / max(0.1, max(math.hypot(px, py) for px, py in pts))
        )
        pts = [(px * norm, py * norm) for px, py in pts]
        for echo in (1, 0):
            scaled = [
                (px * (1 + echo * 0.045), py * (1 + echo * 0.045)) for px, py in pts
            ]
            for segment in range(24):
                start = segment * 30
                part = scaled[start : start + 31]
                color = hue + 0.045 * math.sin(segment / 24 * TAU + phase * 0.2)
                add(
                    1 if echo else 0,
                    part,
                    color,
                    0.10 if echo else 0.92,
                    1.0 if echo else 1.7,
                )
        index = int(((phase * 0.10) % 1) * 720)
        add(0, [pts[index]], hue + 0.03, 0.85, 2.2, True)
        guide = [
            (0.79 * math.cos(j / 160 * TAU), 0.79 * math.sin(j / 160 * TAU))
            for j in range(161)
        ]
        add(2, guide, hue + 0.06, 0.055 * mixes[2], 0.9)
    # Accents are symmetrical and short-lived; retain the exact model hit times.
    active = [(at, strength) for at, strength in p["hits"] if 0 <= time - at < 0.36][
        -2:
    ]
    for at, strength in active:
        age = time - at
        alpha = strength * (1 - age / 0.36) ** 2
        radius = 0.80 + 0.11 * age / 0.36
        for j in range(12):
            theta = j / 12 * TAU + phase * 0.03
            add(
                3,
                [(radius * math.cos(theta), radius * math.sin(theta))],
                hue + 0.50,
                alpha * 0.6,
                1.4,
                True,
            )
    return out
