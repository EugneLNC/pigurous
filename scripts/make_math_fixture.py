"""Python/TypeScript parity for the mathematical profile and inertial pose track."""
import json
from pathlib import Path
from choreographer.math_presentation import compile_math
from choreographer.math_geometry import primitives, sample

p = compile_math(
    dict(
        profile="performance-v1",
        duration=8,
        seed=42,
        hue=0.54,
        times=[0, 2, 4, 6, 8],
        controls=[
            [0.7, 0.3, 0.2, 0.1, 0, i * 3, *w]
            for i, w in enumerate(
                ([1, 0, 0], [0.5, 0.5, 0], [0, 1, 0], [0, 0.4, 0.6], [0, 0, 1])
            )
        ],
        poses=[[0.04, -0.02, a, 1] for a in [0, 2, -2, 1, 0]],
        hits=[[1, 0.9], [3, 0.7]],
    )
)
frames = []
for time in [0.999, 1.04, 3.2, 6.0]:
    values = primitives(p, time)
    compact = [
        dict(
            layer=l,
            count=len(pts),
            rgb=rgb,
            alpha=alpha,
            stroke=stroke,
            dot=dot,
            points=[pts[0], pts[len(pts) // 2], pts[-1]],
        )
        for l, pts, rgb, alpha, stroke, dot in values
    ]
    controls, pose = sample(p, time)
    frames.append(dict(time=time, sample=controls, pose=pose, primitives=compact))
Path("tests/fixtures/math_parity.json").write_text(
    json.dumps(dict(presentation=p, frames=frames)), encoding="utf-8"
)
