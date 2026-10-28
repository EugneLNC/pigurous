"""Write portable Python/TypeScript geometry parity evidence."""
import json
from pathlib import Path
from choreographer.performance_render import primitives, sample

p = dict(
    profile="performance-v1",
    duration=8,
    seed=42,
    hue=0.52,
    times=[0, 4, 8],
    controls=[
        [0.4, 0.5, 0.2, 0.1, 0, 0, 0.5, 0.3, 0.2],
        [0.8, 0.6, 0.8, 0.5, 0, 8, 0.2, 0.3, 0.5],
        [0.2, 0.1, 0.3, 0.2, 0, 16, 0.3, 0.6, 0.1],
    ],
    hits=[[1, 0.9], [3, 0.7]],
)
frames = []
for time in [0.999, 1.04, 3.2, 6.0]:
    values = primitives(p, time)
    compact = []
    for layer, points, rgb, alpha, stroke, dot in values:
        compact.append(
            dict(
                layer=layer,
                count=len(points),
                rgb=rgb,
                alpha=alpha,
                stroke=stroke,
                dot=dot,
                points=[points[0], points[len(points) // 2], points[-1]],
            )
        )
    frames.append(dict(time=time, sample=sample(p, time).tolist(), primitives=compact))
Path("tests/fixtures/presentation_parity.json").write_text(
    json.dumps(dict(presentation=p, frames=frames)), encoding="utf-8"
)
