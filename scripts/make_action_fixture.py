"""Cross-language parity with learned poses and strict layer gates."""
import json
from pathlib import Path
import numpy as np
from choreographer.actions import execute_motion
from choreographer.performance_render import primitives, sample

motion = np.random.default_rng(42).random((128, 12))
accent = np.zeros(128)
accent[16] = 0.9
accent[48] = 0.7
p = execute_motion(motion, accent, np.arange(128) % 64, 8)
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
    frames.append(dict(time=time, sample=sample(p, time).tolist(), primitives=compact))
Path("tests/fixtures/action_parity.json").write_text(
    json.dumps(dict(presentation=p, frames=frames)), encoding="utf-8"
)
