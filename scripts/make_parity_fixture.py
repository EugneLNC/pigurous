import json
from dataclasses import asdict
from pathlib import Path
from choreographer.score import *

s = SceneScore(
    12,
    TempoMap.from_bpm([(0, 120), (4, 90)], 0.2).to_list(),
    [Entity(0)],
    [
        Event(0, "create", 0),
        Event(0, "transform", 0, 8, "rotation", 0, 720),
        Event(1, "transform", 0, 3, "x", -0.5, 0.8),
        Event(2, "transform", 0, 2, "x", 0.1, 0.3),
        Event(10, "destroy", 0),
    ],
)
rt = ScoreRuntime(s)
Path("tests/fixtures").mkdir(exist_ok=True)
Path("tests/fixtures/runtime_parity.json").write_text(
    json.dumps(
        {
            "score": asdict(s),
            "frames": [
                {"time": t, "states": rt.state(t)} for t in [5, 0, 1, 2.2, 8, 3.3]
            ],
        }
    ),
    encoding="utf-8",
)
