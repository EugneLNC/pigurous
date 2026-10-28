"""Hand-authored deterministic renderer fixture. This is NOT model output."""
from pathlib import Path
import math
from choreographer.score import SceneScore, Entity, Event, TempoMap


def make():
    s = SceneScore(
        24,
        TempoMap.from_bpm([(0, 120)]).to_list(),
        metadata={
            "title": "Orbit / 几何主题练习",
            "origin": "hand_authored_demo",
            "training_eligible": False,
        },
    )
    for i in range(12):
        angle = i * 30
        rad = math.radians(angle)
        s.entities.append(
            Entity(
                i,
                "line",
                width=0.42,
                height=0.007,
                initial={
                    "x": 0.48 * math.cos(rad),
                    "y": 0.7 * math.sin(rad),
                    "rotation": angle + 90,
                    "red": 0.52,
                    "green": 0.89,
                    "blue": 0.77,
                    "alpha": 0.85,
                },
            )
        )
        s.events.append(Event(0, "create", i))
        for k in range(8):
            a = k * 6
            theta = math.radians(angle + k * 45)
            next_theta = math.radians(angle + (k + 1) * 45)
            radius = 0.28 if k in (2, 6) else 0.7 if k in (3, 7) else 0.48
            old = (
                0.48
                if k == 0
                else (0.28 if k - 1 in (2, 6) else 0.7 if k - 1 in (3, 7) else 0.48)
            )
            s.events.extend(
                [
                    Event(
                        a,
                        "transform",
                        i,
                        6,
                        "x",
                        old * math.cos(theta),
                        radius * math.cos(next_theta),
                    ),
                    Event(
                        a,
                        "transform",
                        i,
                        6,
                        "y",
                        old * 1.4 * math.sin(theta),
                        radius * 1.4 * math.sin(next_theta),
                    ),
                    Event(
                        a,
                        "transform",
                        i,
                        6,
                        "rotation",
                        angle + k * 45 + 90,
                        angle + (k + 1) * 45 + 90,
                    ),
                ]
            )
    s.events.sort(key=lambda e: (e.beat, 0 if e.op == "create" else 1))
    return s


if __name__ == "__main__":
    make().save("preview/public/demo.json")
    make().save("tests/fixtures/demo.json")
