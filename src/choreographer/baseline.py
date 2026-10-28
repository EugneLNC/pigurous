"""Fixed audio mapping comparator, explicitly not learned choreography."""
import numpy as np
from .audio import Features
from .score import SceneScore, Entity, Event
from .curves import simplify_curve


def fixed_mapping(features, output):
    f = Features(features)
    duration = f.metadata["duration"]
    tempo = f.tempo
    s = SceneScore(
        duration,
        tempo.to_list(),
        metadata={
            "origin": "fixed_mapping_baseline",
            "audio": f.metadata,
            "song_id": f.metadata["audio_sha256"],
        },
    )
    begin = tempo.time_to_beat(0)
    times = f.transient_times
    valid = times < duration
    times = times[valid]
    for i in range(12):
        s.entities.append(
            Entity(
                i,
                "line",
                width=0.08,
                height=0.8,
                initial={
                    "x": (i - 5.5) * 0.14,
                    "y": 0,
                    "red": 0.5,
                    "green": 0.8,
                    "blue": 1,
                },
            )
        )
        s.events.append(Event(begin, "create", i))
        bands = f.transient[valid, i * 10 : (i + 1) * 10].mean(1)
        values = 0.05 + 0.95 * bands / (np.quantile(bands, 0.95) + 1e-5)
        values = values.clip(0.05, 1.5)
        points = simplify_curve(
            [(tempo.time_to_beat(float(t)), float(v)) for t, v in zip(times, values)],
            1 / (0.8 * 540),
        )
        for (a, x), (b, y) in zip(points, points[1:]):
            s.events.append(Event(a, "transform", i, b - a, "scale_y", x, y))
    s.events.sort(key=lambda e: (e.beat, e.op != "create"))
    s.save(output)
    return s
