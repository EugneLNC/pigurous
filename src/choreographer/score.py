"""Versioned, renderer-independent scene score. Coordinates are screen normalized.

All exported motion is world-space; parent_id records provenance/grouping, not an
implicit second transform. x/y use half-width/half-height; rotation is unwrapped
degrees (CCW). A property holds its last endpoint. Later-starting events override
earlier events on the same property; equal starts use stable list order.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from bisect import bisect_right
from pathlib import Path
import json
import math

PROPERTIES = (
    "x",
    "y",
    "rotation",
    "scale_x",
    "scale_y",
    "alpha",
    "red",
    "green",
    "blue",
)
KINDS = ("line", "tap", "hold", "flick", "drag", "circle")
DEFAULTS = dict(
    x=0.0,
    y=0.0,
    rotation=0.0,
    scale_x=1.0,
    scale_y=1.0,
    alpha=1.0,
    red=1.0,
    green=1.0,
    blue=1.0,
)


@dataclass
class TempoPoint:
    beat: float
    time: float
    bpm: float


class TempoMap:
    def __init__(self, points: list[dict] | list[TempoPoint]):
        self.points = [
            p if isinstance(p, TempoPoint) else TempoPoint(**p) for p in points
        ]
        if not self.points:
            raise ValueError("tempo_map must not be empty")
        for i, p in enumerate(self.points):
            if not all(math.isfinite(x) for x in (p.beat, p.time, p.bpm)) or p.bpm <= 0:
                raise ValueError("tempo_map has invalid values")
            if i:
                prev = self.points[i - 1]
                expected = prev.time + (p.beat - prev.beat) * 60 / prev.bpm
                if p.beat <= prev.beat or abs(p.time - expected) > 1e-5:
                    raise ValueError(
                        "tempo_map must be strictly increasing and continuous"
                    )
        self.beats = [p.beat for p in self.points]
        self.times = [p.time for p in self.points]

    @classmethod
    def from_bpm(cls, entries, offset_seconds=0.0):
        entries = sorted(entries, key=lambda p: p[0])
        if not entries or entries[0][0] != 0:
            raise ValueError("BPM map must begin at beat zero")
        points = []
        time = float(offset_seconds)
        for i, (beat, bpm) in enumerate(entries):
            if i:
                time += (beat - entries[i - 1][0]) * 60 / entries[i - 1][1]
            points.append(TempoPoint(float(beat), time, float(bpm)))
        return cls(points)

    @classmethod
    def from_beats(cls, seconds):
        if len(seconds) < 2:
            raise ValueError(
                "At least two detected beats required; supply --bpm if detection fails"
            )
        points = []
        for i, t in enumerate(seconds):
            dt = seconds[i + 1] - t if i + 1 < len(seconds) else t - seconds[i - 1]
            if dt <= 0:
                raise ValueError("Beat timestamps must be strictly increasing")
            points.append(TempoPoint(float(i), float(t), 60 / dt))
        return cls(points)

    def beat_to_time(self, beat):
        p = self.points[max(0, bisect_right(self.beats, beat) - 1)]
        return p.time + (beat - p.beat) * 60 / p.bpm

    def time_to_beat(self, time):
        p = self.points[max(0, bisect_right(self.times, time) - 1)]
        return p.beat + (time - p.time) * p.bpm / 60

    def to_list(self):
        return [asdict(p) for p in self.points]


@dataclass
class Entity:
    id: int
    kind: str = "line"
    parent_id: int | None = None
    width: float = 2.0
    height: float = 0.006
    initial: dict = field(default_factory=lambda: DEFAULTS.copy())
    source: dict = field(default_factory=dict)


@dataclass
class Event:
    beat: float
    op: str
    entity: int
    duration: float = 0.0
    property: str = "x"
    start: float = 0.0
    end: float = 0.0


@dataclass
class SceneScore:
    duration: float
    tempo_map: list[dict]
    entities: list[Entity] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    segments: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    version: str = "1.0"
    coordinate_space: str = "world_normalized_2d"

    def validate(self):
        if self.version != "1.0" or self.coordinate_space != "world_normalized_2d":
            raise ValueError("Unsupported score version or coordinate space")
        if not math.isfinite(self.duration) or self.duration <= 0:
            raise ValueError("duration must be positive and finite")
        tempo = TempoMap(self.tempo_map)
        entities = {e.id: e for e in self.entities}
        if len(entities) != len(self.entities):
            raise ValueError("Duplicate entity ID")
        for e in self.entities:
            if e.id < 0 or e.kind not in KINDS or e.width <= 0 or e.height <= 0:
                raise ValueError("Invalid entity")
            if not all(
                math.isfinite(v) for v in [e.width, e.height, *e.initial.values()]
            ):
                raise ValueError("Non-finite entity values")
            if set(e.initial) - set(PROPERTIES):
                raise ValueError("Unknown initial property")
            seen = {e.id}
            parent = e.parent_id
            while parent is not None:
                if parent not in entities or parent in seen:
                    raise ValueError("Missing parent or parent cycle")
                seen.add(parent)
                parent = entities[parent].parent_id
        previous = -math.inf
        alive = set()
        created = set()
        for ev in self.events:
            if ev.entity not in entities or ev.op not in {
                "create",
                "transform",
                "pulse",
                "destroy",
            }:
                raise ValueError("Invalid event reference or operation")
            if ev.property not in PROPERTIES or not all(
                math.isfinite(v) for v in [ev.beat, ev.duration, ev.start, ev.end]
            ):
                raise ValueError("Invalid event fields")
            if ev.beat < previous - 1e-9 or ev.duration < 0:
                raise ValueError("Event timeline is unordered or duration is negative")
            if (
                tempo.beat_to_time(ev.beat) < -1e-5
                or tempo.beat_to_time(ev.beat + ev.duration) > self.duration + 1e-4
            ):
                raise ValueError("Event lies outside the audio duration")
            if ev.op == "create":
                if ev.entity in created:
                    raise ValueError("Entity IDs cannot be recreated; use a fresh ID")
                alive.add(ev.entity)
                created.add(ev.entity)
            elif ev.entity not in alive:
                raise ValueError("Event references an entity that is not alive")
            elif ev.op == "destroy":
                alive.remove(ev.entity)
            previous = ev.beat
        if self.metadata.get("render_profile") == "choreography-v3":
            from .choreography import validate_program
            program = validate_program(self.metadata["presentation"], self.duration)
            if program["duration"] != self.duration or program["tempo_map"] != self.tempo_map or {g["id"] for g in program["groups"]} != set(entities):
                raise ValueError("Action program/SceneScore mismatch")
        return self

    def save(self, path):
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw["entities"] = [Entity(**e) for e in raw.get("entities", [])]
        raw["events"] = [Event(**e) for e in raw.get("events", [])]
        return cls(**raw).validate()


class ScoreRuntime:
    """Stateless sampling supports seeks, shuffled frame order and offline rendering."""

    def __init__(self, score: SceneScore):
        self.score = score.validate()
        self.tempo = TempoMap(score.tempo_map)
        self.tracks = {}
        self.life = {}
        self.pulses = {}
        for ev in score.events:
            if ev.op == "transform":
                self.tracks.setdefault((ev.entity, ev.property), []).append(ev)
            elif ev.op == "pulse":
                self.pulses.setdefault(ev.entity, []).append(ev)
            else:
                self.life.setdefault(ev.entity, []).append(ev)
        self.starts = {key: [ev.beat for ev in es] for key, es in self.tracks.items()}

    def state(self, time: float):
        beat = self.tempo.time_to_beat(time)
        result = {}
        for ent in self.score.entities:
            life = self.life.get(ent.id, [])
            index = bisect_right([e.beat for e in life], beat) - 1
            if index < 0 or life[index].op != "create":
                continue
            values = DEFAULTS | ent.initial
            for prop in PROPERTIES:
                key = (ent.id, prop)
                es = self.tracks.get(key)
                if es:
                    i = bisect_right(self.starts[key], beat) - 1
                    if i >= 0:
                        e = es[i]
                        u = (
                            min(1.0, max(0.0, (beat - e.beat) / e.duration))
                            if e.duration
                            else 1.0
                        )
                        values[prop] = e.start + (e.end - e.start) * u
            values["pulse"] = max(
                (
                    1 - (beat - e.beat) / max(e.duration, 1e-6)
                    for e in self.pulses.get(ent.id, [])
                    if e.beat <= beat < e.beat + e.duration
                ),
                default=0.0,
            )
            result[ent.id] = values
        return result
