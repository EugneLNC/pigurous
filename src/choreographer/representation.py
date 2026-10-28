"""Compound events with exact coarse+residual reconstruction and absolute anchors."""
from dataclasses import replace
from bisect import bisect_right
import numpy as np
from .score import PROPERTIES, KINDS, DEFAULTS, ScoreRuntime, TempoMap

OPS = ("pad", "bos", "eos", "create", "transform", "pulse", "destroy")
BIN_SIZES = (3073, 3073, 257, 257, 257, 257)
LOW = np.array([0, 0, -8, -8, 0, 0], dtype=np.float32)
STEP = np.array([1 / 192, 1 / 192, 1 / 16, 1 / 16, 1 / 32, 1 / 32], dtype=np.float32)
STATE_DIM = 31


def quantize(values):
    raw = (np.asarray(values) - LOW) / STEP
    bins = np.clip(np.rint(raw), 0, np.array(BIN_SIZES) - 1).astype(np.int64)
    return bins, (raw - bins).astype(np.float32)


def dequantize(bins, residual):
    return LOW + (np.asarray(bins) + np.asarray(residual)) * STEP


def event_values(ev, previous, ent):
    scale = 360 if ev.property == "rotation" else 1
    return [
        ev.beat - previous,
        ev.duration,
        ev.start / scale,
        ev.end / scale,
        ent.width,
        ent.height,
    ]


def descriptor(score, a, b):
    tempo = TempoMap(score.tempo_map)
    rt = ScoreRuntime(score)
    states = [
        rt.state(tempo.beat_to_time(t)) for t in np.linspace(a, b, 5, endpoint=False)
    ]
    visible = [
        s
        for frame in states
        for s in frame.values()
        if s["alpha"] > 0.01 and abs(s["x"]) < 2 and abs(s["y"]) < 2
    ]
    events = [e for e in score.events if a <= e.beat < b]
    if not visible:
        return np.zeros(8, np.float32)
    xy = np.array([[s["x"], s["y"]] for s in visible])
    motion = [
        abs(e.end - e.start)
        / max(e.duration, 0.01)
        / (360 if e.property == "rotation" else 1)
        for e in events
        if e.op == "transform"
    ]
    return np.array(
        [
            np.log1p(len(visible) / 5),
            *xy.mean(0),
            *xy.std(0),
            np.log1p(np.mean(motion) if motion else 0),
            np.mean([s["alpha"] for s in visible]),
            np.log1p(len(events) / (b - a)),
        ],
        np.float32,
    )


def state_memory(score, anchor, slots, max_entities, include_boundary=False):
    tempo = TempoMap(score.tempo_map)
    rt = ScoreRuntime(score)
    states = rt.state(
        tempo.beat_to_time(anchor) + (1e-7 if include_boundary else -1e-7)
    )
    memory = np.zeros((max_entities, STATE_DIM), np.float32)
    entities = {e.id: e for e in score.entities}
    for eid, s in states.items():
        k = slots[eid]
        ent = entities[eid]
        memory[k, :4] = [1, KINDS.index(ent.kind) / 5, ent.width, ent.height]
        for p, prop in enumerate(PROPERTIES):
            v = s[prop]
            remaining = 0
            target = v
            es = [e for e in rt.tracks.get((eid, prop), []) if e.beat < anchor]
            if es and es[-1].beat + es[-1].duration > anchor:
                remaining = es[-1].beat + es[-1].duration - anchor
                target = es[-1].end
            scale = 360 if prop == "rotation" else 1
            memory[k, 4 + 3 * p : 7 + 3 * p] = [
                v / scale,
                remaining / 16,
                target / scale,
            ]
    return memory


def blocks(score, max_events=2048, max_entities=128):
    if len(score.entities) > max_entities:
        raise ValueError(
            f"{len(score.entities)} entities exceeds configured {max_entities}; use an explicitly recorded lines-only projection or a larger vocabulary"
        )
    slots = {e.id: i for i, e in enumerate(score.entities)}
    entities = {e.id: e for e in score.entities}
    tempo = TempoMap(score.tempo_map)
    begin = tempo.time_to_beat(0)
    end = tempo.time_to_beat(score.duration)
    # Materialize initial state fields, so inference never receives target entities.
    source = []
    for e in score.events:
        source.append(e)
        if e.op == "create":
            for p, v in entities[e.entity].initial.items():
                if v != DEFAULTS[p]:
                    source.append(
                        replace(e, op="transform", property=p, start=v, end=v)
                    )
    source.sort(key=lambda e: e.beat)
    tracks = {}
    for e in source:
        if e.op == "transform":
            tracks.setdefault((e.entity, e.property), []).append(e)
    starts = {key: [e.beat for e in values] for key, values in tracks.items()}

    def split(a, b, depth=0):
        memory = state_memory(score, a, slots, max_entities)
        local = []
        for e in source:
            if a <= e.beat < b or (e.beat == end == b):
                if e.op == "transform" and e.beat + e.duration > b:
                    u = (b - e.beat) / e.duration
                    local.append(
                        replace(
                            e, duration=b - e.beat, end=e.start + (e.end - e.start) * u
                        )
                    )
                else:
                    local.append(e)
            elif e.op == "transform" and e.beat < a < e.beat + e.duration:
                if not memory[slots[e.entity], 0]:
                    continue
                # Carry only the winning track at the anchor, not overridden curves.
                key = (e.entity, e.property)
                last = tracks[key][bisect_right(starts[key], a) - 1]
                if last is not e:
                    continue
                stop = min(b, e.beat + e.duration)
                u = (a - e.beat) / e.duration
                v = (stop - e.beat) / e.duration
                local.append(
                    replace(
                        e,
                        beat=a,
                        duration=stop - a,
                        start=e.start + (e.end - e.start) * u,
                        end=e.start + (e.end - e.start) * v,
                    )
                )
        local.sort(key=lambda e: e.beat)
        if len(local) + 1 > max_events:
            if depth >= 16 or b - a < 1 / 192:
                raise ValueError(
                    "Simultaneous event density exceeds budget; sample quarantined"
                )
            yield from split(a, (a + b) / 2, depth + 1)
            yield from split((a + b) / 2, b, depth + 1)
            return
        cats = [[1, 0, 0, 0]]
        nums = [[0.0] * 6]
        previous = a
        for e in local:
            cats.append(
                [
                    OPS.index(e.op),
                    slots[e.entity],
                    PROPERTIES.index(e.property),
                    KINDS.index(entities[e.entity].kind),
                ]
            )
            nums.append(event_values(e, previous, entities[e.entity]))
            previous = e.beat
        cats.append([2, 0, 0, 0])
        nums.append([0.0] * 6)
        bins, residual = quantize(nums)
        yield dict(
            anchor=a,
            end=b,
            cats=np.asarray(cats, np.int64),
            bins=bins,
            residual=residual,
            state=memory,
        )

    for a in np.arange(begin, end, 16):
        yield from split(float(a), min(end, float(a) + 16))


def grouped_split(records, seed=42):
    ids = sorted({r["song_id"] for r in records})
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    nval = max(1, int(round(n * 0.1))) if n >= 3 else 0
    ntest = nval
    mapping = {
        sid: ("test" if i < ntest else "val" if i < ntest + nval else "train")
        for i, sid in enumerate(ids)
    }
    return [dict(r, split=mapping[r["song_id"]]) for r in records]
