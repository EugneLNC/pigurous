"""Native chart evaluator and conservative conversion to world-space SceneScore.

External textures, shaders and nonlinear note controls are quarantined, never
silently used as clean supervision. Visual skins are deliberately geometric.
"""
from __future__ import annotations
from bisect import bisect_right
from functools import lru_cache
from pathlib import Path
import copy
import hashlib
import json
import math
import numpy as np
from .score import (
    SceneScore,
    Entity,
    Event,
    TempoMap,
    PROPERTIES,
    DEFAULTS,
    ScoreRuntime,
)
from .curves import event_value, adaptive_curve, simplify_curve


class UnsupportedChart(ValueError):
    pass


def beat(v):
    if isinstance(v, list):
        if len(v) != 3 or (v[2] == 0 and v[1] != 0):
            raise ValueError("Invalid fractional beat")
        return v[0] + (v[1] / v[2] if v[2] else 0)
    return float(v)


class Track:
    def __init__(self, events, tempo, default=0.0):
        self.events = sorted(events or [], key=lambda e: e["startTime"])
        self.starts = [e["startTime"] for e in self.events]
        self.tempo = tempo
        self.default = default
        for e in self.events:
            if e["endTime"] < e["startTime"]:
                raise ValueError("Event end precedes start")
        self.knots = sorted(
            {0.0, *self.starts, *[e["endTime"] for e in self.events], *tempo.beats}
        )
        self.prefix = None

    def _area(self, a, b):
        return (
            float(self.value(a + 1e-10) + self.value(b - 1e-10))
            * 0.5
            * (self.tempo.beat_to_time(b) - self.tempo.beat_to_time(a))
        )

    def value(self, b):
        if not self.events:
            return self.default
        i = bisect_right(self.starts, b) - 1
        if i < 0:
            return self.events[0]["start"]
        return event_value(self.events[i], b, self.tempo.beat_to_time)

    def integral(self, a, b):
        if self.prefix is None:
            self.prefix = [0.0]
            for l, r in zip(self.knots, self.knots[1:]):
                self.prefix.append(self.prefix[-1] + self._area(l, r))

        def primitive(x):
            i = max(0, bisect_right(self.knots, x) - 1)
            return self.prefix[i] + self._area(self.knots[i], x)

        return primitive(b) - primitive(a)


def official_to_rpe(raw):
    if raw.get("formatVersion") != 3:
        raise UnsupportedChart("Only official formatVersion=3 is currently supported")
    lines = raw.get("judgeLineList", [])
    if not lines:
        raise ValueError("No judgment lines")
    bpm = float(lines[0]["bpm"])
    result = {
        "META": {"offset": raw.get("offset", 0) * 1000},
        "BPMList": [{"startTime": 0.0, "bpm": bpm}],
        "judgeLineList": [],
    }
    for line in lines:
        ratio = bpm / float(line["bpm"]) / 32

        def evs(key, start, end, scale=1.0, shift=0.0):
            return [
                {
                    "startTime": e["startTime"] * ratio,
                    "endTime": e["endTime"] * ratio,
                    "start": e[start] * scale + shift,
                    "end": e[end] * scale + shift,
                    "easingType": 1,
                }
                for e in line.get(key, [])
            ]

        layer = {
            "moveXEvents": evs("judgeLineMoveEvents", "start", "end", 1350, -675),
            "moveYEvents": evs("judgeLineMoveEvents", "start2", "end2", 900, -450),
            "rotateEvents": evs("judgeLineRotateEvents", "start", "end", -1),
            "alphaEvents": evs("judgeLineDisappearEvents", "start", "end", 255),
            "speedEvents": evs("speedEvents", "value", "value", 4.5),
        }
        out = {"eventLayers": [layer], "father": -1, "notes": []}
        for above, key in [(1, "notesAbove"), (2, "notesBelow")]:
            for n in line.get(key, []):
                start, end = (
                    n["time"] * ratio,
                    (n["time"] + n.get("holdTime", 0)) * ratio,
                )
                kind = {1: 1, 2: 4, 3: 2, 4: 3}.get(n["type"])
                if kind is None:
                    raise UnsupportedChart("Unknown official note type")
                speed = float(n.get("speed", 1))
                if kind == 2:
                    at_hit = next(
                        (
                            e["value"]
                            for e in reversed(line.get("speedEvents", []))
                            if e["startTime"] <= n["time"]
                        ),
                        1.0,
                    )
                    speed = speed / at_hit if at_hit else 0.0
                    changes = [
                        e
                        for e in line.get("speedEvents", [])
                        if n["time"] < e["startTime"] < n["time"] + n.get("holdTime", 0)
                        and e["value"] != at_hit
                    ]
                    if changes:
                        raise UnsupportedChart(
                            "Official hold spans a speed change; requires native hold-tail parity"
                        )
                out["notes"].append(
                    dict(
                        type=kind,
                        startTime=start,
                        endTime=end,
                        positionX=n["positionX"] * 75,
                        above=above,
                        speed=speed,
                        size=1.0,
                        alpha=255,
                        visibleTime=999999.0,
                    )
                )
        result["judgeLineList"].append(out)
    return result


class NativeChart:
    def __init__(self, raw, lines_only=False):
        self.lines_only = lines_only
        self.raw_format = "pgr3" if "formatVersion" in raw else "rpe"
        self.raw = (
            official_to_rpe(raw) if self.raw_format == "pgr3" else copy.deepcopy(raw)
        )
        meta = self.raw.get("META", self.raw)
        self.tempo = TempoMap.from_bpm(
            [(beat(e["startTime"]), e["bpm"]) for e in self.raw["BPMList"]],
            meta.get("offset", 0) / 1000,
        )
        self.lines = self.raw["judgeLineList"]
        self.tracks = []
        self.boundaries = set([self.tempo.time_to_beat(0.0)])
        self.notes = []
        self.ids = []
        self._check()
        for i, line in enumerate(self.lines):
            tracks = {}
            for prop, key in [
                ("x", "moveXEvents"),
                ("y", "moveYEvents"),
                ("rotation", "rotateEvents"),
                ("alpha", "alphaEvents"),
                ("speed", "speedEvents"),
            ]:
                tracks[prop] = []
                for layer in line.get("eventLayers", []) or []:
                    events = copy.deepcopy((layer or {}).get(key, []) or [])
                    for e in events:
                        e["startTime"], e["endTime"] = beat(e["startTime"]), beat(
                            e["endTime"]
                        )
                        if prop == "speed":
                            e["easingType"], e["bezier"] = 1, 0
                        self.boundaries.update([e["startTime"], e["endTime"]])
                    tracks[prop].append(Track(events, self.tempo))
            ext = line.get("extended", {}) or {}
            for prop, key, default in [
                ("scale_x", "scaleXEvents", 1.0),
                ("scale_y", "scaleYEvents", 1.0),
                ("color", "colorEvents", [255, 255, 255]),
            ]:
                events = copy.deepcopy(ext.get(key, []))
                for e in events:
                    e["startTime"], e["endTime"] = beat(e["startTime"]), beat(
                        e["endTime"]
                    )
                    self.boundaries.update([e["startTime"], e["endTime"]])
                tracks[prop] = Track(events, self.tempo, default)
            self.tracks.append(tracks)
            for n in line.get("notes", []):
                n["startTime"], n["endTime"] = beat(n["startTime"]), beat(
                    n.get("endTime", n["startTime"])
                )
                self.notes.append((i, n))
                self.boundaries.update([n["startTime"], n["endTime"]])
        self.boundaries.update(p.beat for p in self.tempo.points)

    def _check(self):
        problems = []
        if self.raw.get("videos") or self.raw.get("extra"):
            problems.append("external video/shader effects")
        for i, line in enumerate(self.lines):
            if line.get("Texture", "line.png") not in ("line.png", "", None):
                problems.append(f"line {i}: custom texture")
            if line.get("attachUI"):
                problems.append(f"line {i}: attachUI")
            if line.get("isGif"):
                problems.append(f"line {i}: animated texture")
            if line.get("anchor", [0.5, 0.5]) != [0.5, 0.5]:
                problems.append(f"line {i}: noncentral anchor")
            if line.get("zOrder", 0):
                problems.append(f"line {i}: explicit zOrder")
            if line.get("bpmfactor", 1) != 1:
                problems.append(f"line {i}: bpmfactor")
            if line.get("scaleOnNotes", 0):
                problems.append(f"line {i}: scaleOnNotes")
            for key in (
                "posControl",
                "alphaControl",
                "sizeControl",
                "skewControl",
                "yControl",
            ):
                field = key.removesuffix("Control")
                identity = 0 if field == "skew" else 1
                if line.get(key) and any(
                    node.get(field) != identity for node in line[key]
                ):
                    problems.append(f"line {i}: {key}")
            for key, val in (line.get("extended", {}) or {}).items():
                if (
                    key in ("inclineEvents", "gifEvents")
                    and val
                    and all(e.get("start") == e.get("end") == 0 for e in val)
                ):
                    continue
                if (
                    key == "textEvents"
                    and val
                    and all(e.get("start") == e.get("end") == "" for e in val)
                ):
                    continue
                if val and key not in {"scaleXEvents", "scaleYEvents", "colorEvents"}:
                    problems.append(f"line {i}: {key}")
            seen = {i}
            p = line.get("father", -1)
            while p != -1:
                if p in seen or p < 0 or p >= len(self.lines):
                    raise ValueError("Invalid/cyclic RPE father")
                seen.add(p)
                p = self.lines[p].get("father", -1)
            for layer in line.get("eventLayers", []) or []:
                for key, es in (layer or {}).items():
                    if (
                        key
                        not in {
                            "moveXEvents",
                            "moveYEvents",
                            "rotateEvents",
                            "alphaEvents",
                            "speedEvents",
                        }
                        and es
                    ):
                        problems.append(f"line {i}: unknown channel {key}")
                    if (
                        key == "alphaEvents"
                        and not self.lines_only
                        and any(min(e["start"], e["end"]) < 0 for e in es)
                    ):
                        problems.append(f"line {i}: negative alpha visibility modes")
                    if key == "speedEvents" and any(
                        e.get("bezier") or e.get("easingType", 1) not in (0, 1)
                        for e in es or []
                    ):
                        problems.append(f"line {i}: eased speed")
            if any(n.get("type") not in (1, 2, 3, 4) for n in line.get("notes", [])):
                problems.append(f"line {i}: unknown note type")
        if problems:
            raise UnsupportedChart("; ".join(problems))

    @lru_cache(maxsize=32768)
    def line_state(self, i, b):
        tr = self.tracks[i]
        s = {
            k: float(sum(t.value(b) for t in tr[k]))
            for k in ("x", "y", "rotation", "alpha")
        }
        s["x"] *= 1920 / 1350
        s["y"] *= 1080 / 900
        s["rotation"] *= -1
        s["alpha"] = min(255, max(0, s["alpha"])) / 255
        p = self.lines[i].get("father", -1)
        if p != -1:
            ps = self.line_state(p, b)
            a = math.radians(ps["rotation"])
            s["x"], s["y"] = ps["x"] * 960 + s["x"] * math.cos(a) - s["y"] * math.sin(
                a
            ), ps["y"] * 540 + s["x"] * math.sin(a) + s["y"] * math.cos(a)
            if self.lines[i].get("rotateWithFather", False):
                s["rotation"] += ps["rotation"]
        s["scale_x"], s["scale_y"] = float(tr["scale_x"].value(b)), float(
            tr["scale_y"].value(b)
        )
        for k, v in zip(("red", "green", "blue"), tr["color"].value(b)):
            s[k] = float(v) / 255
        s["x"] /= 960
        s["y"] /= 540
        return s

    @lru_cache(maxsize=65536)
    def floor(self, line, b):
        return sum(t.integral(0.0, b) for t in self.tracks[line]["speed"])

    def note_state(self, idx, b):
        line, n = self.notes[idx]
        ls = self.line_state(line, b)
        hit, end = n["startTime"], n["endTime"]
        speed = n.get("speed", 1.0)
        offset = n.get("yOffset", 0) * speed * 1080 / 900
        head = (self.floor(line, hit) - self.floor(line, b)) * speed * 144 + offset
        tail = (self.floor(line, end) - self.floor(line, b)) * speed * 144 + offset
        hold = n["type"] == 2
        if hold and b >= hit:
            head = offset
        sign = 1 if n.get("above", 1) == 1 else -1
        cover = self.lines[line].get("isCover", 1)
        center = (
            (max(0, head) + max(0, tail)) / 2
            if hold and cover
            else ((head + tail) / 2 if hold else head)
        )
        x = n.get("positionX", 0) * 1920 / 1350
        y = sign * center
        a = math.radians(ls["rotation"])
        visible = self.tempo.beat_to_time(b) >= self.tempo.beat_to_time(hit) - n.get(
            "visibleTime", 999999.0
        )
        visible = visible and b < (end if hold else hit)
        if cover and (tail if hold else head) * speed < 0:
            visible = False
        return dict(
            x=ls["x"] + (x * math.cos(a) - y * math.sin(a)) / 960,
            y=ls["y"] + (x * math.sin(a) + y * math.cos(a)) / 540,
            rotation=ls["rotation"],
            scale_x=n.get("size", 1.0),
            scale_y=max(
                0.001,
                abs((max(0, tail) - max(0, head)) if cover else tail - head) / 1080,
            )
            if hold
            else 1.0,
            alpha=n.get("alpha", 255) / 255 if visible else 0.0,
            red=1.0,
            green=1.0,
            blue=1.0,
        )


def convert_chart(path, duration, song_id=None, pixel_tolerance=1.0, lines_only=False):
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    line_order = {
        i: line.get("zOrder", 0) for i, line in enumerate(raw.get("judgeLineList", []))
    }
    if lines_only:
        for line in raw.get("judgeLineList", []):
            for key in (
                "notes",
                "notesAbove",
                "notesBelow",
                "posControl",
                "alphaControl",
                "sizeControl",
                "skewControl",
                "yControl",
                "scaleOnNotes",
                "zOrder",
            ):
                line.pop(key, None)
            if line.get("extended"):
                line["extended"].pop("inclineEvents", None)
    native = NativeChart(raw, lines_only)
    tempo = native.tempo
    begin, last = tempo.time_to_beat(0), tempo.time_to_beat(duration)
    score = SceneScore(
        duration,
        tempo.to_list(),
        metadata={
            "source_chart": str(Path(path).resolve()),
            "source_format": native.raw_format,
            "song_id": song_id or hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "origin": "converted_chart",
            "training_eligible": True,
            "geometric_skin": True,
            "tolerance_px": pixel_tolerance,
        },
    )
    score.metadata["projection"] = "lines_only" if lines_only else "lines_and_notes"
    # Many baked source boundaries are continuous. Preserve discontinuities while
    # allowing compression across continuous source-event boundaries.
    for idx in range(len(native.lines) + (0 if lines_only else len(native.notes))):
        is_line = idx < len(native.lines)
        if is_line:
            line = native.lines[idx]
            start, end = begin, last
            ent = Entity(
                idx,
                "line",
                line.get("father") if line.get("father", -1) >= 0 else None,
                width=4.0,
                height=0.004,
            )
            fn = lambda b, i=idx: native.line_state(i, b)
        else:
            ni = idx - len(native.lines)
            li, n = native.notes[ni]
            end = min(last, n["endTime"] if n["type"] == 2 else n["startTime"])
            start = max(
                begin,
                tempo.time_to_beat(
                    tempo.beat_to_time(n["startTime"]) - n.get("visibleTime", 999999.0)
                ),
            )
            if end <= start:
                continue
            ent = Entity(
                idx,
                {1: "tap", 2: "hold", 3: "flick", 4: "drag"}[n["type"]],
                li,
                width=0.12,
                height=2.0 if n["type"] == 2 else 0.022,
                source={"hit_beat": n["startTime"], "fake": bool(n.get("isFake", 0))},
            )
            fn = lambda b, i=ni: native.note_state(i, b)
        fn = lru_cache(maxsize=65536)(fn)
        li = idx if is_line else native.notes[idx - len(native.lines)][0]
        related = set(tempo.beats)
        while li != -1:
            for trs in native.tracks[li].values():
                for tr in trs if isinstance(trs, list) else [trs]:
                    related.update(tr.knots)
            li = native.lines[li].get("father", -1)
        if not is_line:
            related.update([n["startTime"], n["endTime"]])
        boundaries = sorted(x for x in related if start < x < end)
        ent.initial = DEFAULTS.copy()
        score.entities.append(ent)
        score.events.append(Event(start, "create", idx))
        if not is_line:
            score.events.append(Event(end, "destroy", idx))
        anchors = [start, *[b for b in boundaries if start < b < end], end]
        for prop in PROPERTIES:
            scale = {
                "x": 960,
                "y": 540,
                "rotation": max(ent.width * 960, ent.height * 540) * math.pi / 180,
                "scale_x": ent.width * 960,
                "scale_y": ent.height * 540,
            }.get(prop, 255.0)
            tol = pixel_tolerance / max(scale, 1.0) / 8
            # Source boundaries are mandatory knots: even a narrow baked spike
            # must be sampled before any compression is attempted.
            accumulated = []

            def emit(points):
                for (t0, v0), (t1, v1) in zip(points, points[1:]):
                    score.events.append(
                        Event(t0, "transform", idx, t1 - t0, prop, v0, v1)
                    )

            for a, b in zip(anchors, anchors[1:]):
                f = lambda t, p=prop, stop=b: fn(min(t, stop - 1e-9))[p]
                points = adaptive_curve(f, a, b, tol, max_step=0.125)
                accumulated.extend(points if not accumulated else points[1:])
                if b == end or abs(fn(b)[prop] - fn(b - 1e-8)[prop]) > tol:
                    emit(simplify_curve(accumulated, tol))
                    accumulated = []
    order = {"create": 0, "transform": 1, "pulse": 2, "destroy": 3}
    score.events.sort(key=lambda e: (e.beat, order[e.op], e.entity))
    if lines_only:
        score.entities.sort(key=lambda e: (line_order[e.id], e.id))
        score.metadata["line_draw_order"] = "source zOrder, stable source index"
    score.validate()
    return score, native


def reconstruction_audit(score, native, frames=240):
    runtime = ScoreRuntime(score)
    times = np.linspace(0, score.duration, frames, endpoint=False)
    max_error = 0.0
    max_alpha = 0.0
    for t in times:
        b = native.tempo.time_to_beat(float(t))
        states = runtime.state(float(t))
        for ent in score.entities:
            if ent.id not in states:
                continue
            ref = (
                native.line_state(ent.id, b)
                if ent.kind == "line"
                else native.note_state(ent.id - len(native.lines), b)
            )
            got = states[ent.id]
            if max(ref["alpha"], got["alpha"]) > 0.001:
                translation = math.hypot(
                    (ref["x"] - got["x"]) * 960, (ref["y"] - got["y"]) * 540
                )
                rotation = (
                    abs(ref["rotation"] - got["rotation"])
                    * math.pi
                    / 180
                    * max(ent.width * 960, ent.height * 540)
                )
                scale = max(
                    abs(ref["scale_x"] - got["scale_x"]) * ent.width * 960,
                    abs(ref["scale_y"] - got["scale_y"]) * ent.height * 540,
                )
                max_error = max(max_error, translation + rotation + scale)
            max_alpha = max(max_alpha, abs(ref["alpha"] - got["alpha"]))
    return dict(
        frames=frames,
        max_sampled_geometry_error_px=max_error,
        max_alpha_error=max_alpha,
        passed=bool(
            max_error <= score.metadata.get("tolerance_px", 1.0) and max_alpha <= 0.01
        ),
        scope="internal evaluator parity; independent player visual review still required",
    )
