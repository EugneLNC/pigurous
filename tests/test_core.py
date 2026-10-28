import json, math
import numpy as np
import pytest
from choreographer.score import *
from choreographer.curves import adaptive_curve, event_value
from choreographer.importer import (
    convert_chart,
    reconstruction_audit,
    NativeChart,
    UnsupportedChart,
)
from choreographer.representation import quantize, dequantize, blocks, grouped_split


def test_tempo_roundtrip_and_offset():
    tm = TempoMap.from_bpm([(0, 120), (4, 180), (12, 60)], 0.25)
    for t in np.linspace(-1, 20, 100):
        assert tm.beat_to_time(tm.time_to_beat(t)) == pytest.approx(t)
    assert tm.beat_to_time(4) == 2.25
    with pytest.raises(ValueError):
        TempoMap([dict(beat=0, time=0, bpm=120), dict(beat=4, time=99, bpm=120)])


def test_seek_overlap_and_lifecycle():
    s = SceneScore(
        8,
        TempoMap.from_bpm([(0, 120)]).to_list(),
        [Entity(0)],
        [
            Event(0, "create", 0),
            Event(0, "transform", 0, 8, "x", 0, 1),
            Event(2, "transform", 0, 2, "x", 3, 4),
            Event(10, "destroy", 0),
        ],
    )
    rt = ScoreRuntime(s)
    assert rt.state(3)[0]["x"] == 4
    assert rt.state(0.5)[0]["x"] == 0.125
    assert rt.state(1.5)[0]["x"] == 3.5
    assert not rt.state(6)
    s.events.append(Event(11, "transform", 0))
    with pytest.raises(ValueError):
        s.validate()


def test_quantization_is_exact_including_overflow():
    rng = np.random.default_rng(8)
    values = rng.normal(size=(100, 6)) * 20
    b, r = quantize(values)
    np.testing.assert_allclose(dequantize(b, r), values, atol=1e-5)


def test_compression_preserves_rotation_turns():
    pts = adaptive_curve(lambda b: b * 720, 0, 4, 0.01)
    assert pts[-1][1] == 2880
    assert len(pts) == 2


def test_song_group_splits():
    records = [
        dict(song_id=str(i), difficulty=d) for i in range(20) for d in ["EZ", "IN"]
    ]
    split = grouped_split(records)
    for sid in range(20):
        assert len({r["split"] for r in split if r["song_id"] == str(sid)}) == 1
    assert sum(r["split"] == "test" for r in split) == 4


def test_recursive_blocks_keep_events_and_carry():
    s = SceneScore(
        16,
        TempoMap.from_bpm([(0, 120)]).to_list(),
        [Entity(0)],
        [Event(0, "create", 0), Event(0, "transform", 0, 32, "x", 0, 1)]
        + [Event(i, "pulse", 0, 0.25) for i in range(1, 31)],
    )
    chunks = list(blocks(s, max_events=12, max_entities=4))
    assert len(chunks) > 2
    assert sum(int((c["cats"][:, 0] == 5).sum()) for c in chunks) == 30
    second = chunks[1]
    assert second["state"][0, 0] == 1
    assert second["state"][0, 4] == pytest.approx(second["anchor"] / 32, abs=1e-6)


def rpe_fixture():
    def ev(a, b, x, y):
        return {"startTime": a, "endTime": b, "start": x, "end": y, "easingType": 1}

    return {
        "META": {"offset": 250},
        "BPMList": [
            {"startTime": [0, 0, 1], "bpm": 120},
            {"startTime": [4, 0, 1], "bpm": 180},
        ],
        "judgeLineList": [
            {
                "father": -1,
                "eventLayers": [
                    {
                        "moveXEvents": [ev(0, 12, 0, 100)],
                        "moveYEvents": [ev(0, 12, 0, 40)],
                        "rotateEvents": [ev(0, 12, 0, 720)],
                        "alphaEvents": [ev(0, 12, 255, 255)],
                        "speedEvents": [ev(0, 12, 2, 2)],
                    }
                ],
            },
            {
                "father": 0,
                "rotateWithFather": True,
                "eventLayers": [
                    {
                        "moveXEvents": [ev(0, 12, 100, 100)],
                        "moveYEvents": [ev(0, 12, 0, 0)],
                        "rotateEvents": [ev(0, 12, 0, 0)],
                        "alphaEvents": [ev(0, 12, 255, 255)],
                    }
                ],
                "notes": [
                    {
                        "type": 1,
                        "startTime": 4,
                        "endTime": 4,
                        "positionX": 0,
                        "above": 1,
                        "speed": 1,
                        "visibleTime": 1,
                    }
                ],
            },
        ],
    }


def test_conversion_parents_variable_tempo_and_audit(tmp_path):
    p = tmp_path / "rpe.json"
    p.write_text(json.dumps(rpe_fixture()))
    s, n = convert_chart(p, 4, song_id="fixture")
    report = reconstruction_audit(s, n, frames=300)
    assert report["passed"], report
    s.save(tmp_path / "score.json")
    SceneScore.load(tmp_path / "score.json")
    child = n.line_state(1, 0)
    assert child["x"] == pytest.approx(100 / 675)
    with pytest.raises(UnsupportedChart):
        raw = rpe_fixture()
        raw["judgeLineList"][0]["Texture"] = "custom.png"
        NativeChart(raw)


def test_shared_server_never_gpu7():
    import importlib.util

    spec = importlib.util.spec_from_file_location("guard", "scripts/shared_gpu.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert (
        mod.select(
            [["7", "forbidden", "0", "32000"], ["5", "allowed", "8000", "32000"]], 8192
        )
        == "allowed"
    )
    with pytest.raises(RuntimeError):
        mod.select([["7", "forbidden", "0", "32000"]], 8192)
    rows = [["6", "shared", "19000", "32607"], ["7", mod.PROHIBITED, "0", "32607"]]
    uuid = mod.select(rows, 8192)
    assert uuid == "shared"
    assert mod.memory_budget(rows, uuid) == 11559
    idle = [["5", "idle", "0", "32607"]]
    assert mod.memory_budget(idle, "idle") == 30559
    # Required memory is a launch minimum, rather than an 8 GiB allocation cap.
    assert mod.memory_budget(idle, "idle") > 8192
    with pytest.raises(RuntimeError):
        mod.select([["6", "busy", "25000", "32607"]], 8192)


def test_rpe_identity_controls_and_line_draw_order(tmp_path):
    raw = rpe_fixture()
    raw["judgeLineList"][0]["extended"] = {
        "inclineEvents": [
            {
                "start": 0,
                "end": 0,
                "startTime": [0, 0, 1],
                "endTime": [1, 0, 1],
                "easingType": 0,
            }
        ]
    }
    raw["judgeLineList"][0]["posControl"] = [
        {"pos": 1, "x": 0},
        {"pos": 1, "x": 9999999},
    ]
    NativeChart(raw)
    raw["judgeLineList"][1]["zOrder"] = -1
    path = tmp_path / "rpe.json"
    path.write_text(json.dumps(raw))
    score, _ = convert_chart(path, 4, "fixture", lines_only=True)
    assert [e.id for e in score.entities] == [1, 0]
    assert score.metadata["projection"] == "lines_only"
