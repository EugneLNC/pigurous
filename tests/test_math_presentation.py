import copy
import numpy as np
import pytest
from choreographer.math_presentation import compile_math
from choreographer.math_geometry import curve, primitives, sample


def source():
    return dict(
        profile="performance-v1",
        duration=4,
        seed=42,
        hue=0.54,
        times=[0, 1, 2, 3, 4],
        controls=[[0.7, 0.3, 0.2, 0.1, 0, t * 2, 0.2, 0.6, 0.2] for t in range(5)],
        poses=[[0, 0, a, 1] for a in [3.1, -3.1, 2.0, -2.0, 3.0]],
        hits=[[1.0, 0.9], [2.1, 0.7]],
    )


def test_closed_mathematical_actions_have_no_closure_seam_and_stay_bounded():
    for family in range(3):
        for phase in (-2, 0, 30):
            pts = np.asarray(curve(family, 0.6, 0.4, 0.3, phase, 0.7))
            np.testing.assert_allclose(pts[0], pts[-1], atol=1e-12)
            assert np.linalg.norm(pts[-2] - pts[0]) < 0.08
            assert np.linalg.norm(pts, axis=1).max() < 0.90


def test_projection_bounds_rotation_and_preserves_accent_timestamps():
    p = source()
    raw = copy.deepcopy(p)
    new = compile_math(p)
    assert p == raw and new["hits"] == p["hits"]
    times = np.asarray(new["times"])
    poses = np.asarray(new["poses"])
    velocity = np.diff(poses[:, 2]) / np.diff(times)
    acceleration = np.diff(velocity) / np.diff(times)[1:]
    assert np.abs(velocity).max() <= 0.24002
    assert np.abs(acceleration).max() <= 0.485
    assert new["times"][-1] == 4 and len(times) == 241
    assert sample(new, 0.999)[0][4] == 0 and sample(new, 1)[0][4] > 0.89
    assert compile_math(p) == new


def test_math_seeking_is_stateless_and_silence_remains_dark():
    p = compile_math(source())
    expected = primitives(p, 2.15)
    assert expected
    primitives(p, 3.8)
    assert primitives(p, 2.15) == expected
    assert all(abs(x) < 1.4 and abs(y) < 1.4 for _, pts, *_ in expected for x, y in pts)
    p["controls"] = [[0, *v[1:]] for v in p["controls"]]
    assert primitives(p, 2.15) == []


def test_projection_rejects_nonfinite_or_incomplete_tracks():
    p = source()
    p["controls"][2][2] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        compile_math(p)
    p = source()
    p["times"][-1] = 3.9
    with pytest.raises(ValueError, match="complete"):
        compile_math(p)


def test_score_compile_preserves_raw_score_and_renders_new_profile(tmp_path):
    import json
    from choreographer.math_presentation import mathematical_score
    from choreographer.score import SceneScore, TempoMap, Entity, Event, ScoreRuntime
    from choreographer.render import frame

    score = SceneScore(
        4,
        TempoMap.from_bpm([(0, 120)]).to_list(),
        metadata=dict(
            origin="learned_actions",
            render_profile="performance-v1",
            presentation=source(),
        ),
    )
    for i in range(5):
        score.entities.append(Entity(i, "circle", width=1, height=1))
        score.events.append(Event(0, "create", i))
    raw = tmp_path / "raw.json"
    output = tmp_path / "math.json"
    score.save(raw)
    original = raw.read_bytes()
    report = mathematical_score(raw, output)
    json.dumps(report)
    assert raw.read_bytes() == original
    updated = SceneScore.load(output)
    assert updated.metadata["render_profile"] == "mathematical-v2"
    assert (
        updated.metadata["presentation"]["hits"]
        == score.metadata["presentation"]["hits"]
    )
    image = frame(ScoreRuntime(updated), 2.1, 320, 180)
    assert np.asarray(image).std() > 2
    with pytest.raises(ValueError, match="Preserve"):
        mathematical_score(raw, raw)
