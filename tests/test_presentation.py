import json
import numpy as np
import pytest
from choreographer.presentation import compile_presentation, polish
from choreographer.performance_render import primitives, sample
from choreographer.score import SceneScore, TempoMap


def cache(tmp_path, silent=False):
    t = np.arange(0, 4.001, 0.01)
    tr = np.zeros((len(t), 129), np.float32)
    if not silent:
        tr[:, :128] = 0.15
        for at in [1.0, 2.0, 3.0]:
            i = int(at * 100)
            tr[i, 128] = 1
            tr[i : i + 10, :35] = 0.8
    path = tmp_path / ("silent.npz" if silent else "pulse.npz")
    np.savez(
        path,
        semantic=np.zeros((len(t), 128)),
        semantic_times=t,
        transient=tr,
        transient_times=t,
        metadata_json=json.dumps(dict(duration=4, audio_sha256="test")),
        tempo_json=json.dumps(TempoMap.from_bpm([(0, 120)]).to_list()),
    )
    return path


def test_onsets_silence_replay_and_response(tmp_path):
    path = cache(tmp_path)
    p = compile_presentation(path)
    assert [h[0] for h in p["hits"]] == [1.0, 2.0, 3.0]
    assert sample(p, 1.0)[4] > 0.95
    assert sample(p, 0.95)[4] == 0
    assert sample(p, 0.999)[4] == 0
    assert sample(p, 1.4)[4] < 0.06
    expected = primitives(p, 2.1)
    primitives(p, 3.0)
    assert primitives(p, 2.1) == expected
    assert p == compile_presentation(path)
    quiet = compile_presentation(cache(tmp_path, silent=True))
    assert quiet["hits"] == [] and primitives(quiet, 2.0) == []
    assert expected and expected != primitives(p, 2.5)
    assert all(abs(x) < 1.8 and abs(y) < 1.1 for _, pts, *_ in expected for x, y in pts)


def test_polish_rejects_wrong_audio(tmp_path):
    s = SceneScore(4, TempoMap.from_bpm([(0, 120)]).to_list())
    s.save(tmp_path / "raw.json")
    (tmp_path / "audio.bin").write_bytes(b"wrong song")
    with pytest.raises(ValueError, match="Features do not belong"):
        polish(
            tmp_path / "raw.json",
            tmp_path / "audio.bin",
            cache(tmp_path),
            tmp_path / "out.json",
        )
