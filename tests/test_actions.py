import json
import numpy as np
import pytest
from choreographer.action_data import grouped_splits, fit_vocabulary
from choreographer.actions import execute_motion
from choreographer.performance_render import primitives


def test_group_union_preserves_holdout():
    rows = [
        dict(song_id="A", audio_hash="one", split="test"),
        dict(song_id="B", audio_hash="one"),
        dict(song_id="B", audio_hash="two"),
    ]
    assert {r["split"] for r in grouped_splits(rows)} == {"test"}
    rows[1]["split"] = "train"
    with pytest.raises(ValueError, match="Conflicting"):
        grouped_splits(rows)


def test_train_only_vocabulary(tmp_path):
    rng = np.random.default_rng(4)
    np.savez(tmp_path / "train.npz", motion=rng.random((100, 12)).astype(np.float32))
    np.savez(tmp_path / "test.npz", motion=np.zeros((100, 12), np.float32))
    records = [
        dict(tensor="train.npz", split="train"),
        dict(tensor="test.npz", split="test"),
    ]
    before = fit_vocabulary(records, tmp_path, 4)
    np.savez(tmp_path / "test.npz", motion=np.ones((100, 12), np.float32))
    np.testing.assert_array_equal(before, fit_vocabulary(records, tmp_path, 4))


def test_action_execution_is_seekable_and_layers_can_be_off():
    motion = np.full((64, 12), 0.5)
    accent = np.zeros(64)
    accent[32] = 0.9
    p = execute_motion(motion, accent, np.zeros(64), 4)
    at = primitives(p, 2.05)
    assert at and any(v[0] == 3 for v in at)
    primitives(p, 3.5)
    assert primitives(p, 2.05) == at
    p["hits"] = []
    p["controls"] = [[1, 0, 0, 0, 0, 0, 0, 0, 0] for _ in p["times"]]
    assert primitives(p, 2) == []


def test_model_predictions_depend_on_audio_and_restore(tmp_path):
    import torch
    from choreographer.action_model import ActionDirector
    from choreographer.action_training import predict, load_model

    torch.set_num_threads(2)
    torch.manual_seed(42)
    vocab = np.random.default_rng(42).random((8, 12)).astype(np.float32)
    model = ActionDirector(vocab, visibility_head=True, direct_effects=True).eval()
    audio = np.random.default_rng(4).normal(size=(40, 897)).astype(np.float32)
    pred = predict(model, audio)
    ck = tmp_path / "ck.pt"
    torch.save(
        dict(
            vocabulary=vocab,
            model=model.state_dict(),
            visibility_head=True,
            direct_effects=True,
        ),
        ck,
    )
    restored, _ = load_model(ck, "cpu")
    other = predict(restored, audio)
    np.testing.assert_allclose(pred["motion"], other["motion"], atol=1e-6)
    np.testing.assert_allclose(pred["effects"], other["effects"], atol=1e-6)
    assert (
        np.mean(
            np.abs(
                pred["motion"] - predict(restored, audio, condition="zero")["motion"]
            )
        )
        > 1e-4
    )
    assert np.isfinite(pred["motion"]).all()


def test_training_batch_accepts_windows_int32_indices():
    import torch
    from choreographer.action_training import collate, forward
    from choreographer.action_model import ActionDirector, objective

    row = dict(
        audio=np.zeros((4, 897), np.float32),
        times=np.arange(4, dtype=np.float32),
        motion=np.full((4, 12), 0.5, np.float32),
        accent=np.zeros(4, np.float32),
        mask=np.ones(4, bool),
        whole=np.zeros((1, 768), np.float32),
        plan=np.full((1, 12), 0.5, np.float32),
        indices=np.zeros(4, np.int32),
    )
    batch = collate([row])
    model = ActionDirector(
        np.random.default_rng(42).random((8, 12)),
        visibility_head=True,
        direct_effects=True,
    )
    assert batch["indices"].dtype == torch.int64
    loss = objective(forward(model, batch), batch, model.vocabulary)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.visibility.weight.grad.abs().sum() > 0
    assert model.effects.weight.grad.abs().sum() > 0


def test_renderer_executes_predicted_effect_selection_without_reclassifying():
    motion = np.full((64, 12), 0.5)
    a = np.tile([1.0, 0.0, 0.0], (64, 1))
    b = np.tile([0.0, 0.0, 1.0], (64, 1))
    first = execute_motion(motion, np.zeros(64), np.zeros(64), 4, effects=a)
    second = execute_motion(motion, np.zeros(64), np.zeros(64), 4, effects=b)
    assert first["controls"][16][-3:] == [1.0, 0.0, 0.0]
    assert second["controls"][16][-3:] == [0.0, 0.0, 1.0]
    assert {v[0] for v in primitives(first, 2)} <= {0, 4}
    assert {v[0] for v in primitives(second, 2)} <= {2, 4}
