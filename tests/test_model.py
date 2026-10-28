import numpy as np
import torch
from choreographer.model import ModelConfig, Choreographer, loss
from choreographer.representation import quantize


def test_forward_backward_resume_and_causality(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(7)
    cfg = ModelConfig(
        audio_dim=128,
        d_model=32,
        layers=1,
        heads=4,
        planner_dim=32,
        planner_layers=1,
        max_entities=4,
        dropout=0,
        checkpointing=True,
    )
    m = Choreographer(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    cats = torch.tensor([[[1, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [2, 0, 0, 0]]] * 2)
    bins, res = quantize(np.zeros((2, 4, 6)))
    bins = torch.tensor(bins)
    res = torch.tensor(res)
    audio = torch.randn(2, 12, 128)
    tr = torch.randn(2, 12, 129)
    whole = torch.randn(2, 48, 128)
    index = torch.tensor([0, 1])
    state = torch.zeros(2, 4, 31)
    out = m(cats[:, :-1], bins[:, :-1], res[:, :-1], audio, tr, whole, index, state, 3)
    value, _ = loss(
        out,
        cats[:, 1:],
        bins[:, 1:],
        res[:, 1:],
        torch.zeros(2, 3, 8),
        torch.zeros(2, 3, dtype=torch.long),
        3,
    )
    value.backward()
    assert m.audio.weight.grad.abs().sum() > 0
    assert m.transient[0].weight.grad.abs().sum() > 0
    opt.step()
    torch.save(
        {"model": m.state_dict(), "optimizer": opt.state_dict()}, tmp_path / "resume.pt"
    )
    restored = Choreographer(cfg)
    ck = torch.load(tmp_path / "resume.pt", weights_only=True)
    restored.load_state_dict(ck["model"])
    m.eval()
    restored.eval()
    with torch.no_grad():
        a = m(cats, bins, res, audio, tr, whole, index, state)["cats"][0]
        b = restored(cats, bins, res, audio, tr, whole, index, state)["cats"][0]
        torch.testing.assert_close(a, b)
        changed = cats.clone()
        changed[:, -1, 0] = 5
        c = m(changed, bins, res, audio, tr, whole, index, state)["cats"][0]
        torch.testing.assert_close(a[:, :-1], c[:, :-1])
        condition = m.condition(audio, tr, whole, index, state)
        cached = m(
            cats,
            bins,
            res,
            audio,
            tr,
            whole,
            index,
            state,
            conditioning=condition,
            last_only=True,
        )["cats"][0]
        torch.testing.assert_close(a[:, -1:], cached)
