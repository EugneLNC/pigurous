"""Bounded autoregressive previous-block geometry roll-in for late training."""
import numpy as np
import torch
from .representation import quantize, dequantize
from .score import DEFAULTS, PROPERTIES


@torch.no_grad()
def rollout_state(model, item, device, stage=2, max_steps=64):
    was_training = model.training
    model.eval()
    state = item["state"].copy()
    alive = set(np.flatnonzero(state[:, 0] > 0).tolist())
    current = item["anchor"]
    stop = item["end"]
    cats = [[1, 0, 0, 0]]
    nums = [[0.0] * 6]

    def tensor(v, dtype=None):
        return torch.as_tensor(v, dtype=dtype, device=device)[None]

    def advance(delta):
        remaining = state[:, 5::3] * 16
        values = state[:, 4::3]
        targets = state[:, 6::3]
        u = np.divide(
            delta, remaining, out=np.ones_like(remaining), where=remaining > 0
        ).clip(0, 1)
        state[:, 4::3] = values + (targets - values) * u
        state[:, 5::3] = np.maximum(0, remaining - delta) / 16

    try:
        audio_t, tr_t, whole_t, index_t, state_t = (
            tensor(item["audio"]),
            tensor(item["transient"]),
            tensor(item["whole"]),
            torch.tensor([item["plan_index"]], device=device),
            tensor(item["state"]),
        )
        conditioning = model.condition(audio_t, tr_t, whole_t, index_t, state_t, stage)
        for _ in range(max_steps):
            bins, res = quantize(nums)
            out = model(
                tensor(cats, torch.long),
                tensor(bins, torch.long),
                tensor(res, torch.float32),
                audio_t,
                tr_t,
                whole_t,
                index_t,
                state_t,
                stage,
                conditioning=conditioning,
                last_only=True,
            )
            op = int(out["cats"][0][0, -1].argmax())
            if op == 2:
                break
            allowed = (
                [i for i in range(len(state)) if i not in alive]
                if op == 3
                else sorted(alive)
            )
            if not allowed:
                break
            hidden = out["hidden"][0, -1]

            def prefix(v):
                return torch.tensor(v, device=device)

            slot = allowed[
                int(model.field_logits(hidden, prefix([op]))[allowed].argmax())
            ]
            prop = int(model.field_logits(hidden, prefix([op, slot])).argmax())
            kind = int(model.field_logits(hidden, prefix([op, slot, prop])).argmax())
            nb, nr = model.numeric(hidden, prefix([op, slot, prop, kind]))
            bins = [int(v.argmax()) for v in nb]
            res = nr.float().cpu().numpy()
            values = dequantize(bins, res)
            delta = float(np.clip(values[0], 0, stop - current))
            advance(delta)
            current += delta
            duration = float(np.clip(values[1], 0, stop - current))
            values[0] = delta
            values[1] = duration
            if op == 3:
                alive.add(slot)
                state[slot] = 0
                state[slot, :4] = [
                    1,
                    kind / 5,
                    max(0.002, values[4]),
                    max(0.002, values[5]),
                ]
                for i, p in enumerate(PROPERTIES):
                    state[slot, 4 + 3 * i] = state[slot, 6 + 3 * i] = DEFAULTS[p]
            elif op == 6:
                alive.remove(slot)
                state[slot] = 0
            elif op == 4:
                state[slot, 4 + 3 * prop : 7 + 3 * prop] = [
                    values[2] if duration else values[3],
                    duration / 16,
                    values[3],
                ]
            cats.append([op, slot, prop, kind])
            nums.append(values.tolist())
            if current >= stop:
                break
        advance(stop - current)
        return state
    finally:
        model.train(was_training)
