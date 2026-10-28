from pathlib import Path
import json
import numpy as np
import torch
from .audio import Features, sha256
from .model import Choreographer, ModelConfig
from .representation import OPS, BIN_SIZES, quantize, dequantize, state_memory
from .score import SceneScore, TempoMap, Entity, Event, PROPERTIES, KINDS, ScoreRuntime


def generate(
    checkpoint,
    features,
    output,
    seed=42,
    device="cpu",
    temperature=0.8,
    max_events=2048,
    condition="correct",
    no_global=False,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ModelConfig(**ck["config"])
    model = Choreographer(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    feature = Features(features)
    tempo = feature.tempo
    duration = feature.metadata["duration"]
    if feature.semantic.shape[1] != cfg.audio_dim:
        raise ValueError("Checkpoint feature dimension mismatch")
    if ck["provenance"]["feature_backend"] != feature.metadata["backend"]:
        raise ValueError("Checkpoint feature backend mismatch")
    score = SceneScore(
        duration,
        tempo.to_list(),
        metadata=dict(
            origin="model",
            checkpoint=str(Path(checkpoint).resolve()),
            checkpoint_sha256=sha256(checkpoint),
            training_stage=ck["stage"],
            seed=seed,
            audio=feature.metadata,
            condition=condition,
            no_global=no_global,
        ),
    )
    begin, end = tempo.time_to_beat(0), tempo.time_to_beat(duration)
    anchors = np.arange(begin, end, 16)
    whole = feature.context(tempo, begin, end, 1)[0]
    if condition == "zero":
        whole[:] = 0
    elif condition == "shuffle":
        np.random.shuffle(whole)

    def tensor(v, dtype=None):
        return torch.as_tensor(v, dtype=dtype, device=device)[None]

    whole = tensor(whole)
    alive = {}
    used = set()
    next_id = 0
    slot_ids = {}
    caps = []

    def sample(logits, allowed=None):
        logits = logits.float().clone()
        if allowed is not None:
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[allowed] = False
            logits[mask] = -torch.inf
        return (
            int(logits.argmax())
            if temperature <= 0
            else int(torch.multinomial(torch.softmax(logits / temperature, -1), 1))
        )

    with torch.inference_mode(), torch.autocast(
        "cuda" if device.startswith("cuda") else "cpu",
        dtype=torch.bfloat16,
        enabled=device.startswith("cuda"),
    ):
        global_plan = model.plan(whole)
        for index, anchor in enumerate(anchors):
            state_before = (
                ScoreRuntime(score).state(tempo.beat_to_time(float(anchor)))
                if score.events
                else {}
            )
            stop = min(end, anchor + 16)
            audio, tr, _ = feature.context(tempo, anchor - 4, stop + 4)
            if condition == "zero":
                audio[:] = 0
                tr[:] = 0
            elif condition == "shuffle":
                np.random.shuffle(audio)
                np.random.shuffle(tr)
            state = (
                state_memory(
                    score,
                    float(anchor),
                    {eid: slot for slot, eid in slot_ids.items()},
                    cfg.max_entities,
                    include_boundary=True,
                )
                if score.events
                else np.zeros((cfg.max_entities, 31), np.float32)
            )
            audio_t, tr_t, state_t, index_t = (
                tensor(audio),
                tensor(tr),
                tensor(state),
                torch.tensor([index], device=device),
            )
            stage = 1 if no_global else ck["stage"]
            conditioning = model.condition(
                audio_t, tr_t, whole, index_t, state_t, stage, plan=global_plan
            )
            cats = [[1, 0, 0, 0]]
            nums = [[0.0] * 6]
            current = float(anchor)
            terminated = False
            termination_reason = "event_budget"
            for step in range(max_events):
                bins, res = quantize(nums)
                pred = model(
                    tensor(cats, torch.long),
                    tensor(bins, torch.long),
                    tensor(res, torch.float32),
                    audio_t,
                    tr_t,
                    whole,
                    index_t,
                    state_t,
                    stage,
                    conditioning=conditioning,
                    last_only=True,
                )
                allowed = [2]
                free = [s for s in range(cfg.max_entities) if s not in used]
                if free:
                    allowed.append(3)
                if alive:
                    allowed.extend([4, 5, 6])
                op = sample(pred["cats"][0][0, -1], allowed)
                if op == 2:
                    terminated = True
                    termination_reason = "eos"
                    break
                hidden = pred["hidden"][0, -1]
                prefix = torch.tensor([op], device=device)
                slot = sample(
                    model.field_logits(hidden, prefix), free if op == 3 else list(alive)
                )
                prefix = torch.tensor([op, slot], device=device)
                prop = sample(model.field_logits(hidden, prefix))
                prefix = torch.tensor([op, slot, prop], device=device)
                kind = sample(model.field_logits(hidden, prefix))
                nb, nr = model.numeric(
                    hidden, torch.tensor([op, slot, prop, kind], device=device)
                )
                bs = [sample(v) for v in nb]
                rr = nr.float().cpu().numpy()
                values = dequantize(bs, rr)
                values = np.nan_to_num(values)
                delta = float(np.clip(values[0], 0, stop - current))
                current += delta
                length = float(np.clip(values[1], 0, stop - current))
                values[0] = delta
                values[1] = length
                # Residuals may encode values outside the coarse vocabulary.
                # Keep unwrapped rotation; constrain only declared output domains.
                low, high = (
                    (0, 1)
                    if PROPERTIES[prop] in ("alpha", "red", "green", "blue")
                    else (
                        (-10000, 10000) if PROPERTIES[prop] == "rotation" else (-16, 16)
                    )
                )
                values[2:4] = np.clip(values[2:4], low, high)
                values[4:] = np.clip(values[4:], 0.002, 8)
                if op == 3:
                    eid = next_id
                    next_id += 1
                    used.add(slot)
                    alive[slot] = eid
                    slot_ids[slot] = eid
                    score.entities.append(
                        Entity(
                            eid,
                            KINDS[kind],
                            width=float(values[4]),
                            height=float(values[5]),
                        )
                    )
                eid = alive[slot]
                scale = 360 if PROPERTIES[prop] == "rotation" else 1
                score.events.append(
                    Event(
                        current,
                        OPS[op],
                        eid,
                        length if op in (4, 5) else 0,
                        PROPERTIES[prop],
                        float(values[2]) * scale,
                        float(values[3]) * scale,
                    )
                )
                if op == 6:
                    del alive[slot]
                cats.append([op, slot, prop, kind])
                nums.append(values.tolist())
                if current >= stop - 1e-8:
                    terminated = True
                    termination_reason = "beat_boundary"
                    break
            if not terminated:
                caps.append(index)
            score.segments.append(
                dict(
                    start=float(anchor),
                    end=float(stop),
                    events=len(cats) - 1,
                    termination=termination_reason,
                    plan_used=stage >= 2,
                    visual_plan={
                        "descriptor": global_plan[1][0, index].float().cpu().tolist(),
                        "code": int(global_plan[2][0, index].argmax()),
                    },
                    state_before=state_before,
                )
            )
            print(
                f"Generated block {index+1}/{len(anchors)}: {len(cats)-1} events, complete={terminated}",
                flush=True,
            )
    score.metadata["budget_limited_segments"] = caps
    score.metadata["complete_generation"] = not caps and bool(score.entities)
    score.metadata["degenerate_empty"] = not bool(score.entities)
    score.save(output)
    return score
