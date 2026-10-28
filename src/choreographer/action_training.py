"""Resumable single-GPU training and held-out conditioning evaluation."""
import json, os, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .action_model import ActionDirector, objective
from .action_data import RATE

WINDOW = 256


def pool(x, stride=RATE * 2):
    return np.stack([x[i : i + stride].mean(0) for i in range(0, len(x), stride)])


class ActionDataset(Dataset):
    def __init__(self, manifest, split):
        self.root = Path(manifest).parent
        self.records = [
            r
            for r in json.loads(Path(manifest).read_text(encoding="utf-8"))
            if r["split"] == split
        ]
        self.songs = []
        self.index = []
        for i, r in enumerate(self.records):
            z = dict(np.load(self.root / r["tensor"]))
            z["whole"] = pool(z["audio"][:, :768])
            z["plan"] = pool(z["motion"])
            self.songs.append(z)
            self.index.extend((i, start) for start in range(0, len(z["audio"]), WINDOW))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        song, start = self.index[i]
        z = self.songs[song]
        n = min(WINDOW, len(z["audio"]) - start)

        def block(k):
            x = z[k][start : start + n]
            return np.pad(x, ((0, WINDOW - n),) + ((0, 0),) * (x.ndim - 1), mode="edge")

        return dict(
            audio=block("audio"),
            times=block("times").astype(np.float32),
            motion=block("motion"),
            accent=block("accent"),
            mask=np.arange(WINDOW) < n,
            whole=z["whole"],
            plan=z["plan"],
            indices=np.minimum(
                (start + np.arange(WINDOW)) // (RATE * 2), len(z["whole"]) - 1
            ),
        )


def collate(rows):
    length = max(len(r["whole"]) for r in rows)
    result = {
        k: torch.as_tensor(np.stack([r[k] for r in rows]))
        for k in rows[0]
        if k not in ("whole", "plan")
    }
    for k in ("whole", "plan"):
        result[k] = torch.as_tensor(
            np.stack([np.pad(r[k], ((0, length - len(r[k])), (0, 0))) for r in rows])
        )
    result["whole_mask"] = torch.as_tensor(
        np.stack([np.arange(length) >= len(r["whole"]) for r in rows])
    )
    result["indices"] = result["indices"].long()
    return result


def forward(model, batch):
    return model(
        *[batch[k] for k in ("audio", "times", "whole", "whole_mask", "indices")]
    )


def device_setup(device):
    if device.startswith("cuda"):
        if "CHOREO_SHARED_MEMORY_MB" in os.environ:
            cap = int(os.environ["CHOREO_SHARED_MEMORY_MB"]) * 1024**2
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(0.94, cap / total))
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(min(8, os.cpu_count() or 1))


def train(
    manifest,
    output,
    epochs=40,
    device="cuda",
    batch_size=8,
    resume=None,
    initialize=None,
    seed=42,
):
    device_setup(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    dataset = ActionDataset(manifest, "train")
    validation = ActionDataset(manifest, "val")
    if not len(dataset) or not len(validation):
        raise ValueError("Training and validation songs required")
    vocabulary = np.load(Path(manifest).parent / "vocabulary.npy")
    resume_config = (
        torch.load(resume, map_location="cpu", weights_only=False) if resume else None
    )
    model = ActionDirector(
        vocabulary,
        visibility_head=resume_config.get("visibility_head", False) if resume else True,
        direct_effects=resume_config.get("direct_effects", False) if resume else True,
    ).to(device)
    transferred = []
    if initialize:
        source = torch.load(initialize, map_location="cpu", weights_only=False)
        is_action = source.get("format") == "motion-actions-v1"
        if is_action and not np.array_equal(source["vocabulary"], vocabulary):
            raise ValueError("Initialization action vocabulary mismatch")
        state = source.get("model", source.get("state_dict", {}))
        target = model.state_dict()
        transferred = [
            k
            for k, v in state.items()
            if (is_action or k.startswith(("global_in.", "planner.")))
            and k in target
            and target[k].shape == v.shape
        ]
        model.load_state_dict({k: state[k] for k in transferred}, strict=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    start = 0
    best = float("inf")
    stale = 0
    steps = 0
    if resume:
        ck = resume_config
        digest = __import__("hashlib").sha256(Path(manifest).read_bytes()).hexdigest()
        if ck.get("manifest_sha256") != digest:
            raise ValueError(
                "Resume dataset manifest mismatch; start a separately recorded run"
            )
        if not np.array_equal(ck["vocabulary"], vocabulary):
            raise ValueError("Resume vocabulary mismatch")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start = ck["epoch"] + 1
        best = ck["best"]
        steps = ck["steps"]
        torch.set_rng_state(ck["rng"])
        np.random.set_state(ck["numpy_rng"])
        if device.startswith("cuda") and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng"])
        transferred = ck.get("transferred", [])
    if epochs <= start:
        raise ValueError("--epochs must exceed the completed checkpoint epoch count")
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    vl = DataLoader(validation, batch_size=batch_size, collate_fn=collate)
    started = time.time()
    for epoch in range(start, epochs):
        model.train()
        losses = []
        # Absolute schedule keeps continuation deterministic across invocations.
        lr = (
            2e-4
            * min(1, (epoch + 1) / 3)
            * (0.25 + 0.75 * (1 + np.cos(np.pi * min(epoch, 60) / 60)) / 2)
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.split(":")[0],
                dtype=torch.bfloat16,
                enabled=device.startswith("cuda"),
            ):
                result = forward(model, batch)
                loss = objective(result, batch, model.vocabulary)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite action training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss))
            steps += 1
        model.eval()
        vals = []
        with torch.inference_mode():
            for batch in vl:
                batch = {k: v.to(device) for k, v in batch.items()}
                vals.append(
                    float(objective(forward(model, batch), batch, model.vocabulary))
                )
        val = float(np.mean(vals))
        improved = val < best
        best = min(best, val)
        stale = 0 if improved else stale + 1
        report = dict(
            epoch=epoch,
            train=float(np.mean(losses)),
            validation=val,
            best=best,
            steps=steps,
            seconds=time.time() - started,
            peak_gib=torch.cuda.max_memory_allocated() / 1024**3
            if device.startswith("cuda")
            else 0,
        )
        with (root / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(report) + "\n")
        print(json.dumps(report), flush=True)
        ck = dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            epoch=epoch,
            best=best,
            steps=steps,
            vocabulary=vocabulary,
            rng=torch.get_rng_state(),
            numpy_rng=np.random.get_state(),
            cuda_rng=torch.cuda.get_rng_state() if device.startswith("cuda") else None,
            transferred=transferred,
            seed=seed,
            format="motion-actions-v1",
            visibility_head=model.visibility is not None,
            direct_effects=model.effects is not None,
            initialized_from=initialize,
            manifest_sha256=__import__("hashlib")
            .sha256(Path(manifest).read_bytes())
            .hexdigest(),
        )
        torch.save(ck, root / "last.tmp")
        (root / "last.tmp").replace(root / "last.pt")
        if improved:
            torch.save(
                {
                    k: v
                    for k, v in ck.items()
                    if k not in ("optimizer", "rng", "numpy_rng", "cuda_rng")
                },
                root / "best.pt",
            )
        if stale >= 10:
            break
    summary = dict(
        parameters=sum(p.numel() for p in model.parameters()),
        train_songs=len(dataset.records),
        val_songs=len(validation.records),
        train_windows=len(dataset),
        best=best,
        steps=steps,
        transferred_tensors=len(transferred),
        last_epoch=epoch,
        peak_gib=report["peak_gib"],
    )
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_model(checkpoint, device):
    device_setup(device)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = ActionDirector(
        ck["vocabulary"],
        visibility_head=ck.get("visibility_head", False),
        direct_effects=ck.get("direct_effects", False),
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


@torch.inference_mode()
def predict(model, audio, device="cpu", condition="correct"):
    audio = np.asarray(audio, dtype=np.float32).copy()
    if condition == "zero":
        audio[:] = 0
    elif condition == "shuffle":
        # Shuffle two-second blocks while preserving local timbre/onsets.
        chunks = [audio[i : i + RATE * 2] for i in range(0, len(audio), RATE * 2)]
        order = np.random.default_rng(42).permutation(len(chunks))
        audio = np.concatenate([chunks[i] for i in order])
    elif condition != "correct":
        raise ValueError("Unknown conditioning")
    whole = pool(audio[:, :768])
    n = len(audio)
    logits = np.zeros((n, len(model.vocabulary)), np.float32)
    residual = np.zeros((n, 12), np.float32)
    accent = np.zeros(n, np.float32)
    visibility = np.zeros((n, 2), np.float32)
    effects = np.zeros((n, 3), np.float32)
    weights = np.zeros(n, np.float32)
    # Overlap/add predictions: same object tracks survive window boundaries.
    for start in range(0, n, WINDOW // 2):
        end = min(start + WINDOW, n)
        size = end - start
        batch = dict(
            audio=torch.tensor(audio[None, start:end], device=device),
            times=torch.arange(start, end, device=device)[None] / RATE,
            whole=torch.tensor(whole[None], device=device),
            whole_mask=torch.zeros((1, len(whole)), dtype=torch.bool, device=device),
            indices=torch.arange(start, end, device=device)[None] // (RATE * 2),
        )
        pred = forward(model, batch)
        w = np.maximum(0.05, np.hanning(size))
        logits[start:end] += pred["logits"][0].cpu().numpy() * w[:, None]
        residual[start:end] += pred["residual"][0].cpu().numpy() * w[:, None]
        accent[start:end] += pred["accent"][0].sigmoid().cpu().numpy() * w
        if "visibility" in pred:
            visibility[start:end] += (
                pred["visibility"][0].sigmoid().cpu().numpy() * w[:, None]
            )
        if "effects" in pred:
            effects[start:end] += (
                pred["effects"][0].softmax(-1).cpu().numpy() * w[:, None]
            )
        weights[start:end] += w
    logits /= weights[:, None]
    residual /= weights[:, None]
    accent /= weights
    tokens = logits.argmax(1)
    motion = np.clip(model.vocabulary.cpu().numpy()[tokens] + residual, 0, 1)
    if model.visibility is not None:
        motion[:, [0, 7]] = visibility / weights[:, None]
    result = dict(tokens=tokens, motion=motion, accent=accent, logits=logits)
    if model.effects is not None:
        result["effects"] = effects / weights[:, None]
    return result
