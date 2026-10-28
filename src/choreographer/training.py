from dataclasses import asdict
from pathlib import Path
import json, math, random, time, os
import numpy as np
import torch
from torch.utils.data import DataLoader
from .model import Choreographer, ModelConfig, loss
from .dataset import SceneDataset, collate, HardNegativeBatches


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(
    manifest,
    output,
    config,
    epochs=20,
    stage=1,
    batch_size=2,
    accumulate=16,
    device="cpu",
    resume=None,
    seed=42,
    max_events=2048,
    patience=5,
    memory_limit_gb=30,
    rollin_probability=0.1,
):
    seed_all(seed)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    cfg = ModelConfig(**config)
    trainset = SceneDataset(manifest, "train", max_events, cfg.max_entities)
    valid = SceneDataset(
        manifest, "val", max_events, cfg.max_entities, trainset.codebook
    )
    if trainset.items[0]["audio"].shape[-1] != cfg.audio_dim:
        raise ValueError("Feature dimension does not match config")
    use_amp = device.startswith("cuda")
    scaler = None
    if use_amp and os.environ.get("CHOREO_SHARED_MEMORY_MB"):
        memory_limit_gb = min(
            memory_limit_gb, float(os.environ["CHOREO_SHARED_MEMORY_MB"]) / 1024
        )
        torch.cuda.set_per_process_memory_fraction(
            memory_limit_gb
            * 1024**3
            / torch.cuda.get_device_properties(0).total_memory
        )
    model = Choreographer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    if use_amp:
        torch.cuda.reset_peak_memory_stats()
    loader = (
        DataLoader(
            trainset,
            batch_sampler=HardNegativeBatches(trainset, batch_size),
            collate_fn=collate,
        )
        if stage == 3
        else DataLoader(
            trainset, batch_size=batch_size, shuffle=True, collate_fn=collate
        )
    )
    val_loader = DataLoader(valid, batch_size=batch_size, collate_fn=collate)
    total_steps = max(1, epochs * math.ceil(len(loader) / accumulate))
    warmup = max(1, int(total_steps * 0.05))

    def schedule(step):
        return (
            (step + 1) / warmup
            if step < warmup
            else 0.5
            * (
                1
                + math.cos(
                    math.pi * min(1, (step - warmup) / max(1, total_steps - warmup))
                )
            )
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
    start = 0
    best = float("inf")
    stale = 0
    steps = 0
    if resume:
        ck = torch.load(resume, map_location=device, weights_only=False)
        if ck["config"] != asdict(cfg):
            raise ValueError("Resume model configuration mismatch")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start = ck["epoch"] + 1
        best = ck["best"]
        steps = ck["steps"]
        stale = ck.get("stale", 0)
        random.setstate(ck["rng_python"])
        np.random.set_state(ck["rng_numpy"])
        torch.set_rng_state(ck["rng_torch"].cpu())
        if use_amp and ck.get("rng_cuda"):
            torch.cuda.set_rng_state_all([s.cpu() for s in ck["rng_cuda"]])
        trainset.codebook = ck["codebook"]
        if stage != ck["stage"]:
            best = float("inf")
            stale = 0

    def run(batch, rollin=False):
        b = {k: v.to(device) for k, v in batch.items()}
        if rollin:
            from .rollin import rollout_state

            for j, index in enumerate(batch["indices"].tolist()):
                prev = trainset[index]["previous"]
                if prev is not None and random.random() < rollin_probability:
                    generated = rollout_state(model, trainset[prev], device, stage)
                    # Gold lifecycle keeps teacher targets legal; geometry is generated.
                    b["state"][j, :, 1:] = torch.as_tensor(
                        generated[:, 1:], device=device
                    )
        with torch.autocast(
            device_type="cuda" if use_amp else "cpu",
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            out = model(
                b["cats"][:, :-1],
                b["bins"][:, :-1],
                b["residual"][:, :-1],
                b["audio"],
                b["transient"],
                b["whole"],
                b["plan_index"],
                b["state"],
                stage,
                b["audio_mask"],
                b["whole_mask"],
                next_cats=b["cats"][:, 1:],
            )
            value, parts = loss(
                out,
                b["cats"][:, 1:],
                b["bins"][:, 1:],
                b["residual"][:, 1:],
                b["plan"],
                b["codes"],
                stage,
            )
        return value, parts

    for epoch in range(start, epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        losses = []
        begun = time.time()
        for i, batch in enumerate(loader):
            value, parts = run(batch, rollin=stage >= 2 and epoch >= int(epochs * 0.75))
            if not torch.isfinite(value):
                raise FloatingPointError(f"Non-finite training loss: {parts}")
            group_size = min(accumulate, len(loader) - (i // accumulate) * accumulate)
            (value / group_size).backward()
            losses.append(float(value.detach()))
            if (i + 1) % accumulate == 0 or i + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                scheduler.step()
                opt.zero_grad(set_to_none=True)
                steps += 1
            if (
                use_amp
                and torch.cuda.max_memory_allocated() > memory_limit_gb * 1024**3
            ):
                raise RuntimeError(
                    "Measured memory exceeds configured limit; reduce batch size"
                )
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([float(run(b)[0]) for b in val_loader]))
        improved = vl < best
        best = min(best, vl)
        stale = 0 if improved else stale + 1
        record = dict(
            epoch=epoch,
            train_loss=float(np.mean(losses)),
            val_loss=vl,
            seconds=time.time() - begun,
            steps=steps,
            peak_memory_gb=torch.cuda.max_memory_allocated() / 1024**3
            if use_amp
            else 0,
            stage=stage,
        )
        with (root / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        checkpoint = dict(
            model=model.state_dict(),
            optimizer=opt.state_dict(),
            scheduler=scheduler.state_dict(),
            config=asdict(cfg),
            epoch=epoch,
            best=best,
            stale=stale,
            steps=steps,
            stage=stage,
            codebook=trainset.codebook,
            rng_python=random.getstate(),
            rng_numpy=np.random.get_state(),
            rng_torch=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state_all() if use_amp else None,
            provenance={
                "manifest": str(Path(manifest).resolve()),
                "songs": trainset.songs,
                "seed": seed,
                "feature_backend": __import__(
                    "choreographer.audio", fromlist=["Features"]
                )
                .Features(trainset.songs[0]["features"])
                .metadata["backend"],
            },
        )
        temporary = root / "last.pt.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(root / "last.pt")
        if improved:
            temporary = root / "best.pt.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(root / "best.pt")
        if stale >= patience:
            break
    summary = dict(
        parameters=sum(p.numel() for p in model.parameters()),
        train_samples=len(trainset),
        val_samples=len(valid),
        quarantine=trainset.quarantine + valid.quarantine,
        best_validation=best,
        stage=stage,
    )
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
