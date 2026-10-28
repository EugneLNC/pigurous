import json
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from choreographer.score import SceneScore, Entity, Event, TempoMap
from choreographer.training import train
from choreographer.generation import generate
from choreographer.model import ModelConfig
from choreographer.dataset import HardNegativeBatches


def test_end_to_end_checkpoint_resume_and_seed(tmp_path):
    torch.set_num_threads(2)
    records = []
    for i, split in enumerate(["train", "train", "val"]):
        score = SceneScore(
            2,
            TempoMap.from_bpm([(0, 120)]).to_list(),
            [Entity(0)],
            [Event(0, "create", 0), Event(0, "transform", 0, 4, "x", -0.5, 0.5)],
            metadata={"song_id": str(i)},
        )
        sp = tmp_path / f"{i}.json"
        score.save(sp)
        fp = tmp_path / f"{i}.npz"
        rng = np.random.default_rng(i)
        times = np.arange(0, 2, 0.01)
        np.savez(
            fp,
            semantic=rng.normal(size=(200, 128)).astype("float32"),
            semantic_times=times,
            transient=rng.normal(size=(200, 129)).astype("float32"),
            transient_times=times,
            metadata_json=json.dumps(
                {"backend": "synthetic_test", "audio_sha256": str(i), "duration": 2}
            ),
            tempo_json=json.dumps(score.tempo_map),
        )
        records.append(
            {
                "song_id": str(i),
                "split": split,
                "score": str(sp),
                "features": str(fp),
                "audio_hash": str(i),
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(records))
    cfg = asdict(
        ModelConfig(
            audio_dim=128,
            d_model=32,
            layers=1,
            heads=4,
            planner_dim=32,
            planner_layers=1,
            max_entities=4,
            dropout=0,
            checkpointing=False,
        )
    )
    train(
        manifest, tmp_path / "uninterrupted", cfg, epochs=2, batch_size=2, accumulate=1
    )
    train(
        manifest,
        tmp_path / "resumed",
        cfg,
        epochs=2,
        batch_size=2,
        accumulate=1,
        patience=0,
    )
    train(
        manifest,
        tmp_path / "resumed",
        cfg,
        epochs=2,
        batch_size=2,
        accumulate=1,
        resume=tmp_path / "resumed/last.pt",
    )
    a = torch.load(tmp_path / "uninterrupted/last.pt", weights_only=False)
    b = torch.load(tmp_path / "resumed/last.pt", weights_only=False)
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    first = generate(
        tmp_path / "resumed/last.pt",
        records[2]["features"],
        tmp_path / "a.json",
        seed=42,
        max_events=8,
    )
    second = generate(
        tmp_path / "resumed/last.pt",
        records[2]["features"],
        tmp_path / "b.json",
        seed=42,
        max_events=8,
    )
    assert asdict(first) == asdict(second)
    first.validate()


def test_hard_negatives_keep_song_distinct_and_all_samples():
    items = [{"song_id": str(i // 3), "bpm": 120 + i, "energy": 0.5} for i in range(10)]
    sampler = HardNegativeBatches(items, 3)
    batches = list(sampler)
    assert len(batches) == len(sampler)
    assert sorted(i for b in batches for i in b) == list(range(10))
    for batch in batches:
        assert len({items[i]["song_id"] for i in batch}) == len(batch)
