"""Real full-size BF16 forward/backward/resume test. Use shared_gpu.py remotely."""
import argparse, json, os
from pathlib import Path
import numpy as np
import torch
from choreographer.model import Choreographer, ModelConfig, loss
from choreographer.representation import quantize

parser = argparse.ArgumentParser()
parser.add_argument("--events", type=int, default=512)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--output-prefix", default="artifacts/gpu-smoke")
args = parser.parse_args()
if args.events < 2 or args.batch_size < 1:
    parser.error("events must be >= 2 and batch-size must be >= 1")
torch.set_num_threads(4)
torch.manual_seed(42)
if os.environ.get("CHOREO_SHARED_MEMORY_MB"):
    budget = int(os.environ["CHOREO_SHARED_MEMORY_MB"]) * 1024**2
    torch.cuda.set_per_process_memory_fraction(
        budget / torch.cuda.get_device_properties(0).total_memory
    )
cfg = ModelConfig()
model = Choreographer(cfg).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
b, n = args.batch_size, args.events
cats = torch.zeros(b, n, 4, dtype=torch.long, device="cuda")
cats[:, :, 0] = 4
cats[:, 0, 0] = 1
cats[:, 1, 0] = 3
targets = cats.clone()
targets[:, :-1] = cats[:, 1:]
targets[:, -1, 0] = 2
bins, res = quantize(np.zeros((b, n, 6)))
bins = torch.tensor(bins, device="cuda")
res = torch.tensor(res, device="cuda")
audio = torch.randn(b, 1200, 768, device="cuda")
tr = torch.randn(b, 1200, 129, device="cuda")
whole = torch.randn(b, 512, 768, device="cuda")
state = torch.zeros(b, 128, 31, device="cuda")
torch.cuda.reset_peak_memory_stats()
with torch.autocast("cuda", dtype=torch.bfloat16):
    output = model(
        cats,
        bins,
        res,
        audio,
        tr,
        whole,
        torch.zeros(b, dtype=torch.long, device="cuda"),
        state,
        2,
        next_cats=targets,
    )
    value, _ = loss(
        output,
        targets,
        bins,
        res,
        torch.zeros(b, 32, 8, device="cuda"),
        torch.zeros(b, 32, dtype=torch.long, device="cuda"),
        2,
    )
value.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
opt.step()
prefix = Path(args.output_prefix)
prefix.parent.mkdir(parents=True, exist_ok=True)
path = prefix.with_suffix(".pt")
torch.save({"model": model.state_dict(), "optimizer": opt.state_dict()}, path)
checkpoint = torch.load(path, map_location="cuda", weights_only=True)
model.load_state_dict(checkpoint["model"])
opt.load_state_dict(checkpoint["optimizer"])
report = {
    "device": torch.cuda.get_device_name(0),
    "visible_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "parameters": sum(p.numel() for p in model.parameters()),
    "loss": float(value),
    "finite": bool(torch.isfinite(value)),
    "peak_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
    "batch": b,
    "events": n,
    "checkpoint_reloaded": True,
    "torch": torch.__version__,
}
prefix.with_suffix(".json").write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
