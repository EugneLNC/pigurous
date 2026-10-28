import argparse
from pathlib import Path
import torch

p = argparse.ArgumentParser()
p.add_argument("checkpoint")
p.add_argument("output")
p.add_argument("--expected-stage", type=int)
a = p.parse_args()
ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
if a.expected_stage is not None and ck["stage"] != a.expected_stage:
    raise ValueError("Unexpected training stage; refusing to mislabel weights")
out = {k: ck[k] for k in ("model", "config", "stage", "codebook", "provenance")}
out["purpose"] = "inference_only"
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
torch.save(out, a.output)
print(a.output)
