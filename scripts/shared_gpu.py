"""Run only inside an isolated project directory; never select physical GPU 7."""
import argparse, os, subprocess, sys, csv, io

PROHIBITED = "GPU-82462e9a-c81d-b66a-8910-eecc59fc0ca1"


def select(rows, required_mb, headroom_mb=2048):
    candidates = [
        r
        for r in rows
        if int(r[0]) != 7
        and r[1].strip() != PROHIBITED
        and int(r[3]) - int(r[2]) >= required_mb + headroom_mb
    ]
    if not candidates:
        raise RuntimeError(
            "No permitted GPU has enough headroom. GPU 7 remains excluded."
        )
    return max(candidates, key=lambda r: int(r[3]) - int(r[2]))[1].strip()


def memory_budget(rows, uuid, limit_gb=30, headroom_mb=2048):
    row = next(r for r in rows if r[1].strip() == uuid)
    free_mb = int(row[3]) - int(row[2])
    return min(int(limit_gb * 1024), free_mb - headroom_mb)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--required-mb", type=int, default=8192)
    p.add_argument("--memory-limit-gb", type=float, default=30)
    p.add_argument("--headroom-mb", type=int, default=2048)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    if not a.command:
        raise ValueError("Provide a command after --")
    if a.required_mb <= 0 or a.headroom_mb < 2048 or a.memory_limit_gb <= 0:
        p.error("Require positive memory and at least 2048 MiB headroom")
    if a.memory_limit_gb > 30 or a.required_mb > a.memory_limit_gb * 1024:
        p.error("Ceiling must be <= 30 GiB and >= required memory")
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = list(csv.reader(io.StringIO(raw)))
    uuid = select(rows, a.required_mb, a.headroom_mb)
    budget = memory_budget(rows, uuid, a.memory_limit_gb, a.headroom_mb)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = uuid
    env["CHOREO_SHARED_MEMORY_MB"] = str(budget)
    command = a.command[1:] if a.command[0] == "--" else a.command
    print(
        f"Selected {uuid}; physical GPU 7 prohibited; allocation ceiling {budget} MiB; "
        f"policy maximum {a.memory_limit_gb:g} GiB; reserved headroom {a.headroom_mb} MiB",
        flush=True,
    )
    sys.exit(subprocess.call(command, env=env))


if __name__ == "__main__":
    main()
