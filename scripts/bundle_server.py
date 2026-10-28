from pathlib import Path
import tarfile

root = Path.cwd()
target = root / "artifacts/server-bundle.tar.gz"
paths = [
    "src",
    "scripts",
    "configs",
    "tests",
    "pyproject.toml",
    "README.md",
    "data/prepared",
    "data/prepared-mert",
    "data/pilot",
]
with tarfile.open(target, "w:gz", compresslevel=1) as tar:
    for name in paths:
        for p in (root / name).rglob("*") if (root / name).is_dir() else [root / name]:
            if p.is_file() and "__pycache__" not in p.parts:
                tar.add(p, arcname=str(p.relative_to(root)), recursive=False)
print(target, target.stat().st_size)
