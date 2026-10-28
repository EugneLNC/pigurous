"""Source + built preview only. Never include downloaded data or model weights."""
from pathlib import Path
import zipfile

roots = [
    "src",
    "scripts",
    "configs",
    "schemas",
    "tests",
    "docs",
    "preview/src",
    "preview/public",
    "preview/dist",
]
files = [
    "README.md",
    "pyproject.toml",
    ".gitignore",
    "preview/package.json",
    "preview/package-lock.json",
    "preview/tsconfig.json",
    "preview/index.html",
]
target = Path("artifacts/music-choreographer-code.zip")
target.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
    for name in roots:
        for file in Path(name).rglob("*"):
            if (
                not file.is_file()
                or "__pycache__" in file.parts
                or any(p.endswith(".egg-info") for p in file.parts)
            ):
                continue
            if file.suffix.lower() in (
                ".pyc",
                ".ogg",
                ".mp3",
                ".wav",
                ".mp4",
                ".webm",
                ".pt",
                ".npz",
            ):
                continue
            archive.write(file, str(file))
    for name in files:
        archive.write(name, name)
print(target, target.stat().st_size)
