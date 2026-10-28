# Public release validation — 2026-10-28

Environment: Windows, Python 3.10.9, NumPy 1.26.4, PyTorch 2.7.0+cu128; frontend Vite 6.4.4.

- Python: `python -m pytest -q -p no:cacheprovider` — 42 passed. The checkpoint/resume test ran in an explicitly writable temporary directory.
- TypeScript: `npm test` — 7 passed, including Python/TypeScript arbitrary-seek parity.
- Production frontend: `npm run build` — passed.
- Edge browser: default authored fixture, evolving motion, identical repeated seeks, math/action fixture loading, object editing and JSON export, 390px responsive layout, zero page errors — passed.
- README workbench image: captured from the actual local preview. Gallery/GIF generated using `scripts/make_public_demo.py`.
- Publication boundary: source/tests/configuration/docs and authored fixtures only; no datasets, checkpoints, imported audio/charts, credential files or local machine-specific paths.

These are engineering/reproducibility checks. They do not establish trained choreography quality, aesthetic acceptance, commercial data rights or Linux/CUDA deployment coverage.
