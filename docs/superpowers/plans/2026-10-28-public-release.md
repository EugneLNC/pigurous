# Pigurous Public Release Implementation Plan

> **For agentic workers:** Execute inline in this isolated new repository. Steps use checkbox syntax for tracking.

**Goal:** Publish a small, reproducible public code release with honest visual examples, then reclaim rebuildable local datasets and intermediates.

**Architecture:** Preserve Python audio/score/model/training modules and the TypeScript deterministic player. Ship authored, silent renderer fixtures instead of downloaded music/charts. Keep unique checkpoints, human preferences and provenance privately; regenerated feature tensors and build outputs stay outside Git.

**Tech Stack:** Python 3.10+, NumPy/SciPy/Pillow/PyTorch, TypeScript, Three.js, Vite, GitHub CLI.

---

### Task 1: Isolate and audit source
- [x] Copy src/, tests/, scripts/, configs/, schemas/, docs/ and preview source/configuration into this directory, excluding bytecode and all datasets, checkpoints, media, node_modules and build output.
- [x] Scan copied text for credential-shaped tokens and replace machine-specific paths with environment-relative equivalents. Preserve historical validation limitations.

### Task 2: Create public examples and documentation
- [x] Generate the existing hand-authored Orbit fixture with `PYTHONPATH=src python scripts/make_demo.py`.
- [x] Add a reproducible, silent mathematical action gallery and animated GIF labelled as authored renderer examples rather than model output.
- [x] Make the preview default to bundled authored examples, preserving file import/export and experimental pages.
- [x] Write README.md with install/preview/test commands, pipeline diagram, screenshots, honest research limits and asset provenance. Add LICENSE, THIRD_PARTY_NOTICES.md and a strict .gitignore.

### Task 3: Validate and publish
- [x] Run the Python test suite with the existing project environment; run preview tests and production build.
- [x] Open the preview through a local server, verify deterministic seeking and capture its actual UI.
- [ ] Initialize a new repository, commit only the audited core, create EugneLNC/pigurous as public and push the release commit.
- [ ] Create a GitHub Release with a source archive and manifest. Confirm remote HEAD equals local HEAD and verify the release asset.

### Task 4: Reclaim local experiment storage after publication
- [ ] Retain source/configuration/tests/docs, selected best checkpoints, human review annotations, source manifests and metrics.
- [ ] List exact generated feature/data/model-cache and redundant build/packaging files; protect unique checkpoints and third-party raw material until retention is established.
- [ ] Delete only listed rebuildable files after the public release is confirmed, then remeasure C/E free space and save a cleanup receipt.
