# OptLab-DF

Optimizer comparison and hybrid optimizer design for multimodal deepfake detection
(Computational Mathematics for AI & ML, Nirma University).

**Status:** Phase 0 (setup) and Phase 1 (toy benchmarks). No deepfake results yet.
See `docs/PRD.md` for the research question and `docs/TRD.md` for the technical spec.

```bash
make setup   # uv sync (Python 3.11)
make test    # pytest on CPU
make toy     # toy benchmark suite -> results/toy_summary.md, results/toy/
```

`make toy` took 349 s on an Apple-silicon laptop CPU (measured 2026-10-12). Toy results:
[results/toy_summary.md](results/toy_summary.md), generated from the results database.

Feature extraction needs the optional extras (`uv sync --extra extract`) and local dataset
copies; no dataset results exist yet.
