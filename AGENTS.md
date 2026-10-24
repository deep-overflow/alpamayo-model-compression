# Repository Guidelines

## Project Structure & Module Organization

- `src/alpamayo_r1/` contains the installable model package: model layers, diffusion, action spaces, geometry, and dataset loading.
- `experiments/head_analysis/` holds pruning and profiling utilities; `experiments/evaluation/` contains benchmarks and quantization; `experiments/recovery/` handles recovery training.
- `recipes/` stores pruning metadata. Read `recipes/README.md` for Alpamayo 1.5 compression; the root README covers the original R1 package.
- `notebooks/` contains examples. Research material lives in `plans/`, `paper/`, and `reports/`; figure assets live in `figures/` and `paper-fig/`. Store experiment outputs in `outputs/` and logs in `logs/`.

## Build, Test, and Development Commands

Run commands from the repository root:

- `uv sync`: install locked dependencies with Python 3.12; FlashAttention requires a compatible CUDA build environment.
- `uv build`: build wheel and source distributions using Hatchling.
- `.venv/bin/python src/alpamayo_r1/test_inference.py`: run end-to-end R1 inference and compute minADE; requires model/dataset access and at least 24 GB GPU memory.
- `.venv/bin/python experiments/llm_pruner/test_lp_formulas.py`: validate pruning formulas on CPU without model weights or calibration data.
- `.venv/bin/python experiments/evaluation/test_quant_lib.py`: check quantization on CPU; requires the hard-coded local Alpamayo 1.5 weight snapshot.
- `ruff check <changed-files>` and `ruff format <changed-files>`: lint and format Python changes when Ruff is installed separately.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` for functions/modules, and `PascalCase` for classes. Preserve license headers and follow nearby code. Ruff's configured line length is 100. Use experiment prefixes such as `run_`, `analyze_`, and `make_`; name plans and reports `YYYY-MM-DD_topic.*`.

## Testing Guidelines

Validation uses standalone `test_*.py` scripts; no centralized framework or coverage threshold is configured. Add a functional test sample for new components. Run relevant checks and document unavailable GPU/data prerequisites. Keep GPU architecture, clip sets, seeds, and model revision consistent across experiment comparisons; record configuration and metrics.

## Commit & Pull Request Guidelines

History mixes topic prefixes and `feat:` messages. Use concise, imperative titles and reference the issue. For upstream contributions, follow `CONTRIBUTING.md`: obtain issue approval before review, use `#<issue> - <title>`, and sign off with `git commit -s`. Keep PRs focused; describe behavior changes, validation results, and experiment provenance.

## Configuration & Artifacts

Keep credentials, datasets, and checkpoints out of Git. Inspect `.venv` and `outputs` symlinks before changing shared resources. Load pruning recipes through `slim_lib.load_slim`, following the import order and cache configuration in `recipes/README.md`.
