# Repository Guidelines

## Project Structure & Module Organization

- `src/alpamayo_r1/` contains the installable model package: model layers, diffusion, action spaces, geometry, and dataset loading.
- `experiments/head_analysis/` holds pruning, profiling, and evaluation utilities; `experiments/evaluation/` handles benchmarks and quantization; `experiments/recovery/` contains recovery training.
- `recipes/` stores reproducible pruning metadata. Read `recipes/README.md` for the Alpamayo 1.5 compression workflow; the root README describes the original R1 package.
- `notebooks/` contains examples; `plans/`, `paper/`, and `reports/` hold research plans, drafts, and results. Generated artifacts belong in `outputs/` and `logs/`.

## Build, Test, and Development Commands

Run commands from the repository root:

- `uv sync`: install locked dependencies with Python 3.12; FlashAttention requires a compatible CUDA build environment.
- `uv build`: build the package's wheel and source distribution with Hatchling.
- `.venv/bin/python src/alpamayo_r1/test_inference.py`: run the R1 end-to-end inference example and compute minADE; requires model/dataset access and at least 24 GB GPU memory.
- `.venv/bin/python experiments/evaluation/test_quant_lib.py`: run CPU quantization checks; requires the hard-coded local Alpamayo 1.5 weight snapshot.
- `ruff check <changed-files>` and `ruff format <changed-files>`: lint and format Python changes when Ruff is installed separately.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` for functions/modules, and `PascalCase` for classes. Follow nearby code and preserve license headers. Ruff's configured line length is 100. Use descriptive experiment prefixes such as `run_`, `analyze_`, and `make_`; name plans and reports `YYYY-MM-DD_topic.*`.

## Testing Guidelines

Validation uses standalone scripts; no centralized test framework or coverage threshold is configured. Name checks `test_*.py` and add a functional test sample for new components. Run relevant checks before submitting and document unavailable GPU/data prerequisites. For experiment comparisons, keep GPU architecture, clip sets, seeds, and model revision consistent; record configuration and metrics.

## Commit & Pull Request Guidelines

Recent commits use topic prefixes such as `hard100:` followed by a concise description. Use imperative titles and reference the issue. Follow `CONTRIBUTING.md` for upstream contributions: approved issue before review, `#<issue> - <title>` commit format, and sign-off with `git commit -s`. Keep PRs focused; describe behavior changes, validation commands/results, and experiment provenance.

## Configuration & Artifacts

Keep credentials, datasets, and checkpoints out of Git. Inspect `.venv` and `outputs` symlinks before changing shared resources. Use `slim_lib.load_slim` to load pruning recipes, following the import order and cache configuration in `recipes/README.md`.
