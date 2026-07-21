# Repository Guidelines

## Project Structure & Module Organization

This repository contains the MLB pregame dataset, modeling, evaluation, and training pipeline under `src/mlb_predictor/`, with corresponding regression coverage under `tests/`. Keep Python packages under `src/mlb_predictor/` and mirror modules in `tests/` (for example, `src/mlb_predictor/features.py` and `tests/test_features.py`). Store generated API caches, CSV, Parquet, manifests, and quality reports under `data/`; this directory is generated and must remain untracked. Put the ChatGPT Sites application in `sites-app/`, with static assets in `sites-app/public/` and Sites metadata in `.openai/hosting.json` once created.

## Build, Test, and Development Commands

Use Python 3.11 or newer. From the repository root:

```powershell
python -m pip install -e .
python -m pytest
python -m compileall -q src tests
$env:PYTHONPATH = "src"
python -m mlb_predictor build --start-date 2025-03-27 --end-date 2025-04-05 --output-dir data\sample
```

Run `python -m mlb_predictor validate --games <games.parquet> --features <features.parquet>` before accepting generated data. When `sites-app/` is added, document and run its test, type-check, lint, and Sites build scripts before deployment.

## Coding Style & Naming Conventions

Use four-space indentation, type hints on public Python APIs, `snake_case` for modules/functions/variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Keep collectors, normalization, feature generation, quality checks, and persistence in separate modules. No formatter or linter is configured yet; avoid unrelated formatting changes and follow neighboring code.

## Testing Guidelines

Pytest is the test runner. Name files `test_<module>.py` and tests `test_<behavior>`. Add regression tests for every leakage boundary, date rule, exclusion reason, and failed-run preservation path. Tests must not require live network access; use cached fixtures. A change is ready only when pytest, `compileall`, and applicable data-quality gates pass.

## Commit & Pull Request Guidelines

The repository has no commit history, so no established convention exists. Use concise imperative subjects with prefixes such as `feat:`, `fix:`, `test:`, or `docs:`. Pull requests should explain scope, leakage implications, commands run, and data/model versions. Include screenshots for Sites UI changes and never claim production visual QA unless it was actually performed.

## Security & Deployment

Never commit API keys, authentication tokens, raw credentials, or generated datasets. Keep secrets in environment variables. Treat model artifacts as versioned snapshots with feature schema, training cutoff, calibration status, and checksums; preserve the exact source revision used for each Sites deployment.
