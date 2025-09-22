# AGENT WORKING AGREEMENT

These instructions apply to the entire repository. Follow them so future tasks can be completed quickly and safely.

## 0. Always sync with the canonical docs
- Read `CLAUDE.md` before starting substantive work. It describes the package layout, major features, and recent fixes. Re-read it whenever unsure about architecture or directory responsibilities.

## 1. Environment setup
- Use `uv` for Python workflows (already configured through `pyproject.toml`).
- When dependencies are missing, run `uv pip install -e .` from the project root to install the package in editable mode.
- Keep the repo ASCII-only (recent cleanup removed non-ASCII characters). Avoid reintroducing emojis or special symbols into source files and documentation.

## 2. Code changes
- Production code lives in `src/polars_eval_metrics/`. Respect the separation of concerns described in `CLAUDE.md`:
  - `metric_registry.py` owns expression registration.
  - `metric_define.py` holds definition metadata.
  - `metric_evaluator.py` executes evaluations and formatting.
- Do **not** edit the `plan/` directory; it is reference documentation only.
- Maintain type hints and mirror existing Polars expression patterns. Prefer composing helper functions instead of duplicating logic.

## 3. Documentation
- Update the Quarto docs under `docs/` when adding user-facing functionality.
- Ensure example code in docs imports from the public package surface (`from polars_eval_metrics import ...`).
- Rebuild docs locally with `uv run quarto render` when touching `.qmd` files (skip if Quarto is unavailable, but note the limitation in the PR/testing summary).

## 4. Testing and QA
- Run the full test suite with `uv run pytest` from the repository root. This exercises both unit and integration tests (currently 92 total).
- If tests fail due to missing fixtures or data, inspect the corresponding files in `tests/`—they provide good usage examples.
- Include additional targeted tests when fixing bugs or adding features.

## 5. Pull requests & commits
- Follow the Git instructions from the system prompt (no new branches, clean worktree, run required checks).
- Summaries should mention affected modules and highlight any updates to docs/tests.

Keeping these practices in mind will make it much easier to extend or debug the evaluation framework.
