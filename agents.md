# Repository Guidelines

## Mandatory Task Checks
- `ruff format` – format code before submitting changes.
- `pyre check` – ensure the static type checker passes.
- `pytest` – run the unit test suite and confirm it passes.
- `rm -r _site; quarto render` – from the `docs/` directory ensure documentation builds cleanly.

## Project Structure & Module Organization
- `src/polars_eval_metrics/` is the production package; use `metric_registry.py`, `metric_define.py`, and `metric_evaluator.py` for expression registration, configs, and lazy execution.
- `tests/` mirrors the module layout with parametrized pytest suites; extend fixtures there instead of cloning sample data.
- `docs/` carries the Quarto site; edit `.qmd` files for user-facing changes and keep imports on `from polars_eval_metrics import ...`.
- `plan/` remains read-only reference material, while `scripts/` and `build/` house supporting utilities.

## Build, Test, and Development Commands
- `uv pip install -e .` installs the editable package; add extras like `".[dev]"` when linting or testing locally.
- `uv run ruff check src tests` drives the canonical style check; run after major refactors to catch ASCII or typing issues.
- `uv run pytest` executes the 92-test suite; combine with `-k pattern` for focused debugging.
- `uv run quarto render` rebuilds documentation from the repository root when Quarto is available.

## Coding Style & Naming Conventions
- Target Python 3.11 syntax with four-space indentation and explicit type hints using `|` unions and `dict[str, T]` aliases.
- Keep the codebase ASCII-only and rely on Ruff for linting; avoid decorative comments and prefer descriptive names over abbreviations.
- Compose Polars logic with vectorized expressions (`pl.col`, `cs.numeric()`) and helper composition-never row-by-row loops or in-place mutation.

## Testing Guidelines
- Pytest powers the suite; name files and functions `test_*` and place new coverage under `tests/`.
- Expand parametrized cases to protect ARD schema, selector usage, and subgroup behavior whenever metrics evolve.
- Run `uv run pytest` (and document the result) before every commit or pull request.

## Commit & Pull Request Guidelines
- Follow the local history of short, present-tense summaries (e.g., `refactor using ARD class`) and stage only intentional changes.
- Keep the main branch clean-no ad-hoc branches or uncommitted work when pushing.
- PRs should link related issues, summarise module impacts, list tests or builds run, and call out doc updates.

## Documentation & Quarto
- Update `docs/` alongside API or formatting changes, then render with `uv run quarto render`.
- If Quarto cannot run locally, note the limitation in the PR so reviewers can rebuild downstream.
