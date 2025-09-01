# Claude Code Guidelines

## Code Style Preferences

### Examples and Demonstrations
- **No print statements in examples**: Examples should demonstrate functionality through direct code execution and variable assignment
- **Return values instead of printing**: Functions should return data structures that can be inspected
- **Use docstrings for documentation**: Explain functionality in docstrings rather than print statements
- **Keep examples minimal**: Focus on demonstrating the API usage, not verbose output

### General Coding Principles
- **Minimal sufficient design**: Only include necessary functionality
- **Type hints**: Use modern Python type hints (no `from typing import` when possible)
- **No default parameters**: Require explicit specification with proper error messages
- **Lazy evaluation**: Prefer Polars LazyFrame operations for performance
- **Clear separation of concerns**: Each module should have a single, well-defined purpose

### YAML Configuration
- **Custom expressions**: Support direct Polars expressions in YAML
- **No ambiguous parameters**: Avoid parameters that could be interpreted multiple ways
- **Explicit over implicit**: Require clear specification of intent

### Testing and Validation
- **Lint and typecheck**: Always run `uv run ruff check` and type checking when available
- **Test with real data**: Use the provided sample data functions for testing
- **Verify lazy query plans**: Check optimized query plans for complex operations

## Project Structure
```
plan/
├── CLAUDE.md                 # This file - guidelines for Claude
├── DESIGN_SUMMARY.md         # Architecture and design documentation
├── YAML_PATTERNS.md          # YAML configuration patterns
├── evaluation_schema.yaml    # YAML configuration for metrics
├── yaml_to_pl_lazy.py       # Core translator (minimal, focused)
├── data.py                   # Sample data generation
└── example.py                # Usage examples (no print statements)
```

## Key Implementation Details

### Metric Types
- `across_samples`: Aggregate across all samples
- `within_subject`: Per-subject metrics
- `across_subject`: Two-level aggregation (within then across subjects)
- `within_visit`: Per-visit metrics
- `across_visit`: Two-level aggregation (within then across visits)

### Expression System
- Built-in metrics: `mae`, `mse`, `rmse`, `bias`, etc.
- Selection functions: `:mean`, `:median`, `:std`, `:min`, `:max`
- Custom expressions: Direct Polars expressions in YAML

### Shared Semantics
- `all`: Same value for all models and groups
- `model`: Same value across models, varies by group
- `None`: Unique per model and group combination