# CLAUDE.md - AI Development Guidelines for `polars-eval-metrics`

This document provides focused guidance for AI assistants working on the `polars-eval-metrics` package.

## Project Overview

**Package**: High-performance model evaluation framework using Polars lazy evaluation
**Core Pattern**: Definition -> Evaluation -> Formatting pipeline
**Key Features**: Vectorized operations, hierarchical aggregations, pivot tables, YAML configuration

## Architecture Principles

### 1. **Polars-First Design**
- Use LazyFrame for all data operations
- Leverage vectorized operations over loops
- Utilize `cs.numeric()`, `cs.by_dtype()` selectors
- Apply schema harmonization for concatenation

### 2. **Clean Separation of Concerns**
```
MetricRegistry -> MetricDefine -> MetricEvaluator -> TableFormatter
   (expressions)    (config)       (computation)      (presentation)
```

### 3. **Validation Strategy**
- Validate once at initialization, not in internal methods
- Use Pydantic for input validation
- Centralize validation logic in dedicated methods
- Fail fast with clear error messages

## Core Components

### MetricRegistry (`metric_registry.py`)
- **Purpose**: Unified expression storage
- **Pattern**: Register once, use everywhere
- **Key Methods**: `register_error()`, `register_metric()`, `get_*()`, `list_*()`

### MetricDefine (`metric_define.py`)
- **Purpose**: Metric configuration and compilation
- **Pattern**: Immutable configuration objects
- **Key Features**: Type/scope validation, expression compilation, Pydantic validation

### MetricEvaluator (`metric_evaluator.py`)
- **Purpose**: Vectorized evaluation engine
- **Pattern**: Cache-optimized lazy evaluation
- **Key Features**: Subgroup vectorization, pivot operations, enum preservation

## Development Guidelines

### Code Style
- **No comments** unless explicitly requested
- Use type hints consistently (`|` syntax, not `Union`)
- Prefer `dict[str, str]` over `Dict[str, str]`
- **ASCII characters only** (enforced by pre-commit hooks and ruff)
- Follow existing naming conventions

### ASCII Compliance (MANDATORY)
All code, documentation, and configuration files MUST use ASCII characters only:
- No Unicode symbols, emojis, or special characters
- No smart quotes, dashes, or accented characters
- Enforced automatically by:
  - Ruff rules: RUF001, RUF002, RUF003
  - Pre-commit hook: `check-ascii-only`
  - CI/CD pipeline validation

### Polars Optimization Patterns
```python
# Good: Vectorized operations
df.unpivot(index=id_vars, on=subgroup_cols, variable_name="subgroup_name")

# Bad: Nested loops
for subgroup in subgroups:
    for value in values:
        # process individually

# Good: Schema harmonization
harmonized_results = self._harmonize_result_schemas(results)
return pl.concat(harmonized_results, how="diagonal")

# Good: Selectors
df.select(cs.numeric()).fmt_number(decimals=2)
```

### Testing Strategy
- Test public APIs, not internal implementation
- Use parametrized tests for multiple scenarios
- Mock external dependencies
- Validate both happy path and edge cases

### Error Handling
```python
# Good: Early validation with clear messages
if missing_columns:
    raise ValueError(f"Required columns not found: {missing_columns}")

# Good: Type-specific error handling
try:
    schema = df.collect_schema()
except Exception:
    schema = df.limit(1).collect().schema
```

## Common Patterns

### Input Normalization
```python
@staticmethod
def _process_estimates(estimates: str | list[str] | dict[str, str] | None) -> dict[str, str]:
    if isinstance(estimates, str):
        return {estimates: estimates}
    elif isinstance(estimates, dict):
        return estimates
    elif isinstance(estimates, list):
        return {est: est for est in (estimates or [])}
    return {}
```

### Conditional Logic for Optional Features
```python
# Check data structure, not configuration
if "subgroup_name" in df.columns:
    # Handle subgroup logic

# Build columns dynamically
index_cols = []
if "subgroup_name" in df.columns:
    index_cols.extend(["subgroup_name", "subgroup_value"])
if self.group_by:
    index_cols.extend(list(self.group_by.keys()))
```

### Caching Pattern
```python
def _get_cached_evaluation(self, metrics=None, estimates=None) -> pl.DataFrame:
    cache_key = self._get_cache_key(metrics, estimates)
    if cache_key not in self._evaluation_cache:
        result = self.evaluate(metrics=metrics, estimates=estimates, collect=True)
        self._evaluation_cache[cache_key] = result
    return self._evaluation_cache[cache_key]
```

## Anti-Patterns to Avoid

### Don't
- Create dummy columns for optional features
- Mix validation with business logic
- Use nested loops where vectorization is possible
- Hardcode column names or magic strings
- Modify data in-place (use immutable operations)
- Use complex inheritance hierarchies

### Do
- Use conditional logic for optional features
- Validate inputs once at boundaries
- Leverage Polars vectorized operations
- Use configuration-driven column handling
- Return new objects from transformations
- Prefer composition over inheritance

## Performance Considerations

1. **Lazy Evaluation**: Keep operations as LazyFrame until collection needed
2. **Schema Harmonization**: Essential for concat operations with different column sets
3. **Vectorized Subgroups**: Use `unpivot()` instead of loops for marginal analysis
4. **Caching**: Cache expensive evaluations with proper key generation
5. **Enum Preservation**: Maintain original data types through transformations

## Integration Points

### External Dependencies
- **Polars**: Core data processing (require exact version matching)
- **Pydantic**: Input validation and type safety
- **Great Tables**: HTML table formatting
- **Pytest**: Testing framework

### File Structure
```
src/polars_eval_metrics/
|-- __init__.py          # Public API exports
|-- metric_registry.py   # Expression storage
|-- metric_define.py     # Configuration objects
|-- metric_evaluator.py  # Computation engine
|-- metric_helpers.py    # Convenience functions
|-- table_formatter.py   # Output formatting
```

## Future Development Notes

- Keep vectorized patterns when adding new features
- Maintain separation between definition and evaluation
- Consider performance impact of new aggregation types
- Preserve enum ordering in all transformations
- Add new metrics to registry, don't hardcode
- Test pivot operations thoroughly (complex column naming)

---

*Focus: Code quality, performance, maintainability*
*Updated: 2025-01-18*