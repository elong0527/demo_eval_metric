# CLAUDE.md - Development Guidelines for `polars-eval-metrics` Package

This document provides comprehensive guidance for developing and maintaining the `polars-eval-metrics` package, a high-performance model evaluation framework using Polars lazy evaluation.

## Package Overview

**Package Name**: `polars-eval-metrics`  
**Purpose**: A flexible, high-performance framework for evaluating model predictions with support for hierarchical aggregations, custom metrics, and YAML-based configuration.

### Key Features
- 🚀 **Lazy Evaluation**: Leverages Polars LazyFrame for optimal query planning
- 📊 **Flexible Metrics**: Built-in and custom metric definitions
- 🔧 **YAML Configuration**: Complete evaluation setup via YAML
- 🎯 **Type Safety**: Pydantic validation throughout
- 🔄 **Hierarchical Aggregation**: Support for subject/visit level metrics

## Project Structure

```
demo_eval_metric/
├── src/
│   └── polars_eval_metrics/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── metric_data.py      # Pure data models (no Polars)
│       │   ├── metric_factory.py    # Factory for creating metrics
│       │   └── metric_compiler.py   # Polars expression compilation
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metric_evaluator.py  # Main evaluation engine
│       │   └── evaluation_config.py # Configuration management
│       ├── utils/
│       │   ├── __init__.py
│       │   └── data_generator.py    # Sample data generation
│       └── py.typed                 # PEP 561 type hint marker
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── test_metric_data.py
│   │   ├── test_metric_factory.py
│   │   ├── test_metric_compiler.py
│   │   └── test_metric_evaluator.py
│   ├── integration/
│   │   ├── test_yaml_workflow.py
│   │   └── test_end_to_end.py
│   └── fixtures/
│       ├── sample_configs.yaml
│       └── sample_data.py
├── examples/
│   ├── basic_usage.py
│   ├── custom_metrics.py
│   ├── yaml_configuration.py
│   └── configs/
│       └── example_evaluation.yaml
├── docs/
│   ├── getting_started.md
│   ├── api_reference.md
│   ├── configuration_guide.md
│   └── architecture.md
├── pyproject.toml
├── README.md
├── LICENSE
└── CHANGELOG.md
```

## Development Commands

### Environment Setup
```bash
# Create virtual environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package in development mode
uv pip install -e ".[dev]"

# Install all dependencies
uv pip install -e ".[dev,test,docs]"
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/polars_eval_metrics --cov-report=html

# Run specific test file
pytest tests/unit/test_metric_data.py

# Run integration tests only
pytest tests/integration/

# Run with verbose output
pytest -v
```

### Code Quality
```bash
# Format code with black
black src/ tests/

# Sort imports
isort src/ tests/

# Run linting
ruff check src/ tests/

# Type checking
mypy src/polars_eval_metrics

# Run all checks (format, lint, type check)
make quality  # If Makefile is set up
```

### Documentation
```bash
# Build documentation
mkdocs build

# Serve documentation locally
mkdocs serve

# Generate API documentation
pdoc --html src/polars_eval_metrics --output-dir docs/api
```

## Architecture Guidelines

### 1. Separation of Concerns

The package follows a clean architecture with clear separation:

- **`core/`**: Pure data models and business logic (no framework dependencies)
  - `metric_data.py`: Pydantic models for metric definitions
  - `metric_factory.py`: Factory pattern for metric creation
  - `metric_compiler.py`: Polars-specific expression compilation

- **`evaluation/`**: Evaluation engine and configuration
  - `metric_evaluator.py`: Main evaluation orchestration
  - `evaluation_config.py`: YAML configuration management

- **`utils/`**: Helper utilities
  - `data_generator.py`: Test data generation

### 2. Design Principles

#### Single Responsibility
Each module has one clear purpose:
- `MetricData`: Configuration and validation only
- `MetricCompiler`: Expression compilation only
- `MetricEvaluator`: Pipeline orchestration only
- `MetricFactory`: Object creation only

#### Dependency Inversion
- High-level modules don't depend on low-level details
- `MetricData` doesn't know about Polars
- `MetricEvaluator` uses `MetricCompiler` interface

#### Open/Closed Principle
- Open for extension (new metrics, expressions)
- Closed for modification (stable interfaces)

### 3. Type Safety

Use type hints throughout:
```python
from typing import Any
import polars as pl

def evaluate_metric(
    df: pl.DataFrame | pl.LazyFrame,
    metric: MetricData,
    estimate: str
) -> pl.LazyFrame:
    ...
```

### 4. Error Handling

Provide clear, actionable error messages:
```python
if not metric_name:
    raise ValueError(
        f"Metric '{metric_name}' not found. "
        f"Available metrics: {', '.join(available_metrics)}"
    )
```

## API Design Guidelines

### 1. Configuration-First API

Support both programmatic and YAML configuration:
```python
# YAML configuration
evaluator = MetricEvaluator.from_config(df, "config.yaml")

# Programmatic with overrides
evaluator = MetricEvaluator.from_config(
    df, config,
    group_by=["treatment", "gender"]  # Override
)
```

### 2. Fluent Interfaces

Enable method chaining where appropriate:
```python
factory = (MetricFactory()
    .from_yaml(config)
    .add_metric(custom_metric)
    .validate())
```

### 3. Sensible Defaults

Provide defaults but allow overrides:
```python
class MetricEvaluator:
    def __init__(self,
                 ground_truth: str = "actual",  # Sensible default
                 compiler: MetricCompiler = None):  # Auto-create if None
        self.compiler = compiler or MetricCompiler()
```

## Testing Strategy

### 1. Unit Tests
Test each component in isolation:
- Metric validation
- Expression compilation
- Factory creation
- Configuration parsing

### 2. Integration Tests
Test component interactions:
- YAML to evaluation workflow
- Multiple metric evaluation
- Group/filter combinations

### 3. Property-Based Testing
Use hypothesis for edge cases:
```python
from hypothesis import given, strategies as st

@given(st.lists(st.floats()))
def test_metric_calculation(values):
    # Test metric properties
    ...
```

### 4. Performance Testing
Benchmark critical paths:
```python
import pytest

@pytest.mark.benchmark
def test_large_dataset_performance(benchmark):
    result = benchmark(evaluate_metrics, large_df)
    assert result.shape[0] > 0
```

## Performance Considerations

### 1. Lazy Evaluation
Always use LazyFrames for large datasets:
```python
# Good
df_lazy = df.lazy()
result = df_lazy.filter(...).group_by(...).agg(...).collect()

# Avoid
result = df.filter(...).group_by(...).agg(...)
```

### 2. Expression Reuse
Compile expressions once:
```python
# Good
compiler = MetricCompiler()
expr = compiler.compile_expression(metric)
for df in dataframes:
    result = df.select(expr)

# Avoid
for df in dataframes:
    expr = compiler.compile_expression(metric)  # Recompiling
    result = df.select(expr)
```

### 3. Memory Management
Use streaming for large results:
```python
# For large datasets
results = evaluator.evaluate_streaming()
for chunk in results:
    process(chunk)
```

## YAML Configuration Schema

### Complete Configuration Example
```yaml
# Data configuration
data:
  population: adsv
  observation: adlb

# Column mappings
columns:
  subject_id: usubjid
  visit_id: visitid
  ground_truth: aval
  estimates: [model1, model2]
  group: [treatment, region]
  subgroup: [age_group, sex]

# Filters
filter:
  - "pl.col('visit_id') <= 3"
  - "pl.col('treatment').is_not_null()"

# Metrics
metrics:
  - name: mae
    label: Mean Absolute Error
    type: across_samples
    
  - name: rmse_per_subject
    label: RMSE per Subject
    type: within_subject
    
  - name: custom_weighted_mae
    label: Weighted MAE
    type: across_samples
    agg:
      expr: 
        - "pl.col('absolute_error').mean().alias('value')"
        - "pl.col('weight').sum().alias('total_weight')"
    select:
      expr: "pl.col('value') * pl.col('total_weight')"
```

## Release Process

### 1. Version Management
Follow semantic versioning (MAJOR.MINOR.PATCH):
- MAJOR: Breaking API changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

### 2. Release Checklist
- [ ] Update CHANGELOG.md
- [ ] Run full test suite
- [ ] Update version in pyproject.toml
- [ ] Create git tag
- [ ] Build distribution
- [ ] Upload to PyPI

### 3. Publishing
```bash
# Build package
python -m build

# Test on TestPyPI first
twine upload --repository testpypi dist/*

# Publish to PyPI
twine upload dist/*
```

## Migration from `plan/` to `src/`

### File Mapping
```
plan/metric_data.py      → src/polars_eval_metrics/core/metric_data.py
plan/metric_factory.py   → src/polars_eval_metrics/core/metric_factory.py
plan/metric_compiler.py  → src/polars_eval_metrics/core/metric_compiler.py
plan/metric_evaluator.py → src/polars_eval_metrics/evaluation/metric_evaluator.py
plan/evaluation_config.py → src/polars_eval_metrics/evaluation/evaluation_config.py
plan/data.py             → src/polars_eval_metrics/utils/data_generator.py
```

### Import Updates
```python
# Old (plan/)
from metric_data import MetricData
from metric_evaluator import MetricEvaluator

# New (src/)
from polars_eval_metrics.core import MetricData
from polars_eval_metrics.evaluation import MetricEvaluator
```

## Contributing Guidelines

### 1. Code Style
- Follow PEP 8
- Use type hints
- Write docstrings (Google style)
- Keep functions focused and small

### 2. Testing Requirements
- Write tests for new features
- Maintain >90% code coverage
- Include integration tests for workflows

### 3. Documentation
- Update docstrings
- Add examples for new features
- Update CHANGELOG.md

### 4. Pull Request Process
1. Create feature branch
2. Write tests first (TDD)
3. Implement feature
4. Run quality checks
5. Update documentation
6. Submit PR with clear description

## Common Issues and Solutions

### Issue: Import errors after migration
**Solution**: Update all imports to use package structure:
```python
from polars_eval_metrics.core import MetricData, MetricFactory
from polars_eval_metrics.evaluation import MetricEvaluator
```

### Issue: YAML configuration not loading
**Solution**: Ensure YAML follows the schema and check for indentation:
```yaml
metrics:  # Note: list of metrics
  - name: mae
    type: across_samples
```

### Issue: Performance degradation
**Solution**: Ensure using LazyFrames and check query plan:
```python
lazy_df = df.lazy()
print(lazy_df.explain())  # Check optimization
```

## Support and Resources

- **Documentation**: [docs/](./docs/)
- **Examples**: [examples/](./examples/)
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **API Reference**: [API Docs](./docs/api_reference.md)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Last updated: 2024*  
*Maintainer: Engineering Team*