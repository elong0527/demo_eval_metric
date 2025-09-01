# CLAUDE.md - Development Guidelines for `polars-eval-metrics` Package

This document provides comprehensive guidance for developing and maintaining the `polars-eval-metrics` package, a high-performance model evaluation framework using Polars lazy evaluation.

## Project Overview

**Package Name**: `polars-eval-metrics`  
**Purpose**: A flexible, high-performance framework for evaluating model predictions with support for hierarchical aggregations, custom metrics, and YAML-based configuration.

### Key Features
- 🚀 **Lazy Evaluation**: Leverages Polars LazyFrame for optimal query planning
- 📊 **Flexible Metrics**: Built-in and custom metric definitions
- 🔧 **YAML Configuration**: Complete evaluation setup via YAML
- 🎯 **Type Safety**: Pydantic validation throughout
- 🔄 **Hierarchical Aggregation**: Support for subject/visit level metrics
- 📈 **LazyFrame Chain Visualization**: See exact Polars operations via `pl_expr()`

## Project Structure

```
demo_eval_metric/
├── plan/                        # [FINALIZED - DO NOT MODIFY]
│   ├── metric_data.py          # Reference implementation
│   ├── metric_compiler.py      # Expression compilation logic
│   ├── metric_factory.py       # Factory pattern implementation
│   ├── metric_evaluator.py     # Evaluation engine
│   ├── evaluation_config.py    # Configuration management
│   └── *.qmd                   # Development documentation
│
├── src/                         # [PRODUCTION CODE]
│   └── polars_eval_metrics/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── metric_data.py      # MetricData with pl_expr() visualization
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
│
├── docs/                        # [DOCUMENTATION WEBSITE SOURCE]
│   ├── _quarto.yml             # Quarto configuration for website
│   ├── index.qmd               # Homepage
│   ├── getting_started.qmd    # Getting started guide
│   ├── metric_examples.qmd    # Interactive metric examples
│   └── _site/                 # [GENERATED - git ignored]
│
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── test_metric_data.py
│   │   ├── test_metric_factory.py
│   │   ├── test_metric_compiler.py
│   │   └── test_metric_evaluator.py
│   └── integration/
│       ├── test_yaml_workflow.py
│       └── test_end_to_end.py
│
├── .github/
│   └── workflows/
│       └── docs.yml            # GitHub Actions for documentation website
│
├── pyproject.toml
├── README.md
├── LICENSE
├── CHANGELOG.md
└── CLAUDE.md                   # This file
```

## Directory Purposes

### 📁 `plan/` - Finalized Reference Implementation
**STATUS: COMPLETE - DO NOT MODIFY**

This directory contains the finalized reference implementation developed during the planning phase. It serves as:
- Reference architecture for the src/ implementation
- Development documentation and examples
- Proof of concept for key features

**Important**: This directory is frozen and should not be modified. All active development happens in `src/`.

### 📁 `src/` - Production Code
**STATUS: ACTIVE DEVELOPMENT**

The production implementation of the package. Key features:
- Full implementation based on plan/ reference
- Includes `pl_expr()` method for LazyFrame chain visualization
- Type-safe with Pydantic models
- Optimized for performance with Polars lazy evaluation

### 📁 `docs/` - Documentation Website
**STATUS: ACTIVE - AUTO-DEPLOYED**

Quarto-based documentation that:
- Imports from `src/` for live examples
- Automatically deployed via GitHub Actions
- Provides interactive examples with LazyFrame chain visualization
- Shows real output including the `pl_expr()` chains

## Key Implementation Details

### MetricData Class Features

The `MetricData` class in `src/polars_eval_metrics/core/metric_data.py` includes:

```python
class MetricData(BaseModel):
    name: str
    label: str | None = None  # Auto-generated from name if not provided
    type: MetricType
    shared_by: SharedType | None = None
    agg_expr: list[str] | None = None
    select_expr: str | None = None
    
    def pl_expr(self) -> str:
        """Returns the Polars LazyFrame chain for this metric"""
        # Shows exact operations like:
        # (
        #   pl.LazyFrame
        #   .group_by('subject_id')
        #   .agg(col("absolute_error").mean().alias("value"))
        #   .select(col("value").mean())
        # )
```

### Metric Types and Their LazyFrame Chains

| MetricType | LazyFrame Chain Pattern |
|------------|-------------------------|
| ACROSS_SAMPLES | `.select(expr)` |
| WITHIN_SUBJECT | `.group_by('subject_id').agg(expr)` |
| ACROSS_SUBJECT | `.group_by('subject_id').agg(expr).select(selector)` |
| WITHIN_VISIT | `.group_by(['subject_id', 'visit_id']).agg(expr)` |
| ACROSS_VISIT | `.group_by(['subject_id', 'visit_id']).agg(expr).select(selector)` |

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
```

### Documentation
```bash
# Build documentation locally
cd docs
quarto render

# Preview documentation
quarto preview

# The website is automatically deployed via GitHub Actions on push to main
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
```

## GitHub Actions Documentation Workflow

The `.github/workflows/docs.yml` file automatically:
1. Builds the Quarto documentation from `docs/`
2. Deploys to GitHub Pages
3. Runs on every push to main branch

Example workflow:
```yaml
name: Deploy Documentation

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -e .
          pip install quarto
      
      - name: Build documentation
        run: |
          cd docs
          quarto render
      
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs/_site
```

## API Usage Examples

### Basic Usage
```python
from polars_eval_metrics.core import MetricData, MetricType

# Create a metric
mae = MetricData(name="mae", type=MetricType.ACROSS_SAMPLES)

# View the metric details and LazyFrame chain
print(mae)
# Output:
# MetricData(name='mae', type=across_samples)
#   Label: 'mae'
#   Shared by: none
#   Aggregation expressions: none
#   Selection expression:
#       - [mae] col("absolute_error").mean().alias("value")
# 
# (
#   pl.LazyFrame
#   .select(col("absolute_error").mean().alias("value"))
# )

# Get just the LazyFrame chain
chain = mae.pl_expr()
```

### Hierarchical Metrics
```python
# Two-level aggregation
mae_mean = MetricData(
    name="mae:mean", 
    type=MetricType.ACROSS_SUBJECT
)

print(mae_mean.pl_expr())
# Output:
# (
#   pl.LazyFrame
#   .group_by('subject_id')
#   .agg(col("absolute_error").mean().alias("value"))
#   .select(col("value").mean())
# )
```

## Important Development Notes

### 1. DO NOT Modify `plan/` Directory
The `plan/` directory contains the finalized reference implementation. Any changes should be made in `src/` only.

### 2. Documentation Must Import from `src/`
All documentation in `docs/` should import from the production code:
```python
import sys
sys.path.append('../src')
from polars_eval_metrics.core.metric_data import MetricData
```

### 3. LazyFrame Chain Visualization
The `pl_expr()` method is a key feature that shows users exactly how their metrics will be computed. Ensure this is prominently featured in documentation.

### 4. Type Safety
Always use type hints and Pydantic validation for configuration and data models.

### 5. Performance First
Leverage Polars lazy evaluation wherever possible. Use `.lazy()` and `.collect()` appropriately.

## Testing Strategy

### Unit Tests
Test each component in isolation:
- Metric validation
- Expression compilation  
- Factory creation
- Configuration parsing

### Integration Tests
Test component interactions:
- YAML to evaluation workflow
- Multiple metric evaluation
- Group/filter combinations
- LazyFrame chain execution

### Documentation Tests
Ensure all code examples in documentation work:
- Quarto documents render without errors
- Examples produce expected output
- LazyFrame chains are displayed correctly

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
- Add examples to docs/
- Update CHANGELOG.md

### 4. Pull Request Process
1. Create feature branch from main
2. Make changes in `src/` (never in `plan/`)
3. Write/update tests
4. Update documentation
5. Run quality checks
6. Submit PR with clear description

## Common Issues and Solutions

### Issue: Import errors in documentation
**Solution**: Ensure proper path setup:
```python
import sys
sys.path.append('../src')
```

### Issue: LazyFrame chain not displaying
**Solution**: Check that `pl_expr()` method is properly implemented and MetricCompiler handles the metric type correctly.

### Issue: YAML configuration not loading
**Solution**: Verify YAML schema and check Pydantic validation errors.

## Support and Resources

- **Documentation Website**: Auto-deployed to GitHub Pages
- **Source Code**: `src/polars_eval_metrics/`
- **Reference Implementation**: `plan/` (read-only)
- **Issues**: GitHub Issues
- **API Reference**: Generated from docstrings

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Last updated: 2024*  
*Maintainer: Engineering Team*