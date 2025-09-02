# Polars Eval Metrics

[![codecov](https://codecov.io/gh/elong0527/demo_eval_metric/branch/main/graph/badge.svg)](https://codecov.io/gh/elong0527/demo_eval_metric)
[![Tests](https://github.com/elong0527/demo_eval_metric/actions/workflows/test.yml/badge.svg)](https://github.com/elong0527/demo_eval_metric/actions/workflows/test.yml)
[![Documentation](https://github.com/elong0527/demo_eval_metric/actions/workflows/docs.yml/badge.svg)](https://github.com/elong0527/demo_eval_metric/actions/workflows/docs.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A high-performance model evaluation framework built on Polars lazy evaluation.

## =Ú Documentation

Visit our [documentation website](https://elong0527.github.io/demo_eval_metric/) for detailed guides and examples.

## ( Features

- **=€ Fast**: Leverages Polars lazy evaluation for optimal performance
- **=' Flexible**: Support for custom metrics and expressions
- **<¯ Type-safe**: Pydantic models with validation
- **=Ê Simple**: Clean API with sensible defaults
- **= Extensible**: Easy to add new metrics and aggregation types

## =æ Installation

```bash
# Install from source
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Install with all dependencies (dev, test, docs)
pip install -e ".[dev,test,docs]"
```

## =€ Quick Start

```python
from polars_eval_metrics import MetricEvaluator, MetricFactory
import polars as pl

# Define metrics from configuration
config = {
    'metrics': [
        {'name': 'mae', 'label': 'Mean Absolute Error'},
        {'name': 'rmse', 'label': 'Root Mean Squared Error'}
    ]
}

# Create metrics
metrics = MetricFactory.from_dict(config)

# Create sample data
df = pl.DataFrame({
    "actual": [1.0, 2.0, 3.0, 4.0],
    "model1": [1.1, 2.2, 2.9, 4.1],
    "model2": [0.9, 2.1, 3.2, 3.8],
    "treatment": ["A", "A", "B", "B"]
})

# Evaluate
evaluator = MetricEvaluator(
    df=df,
    metrics=metrics,
    ground_truth="actual",
    estimates=["model1", "model2"],
    group_by=["treatment"]
)

results = evaluator.evaluate_all()
print(results)
```

## =Ê Code Coverage

We maintain comprehensive test coverage to ensure code quality:

- View our [coverage report](https://elong0527.github.io/demo_eval_metric/coverage.html)
- Check [Codecov](https://codecov.io/gh/elong0527/demo_eval_metric) for detailed metrics

## >ê Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/polars_eval_metrics --cov-report=html

# View coverage report
open htmlcov/index.html
```

## =Ö Documentation Development

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation locally
cd docs
quarto render

# Preview documentation
quarto preview
```

## > Contributing

Contributions are welcome! Please:

1. Write tests for new features
2. Ensure all tests pass
3. Maintain or improve code coverage
4. Follow the existing code style

## =Ä License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## = Links

- [Documentation](https://elong0527.github.io/demo_eval_metric/)
- [GitHub Repository](https://github.com/elong0527/demo_eval_metric)
- [Issue Tracker](https://github.com/elong0527/demo_eval_metric/issues)
- [Code Coverage](https://codecov.io/gh/elong0527/demo_eval_metric)