# Documentation

This directory contains the documentation for the polars-eval-metrics package.

## Structure

- `index.qmd` - Main documentation page
- `getting_started.qmd` - Installation and basic concepts
- `examples/` - Detailed usage examples
  - `basic_usage.qmd` - Common use cases
  - `metric_factory.qmd` - Creating metrics from configuration
  - `advanced_usage.qmd` - Advanced features
- `data_generator.py` - Sample data generation for examples

## Running the Examples

The documentation uses Quarto notebooks (`.qmd` files) that can be rendered to HTML.

### Prerequisites

Make sure you have the required dependencies:
- polars
- pydantic
- pyyaml

### Import Setup

The `.qmd` files are configured to automatically add the correct paths:

```python
import sys
import os
sys.path.insert(0, os.path.abspath("../src"))  # For package imports
```

### Running Individual Examples

You can run the Python code from the examples directly:

```bash
cd docs
python -c "
import sys, os
sys.path.insert(0, os.path.abspath('../src'))

from polars_eval_metrics import MetricFactory
from data_generator import generate_sample_data

# Your example code here
"
```

### Rendering with Quarto

If you have Quarto installed, you can render the documentation:

```bash
quarto render
```

This will create an HTML website in the `_site` directory.

## Usage Pattern

The main pattern demonstrated across all examples:

```python
# 1. Create metrics from configuration
config = {
    'metrics': [
        {'name': 'mae', 'label': 'Mean Absolute Error'},
        {'name': 'rmse', 'label': 'Root Mean Squared Error'}
    ]
}
metrics = MetricFactory.from_config(config)

# 2. Initialize evaluator with complete context
evaluator = MetricEvaluator(
    df=data,
    metrics=metrics,
    ground_truth="actual",
    estimates=["model1", "model2"],
    group_by=["treatment"]
)

# 3. Evaluate
results = evaluator.evaluate_all()
```