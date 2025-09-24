# Documentation

This directory contains the documentation for the polars-eval-metrics package.

## Structure

- `index.qmd` - Main documentation landing page and navigation links
- `quickstart.qmd` - Installation steps and the core evaluation workflow
- `metric_define.qmd` - Detailed MetricDefine usage and configuration examples
- `metric_evaluator.qmd` - End-to-end evaluation pipeline walkthroughs
- `metric_pivot.qmd` - Pivot helper guides for reporting outputs
- `developer_notes.qmd` - Additional reference notes for contributors
- `data_generator.py` - Sample data generation helpers used across notebooks
- `styles.css` - Custom styling overrides for the rendered site

## Running the Examples

The documentation uses Quarto notebooks (`.qmd` files) that can be rendered to HTML.

### Prerequisites

Make sure you have the required dependencies:
- polars
- pydantic
- pyyaml

### Running Individual Examples

You can run the Python code from the examples directly:

```bash
cd docs
python -c "
import sys, os
sys.path.insert(0, os.path.abspath('../src'))

from polars_eval_metrics import MetricDefine, MetricEvaluator, create_metrics
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