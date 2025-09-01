# GitHub Actions Configuration

This directory contains GitHub Actions workflows for the demo_eval_metric project.

## Workflows

### 1. Documentation Build (`docs.yml`)
- **Trigger**: Push to main, pull requests, manual dispatch
- **Purpose**: Build and deploy Quarto documentation to GitHub Pages
- **Steps**:
  - Set up Python 3.11
  - Install dependencies (polars, pydantic, pyyaml, jupyter, etc.)
  - Install Quarto
  - Render examples.qmd to HTML
  - Deploy to GitHub Pages

### 2. Python Tests (`test.yml`)
- **Trigger**: Push to main, pull requests
- **Purpose**: Run tests and code quality checks
- **Matrix**: Python 3.11 and 3.12
- **Steps**:
  - Run pytest with coverage
  - Upload coverage to Codecov
  - Run ruff linting
  - Run black formatting check
  - Run mypy type checking

## Setup Requirements

To enable these workflows:

1. **GitHub Pages**: Enable GitHub Pages in repository settings with "GitHub Actions" as the source
2. **Secrets**: No secrets required for basic functionality
3. **Permissions**: The workflows use the default GITHUB_TOKEN with appropriate permissions

## Local Testing

To test locally:
```bash
# Install dependencies
pip install polars pydantic pyyaml plotnine pytest ruff black mypy

# Run tests
cd plan
python -m pytest tests/ -v

# Run linting
ruff check .
black --check .
mypy .

# Build documentation
quarto render examples.qmd --to html
```