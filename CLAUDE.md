# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a model evaluation framework repository that uses:
- **Polars** and **Pandas** for data manipulation and analysis
- **Plotnine** for creating grammar of graphics visualizations (ggplot2-style)
- **Pydantic v2** for data validation and type checking
- **Pytest** for unit testing

## Development Commands

### Testing
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_metrics.py

# Run tests with verbose output
pytest -v

# Run specific test function
pytest tests/test_metrics.py::test_specific_function
```

### Linting and Type Checking
```bash
# Run type checking with mypy
mypy src/

# Format code with black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Run ruff for linting
ruff check src/ tests/
```

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

## Architecture Guidelines

### Data Processing
- Use **Polars** for high-performance data operations when dealing with large datasets
- Use **Pandas** for compatibility with existing data science workflows or when specific pandas functionality is needed
- Prefer Polars LazyFrames for query optimization on large datasets

### Type Safety
- All data models should use **Pydantic v2** BaseModel or dataclasses
- Define explicit types for evaluation metrics, model outputs, and configuration
- Use Pydantic's validators for data validation

### Visualization
- Use **Plotnine** for all statistical visualizations
- Follow grammar of graphics principles
- Create reusable plotting functions that return ggplot objects

### Testing Strategy
- Write unit tests for all metric calculations
- Use pytest fixtures for test data setup
- Mock external dependencies appropriately
- Aim for high test coverage on core evaluation logic

## Project Structure Conventions

```
src/
  metrics/       # Evaluation metric implementations
  visualization/ # Plotnine visualization functions
  models/        # Pydantic models for type safety
  utils/         # Helper functions
tests/
  fixtures/      # Pytest fixtures and test data
  unit/          # Unit tests mirroring src structure
```

## Key Implementation Notes

- When implementing new metrics, ensure they work with both Polars and Pandas DataFrames
- All metric functions should have type hints and docstrings
- Visualization functions should be parameterizable and return plotnine objects
- Use Pydantic models for configuration and validation of metric parameters