"""
Unified Expression Registry System

This module provides an extensible registry system for all types of expressions:
- Error expressions (for data preparation)
- Metric expressions (for aggregation)
- Selector expressions (for second-level aggregation)
"""

import polars as pl
from typing import Callable, Dict, Any


class MetricRegistry:
    """
    Unified registry for all expression types used in metric evaluation.

    Manages error expressions, metrics, and selectors in a single place,
    providing a consistent API for registration and retrieval.
    """

    # Class-level registries
    _errors: Dict[str, Callable[[str, str, ...], pl.Expr]] = {}
    _metrics: Dict[str, pl.Expr | Callable[[], pl.Expr]] = {}
    _selectors: Dict[str, pl.Expr | Callable[[], pl.Expr]] = {}

    # Error Expression Methods
    @classmethod
    def register_error(cls, name: str, func: Callable[..., pl.Expr]) -> None:
        """
        Register a custom error expression function.

        Args:
            name: Name of the error type (e.g., 'absolute_error', 'buffer_error')
            func: Function that takes (estimate, ground_truth, *args) and returns pl.Expr

        Example:
            def buffer_error(estimate: str, ground_truth: str, threshold: float = 0.5):
                return (pl.col(estimate) - pl.col(ground_truth)).abs() <= threshold

            MetricRegistry.register_error('buffer_error', buffer_error)
        """
        cls._errors[name] = func

    @classmethod
    def get_error(
        cls, name: str, estimate: str, ground_truth: str, **params
    ) -> pl.Expr:
        """
        Get an error expression by name.

        Args:
            name: Name of the error type
            estimate: Estimate column name
            ground_truth: Ground truth column name
            **params: Additional parameters for parameterized error functions

        Returns:
            Polars expression that computes the error
        """
        if name not in cls._errors:
            raise ValueError(
                f"Error type '{name}' not registered. Available: {list(cls._errors.keys())}"
            )

        func = cls._errors[name]
        return func(estimate, ground_truth, **params)

    @classmethod
    def generate_error_columns(
        cls,
        estimate: str,
        ground_truth: str,
        error_types: list[str] | None = None,
        error_params: Dict[str, Dict[str, Any]] | None = None,
    ) -> list[pl.Expr]:
        """
        Generate error column expressions for specified error types.

        Args:
            estimate: Estimate column name
            ground_truth: Ground truth column name
            error_types: List of error types to generate. If None, uses default built-in types.
            error_params: Dictionary mapping error types to their parameters

        Returns:
            List of Polars expressions with aliases matching the error_types names
        """
        if error_types is None:
            # Default built-in error types for backward compatibility
            error_types = [
                "error",
                "absolute_error",
                "squared_error",
                "percent_error",
                "absolute_percent_error",
            ]

        error_params = error_params or {}
        expressions = []

        for error_type in error_types:
            params = error_params.get(error_type, {})
            expr = cls.get_error(error_type, estimate, ground_truth, **params)
            expressions.append(expr.alias(error_type))

        return expressions

    @classmethod
    def list_errors(cls) -> list[str]:
        """List all available error types."""
        return list(cls._errors.keys())

    @classmethod
    def has_error(cls, name: str) -> bool:
        """Check if an error type is registered."""
        return name in cls._errors

    # Metric Expression Methods
    @classmethod
    def register_metric(cls, name: str, expr: pl.Expr | Callable[[], pl.Expr]) -> None:
        """
        Register a custom metric expression.

        Args:
            name: Name of the metric (e.g., 'mae', 'custom_accuracy')
            expr: Polars expression or callable that returns a Polars expression
                  The expression should typically reference error columns and produce a 'value' alias

        Example:
            # Register as expression
            MetricRegistry.register_metric('mae', pl.col('absolute_error').mean().alias('value'))

            # Register as callable
            def custom_mae():
                return pl.col('absolute_error').mean().alias('value')
            MetricRegistry.register_metric('custom_mae', custom_mae)
        """
        cls._metrics[name] = expr

    @classmethod
    def register_selector(
        cls, name: str, expr: pl.Expr | Callable[[], pl.Expr]
    ) -> None:
        """
        Register a custom selector expression.

        Args:
            name: Name of the selector (e.g., 'mean', 'p90')
            expr: Polars expression or callable that returns a Polars expression
                  The expression should typically operate on 'value' column

        Example:
            # Register percentile selector
            MetricRegistry.register_selector('p90', pl.col('value').quantile(0.9))
        """
        cls._selectors[name] = expr

    @classmethod
    def get_metric(cls, name: str) -> pl.Expr:
        """
        Get a metric expression by name.

        Args:
            name: Name of the metric

        Returns:
            Polars expression for the metric

        Raises:
            ValueError: If the metric is not registered
        """
        if name not in cls._metrics:
            raise ValueError(
                f"Metric '{name}' not registered. Available metrics: {list(cls._metrics.keys())}"
            )

        expr = cls._metrics[name]
        # If it's a callable, call it to get the expression
        if callable(expr):
            return expr()
        return expr

    @classmethod
    def get_selector(cls, name: str) -> pl.Expr:
        """
        Get a selector expression by name.

        Args:
            name: Name of the selector

        Returns:
            Polars expression for the selector

        Raises:
            ValueError: If the selector is not registered
        """
        if name not in cls._selectors:
            raise ValueError(
                f"Selector '{name}' not registered. Available selectors: {list(cls._selectors.keys())}"
            )

        expr = cls._selectors[name]
        # If it's a callable, call it to get the expression
        if callable(expr):
            return expr()
        return expr

    @classmethod
    def list_metrics(cls) -> list[str]:
        """List all available metrics."""
        return list(cls._metrics.keys())

    @classmethod
    def list_selectors(cls) -> list[str]:
        """List all available selectors."""
        return list(cls._selectors.keys())

    @classmethod
    def has_metric(cls, name: str) -> bool:
        """Check if a metric is registered."""
        return name in cls._metrics

    @classmethod
    def has_selector(cls, name: str) -> bool:
        """Check if a selector is registered."""
        return name in cls._selectors


# Register all built-in metrics
MetricRegistry.register_metric("mae", pl.col("absolute_error").mean().alias("value"))
MetricRegistry.register_metric("mse", pl.col("squared_error").mean().alias("value"))
MetricRegistry.register_metric(
    "rmse", pl.col("squared_error").mean().sqrt().alias("value")
)
MetricRegistry.register_metric("bias", pl.col("error").mean().alias("value"))
MetricRegistry.register_metric(
    "mape", pl.col("absolute_percent_error").mean().alias("value")
)
MetricRegistry.register_metric(
    "n_subject", pl.col("subject_id").n_unique().alias("value")
)
MetricRegistry.register_metric(
    "n_visit", pl.struct(["subject_id", "visit_id"]).n_unique().alias("value")
)
MetricRegistry.register_metric("n_sample", pl.len().alias("value"))
MetricRegistry.register_metric(
    "total_subject", pl.col("subject_id").n_unique().alias("value")
)
MetricRegistry.register_metric(
    "total_visit", pl.struct(["subject_id", "visit_id"]).n_unique().alias("value")
)

# Register all built-in selectors
MetricRegistry.register_selector("mean", pl.col("value").mean())
MetricRegistry.register_selector("median", pl.col("value").median())
MetricRegistry.register_selector("std", pl.col("value").std())
MetricRegistry.register_selector("min", pl.col("value").min())
MetricRegistry.register_selector("max", pl.col("value").max())
MetricRegistry.register_selector("sum", pl.col("value").sum())
MetricRegistry.register_selector("sqrt", pl.col("value").sqrt())


# Built-in error expression functions
def _error(estimate: str, ground_truth: str) -> pl.Expr:
    """Basic error: estimate - ground_truth"""
    return pl.col(estimate) - pl.col(ground_truth)


def _absolute_error(estimate: str, ground_truth: str) -> pl.Expr:
    """Absolute error: |estimate - ground_truth|"""
    error = pl.col(estimate) - pl.col(ground_truth)
    return error.abs()


def _squared_error(estimate: str, ground_truth: str) -> pl.Expr:
    """Squared error: (estimate - ground_truth)^2"""
    error = pl.col(estimate) - pl.col(ground_truth)
    return error**2


def _percent_error(estimate: str, ground_truth: str) -> pl.Expr:
    """Percent error: (estimate - ground_truth) / ground_truth * 100"""
    error = pl.col(estimate) - pl.col(ground_truth)
    return (
        pl.when(pl.col(ground_truth) != 0)
        .then(error / pl.col(ground_truth) * 100)
        .otherwise(None)
    )


def _absolute_percent_error(estimate: str, ground_truth: str) -> pl.Expr:
    """Absolute percent error: |(estimate - ground_truth) / ground_truth| * 100"""
    error = pl.col(estimate) - pl.col(ground_truth)
    return (
        pl.when(pl.col(ground_truth) != 0)
        .then((error / pl.col(ground_truth) * 100).abs())
        .otherwise(None)
    )


# Register all built-in error types
MetricRegistry.register_error("error", _error)
MetricRegistry.register_error("absolute_error", _absolute_error)
MetricRegistry.register_error("squared_error", _squared_error)
MetricRegistry.register_error("percent_error", _percent_error)
MetricRegistry.register_error("absolute_percent_error", _absolute_percent_error)
