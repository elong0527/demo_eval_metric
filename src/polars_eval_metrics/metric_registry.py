"""
Unified Expression Registry System

This module provides an extensible registry system for all types of expressions:
- Error expressions (for data preparation)
- Metric expressions (for aggregation)
- Summary expressions (for second-level aggregation)

Supports both global (class-level) and local (instance-level) registries.
"""

from typing import Any, Callable

# pyre-strict

import polars as pl


class MetricNotFoundError(ValueError):
    """Exception raised when a requested metric/error/summary is not found."""

    def __init__(
        self, name: str, available: list[str], expr_type: str = "expression"
    ) -> None:
        self.name = name
        self.available = available
        self.expr_type = expr_type
        super().__init__(
            f"{expr_type.capitalize()} '{name}' not found. "
            f"Available {expr_type}s: {', '.join(available) if available else 'none'}"
        )


class MetricRegistry:
    """
    Unified registry for all expression types used in metric evaluation.

    This is a singleton-style class that provides global registration
    and retrieval of error types, metrics, and summary expressions.
    """

    # Class-level registries
    _errors: dict[str, Callable[..., pl.Expr]] = {}
    _metrics: dict[str, pl.Expr | Callable[[], pl.Expr]] = {}
    _summaries: dict[str, pl.Expr | Callable[[], pl.Expr]] = {}

    # ============ Error Expression Methods ============

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
        cls,
        name: str,
        estimate: str,
        ground_truth: str,
        **params: Any,
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
            raise MetricNotFoundError(name, list(cls._errors.keys()), "error")

        func = cls._errors[name]
        return func(estimate, ground_truth, **params)

    @classmethod
    def generate_error_columns(
        cls,
        estimate: str,
        ground_truth: str,
        error_types: list[str] | None = None,
        error_params: dict[str, dict[str, Any]] | None = None,
    ) -> list[pl.Expr]:
        """
        Generate error column expressions for specified error types.

        Args:
            estimate: Estimate column name
            ground_truth: Ground truth column name
            error_types: List of error types to generate. If None, uses ALL registered types.
            error_params: Dictionary mapping error types to their parameters

        Returns:
            List of Polars expressions with aliases matching the error_types names
        """
        if error_types is None:
            # Use all registered error types
            error_types = list(cls._errors.keys())

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

    # ============ Metric Expression Methods ============

    @classmethod
    def register_metric(cls, name: str, expr: pl.Expr | Callable[[], pl.Expr]) -> None:
        """
        Register a custom metric expression.

        Args:
            name: Name of the metric (e.g., 'mae', 'custom_accuracy')
            expr: Polars expression or callable that returns a Polars expression
                  The expression should typically reference error columns and produce a 'value' alias

        Example:
            MetricRegistry.register_metric('mae', pl.col('absolute_error').mean().alias('value'))
        """
        cls._metrics[name] = expr

    @classmethod
    def register_summary(cls, name: str, expr: pl.Expr | Callable[[], pl.Expr]) -> None:
        """
        Register a custom summary expression.

        Args:
            name: Name of the summary (e.g., 'mean', 'p90')
            expr: Polars expression or callable that returns a Polars expression
                  The expression should typically operate on 'value' column

        Example:
            MetricRegistry.register_summary('p90', pl.col('value').quantile(0.9))
        """
        cls._summaries[name] = expr

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
            raise MetricNotFoundError(name, list(cls._metrics.keys()), "metric")

        expr = cls._metrics[name]
        # If it's a callable, call it to get the expression
        if callable(expr):
            return expr()
        return expr

    @classmethod
    def get_summary(cls, name: str) -> pl.Expr:
        """
        Get a summary expression by name.

        Args:
            name: Name of the summary

        Returns:
            Polars expression for the summary

        Raises:
            ValueError: If the summary is not registered
        """
        if name not in cls._summaries:
            raise MetricNotFoundError(name, list(cls._summaries.keys()), "summary")

        expr = cls._summaries[name]
        # If it's a callable, call it to get the expression
        if callable(expr):
            return expr()
        return expr

    @classmethod
    def list_metrics(cls) -> list[str]:
        """List all available metrics."""
        return list(cls._metrics.keys())

    @classmethod
    def list_summaries(cls) -> list[str]:
        """List all available summaries."""
        return list(cls._summaries.keys())

    @classmethod
    def has_metric(cls, name: str) -> bool:
        """Check if a metric is registered."""
        return name in cls._metrics

    @classmethod
    def has_summary(cls, name: str) -> bool:
        """Check if a summary is registered."""
        return name in cls._summaries


# ============ Built-in Error Expression Functions ============


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


# ============ Register All Built-in Expressions Globally ============

# Register built-in error types
MetricRegistry.register_error("error", _error)
MetricRegistry.register_error("absolute_error", _absolute_error)
MetricRegistry.register_error("squared_error", _squared_error)
MetricRegistry.register_error("percent_error", _percent_error)
MetricRegistry.register_error("absolute_percent_error", _absolute_percent_error)

# Register built-in metrics
MetricRegistry.register_metric("me", pl.col("error").mean().alias("value"))

MetricRegistry.register_metric("mae", pl.col("absolute_error").mean().alias("value"))
MetricRegistry.register_metric("mse", pl.col("squared_error").mean().alias("value"))
MetricRegistry.register_metric(
    "rmse", pl.col("squared_error").mean().sqrt().alias("value")
)
MetricRegistry.register_metric("mpe", pl.col("percent_error").mean().alias("value"))
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

# Metrics for subjects with data (non-null ground truth or estimates)
MetricRegistry.register_metric(
    "n_subject_with_data",
    pl.col("subject_id")
    .filter(pl.col("error").is_not_null())
    .n_unique()
    .alias("value"),
)
MetricRegistry.register_metric(
    "pct_subject_with_data",
    (
        pl.col("subject_id").filter(pl.col("error").is_not_null()).n_unique()
        / pl.col("subject_id").n_unique()
        * 100
    ).alias("value"),
)

# Metrics for visits with data
MetricRegistry.register_metric(
    "n_visit_with_data",
    pl.struct(["subject_id", "visit_id"])
    .filter(pl.col("error").is_not_null())
    .n_unique()
    .alias("value"),
)
MetricRegistry.register_metric(
    "pct_visit_with_data",
    (
        pl.struct(["subject_id", "visit_id"])
        .filter(pl.col("error").is_not_null())
        .n_unique()
        / pl.struct(["subject_id", "visit_id"]).n_unique()
        * 100
    ).alias("value"),
)

# Metrics for samples with data
MetricRegistry.register_metric(
    "n_sample_with_data", pl.col("error").is_not_null().sum().alias("value")
)
MetricRegistry.register_metric(
    "pct_sample_with_data", (pl.col("error").is_not_null().mean() * 100).alias("value")
)

# Register built-in summaries
MetricRegistry.register_summary("mean", pl.col("value").mean())
MetricRegistry.register_summary("median", pl.col("value").median())
MetricRegistry.register_summary("std", pl.col("value").std())
MetricRegistry.register_summary("min", pl.col("value").min())
MetricRegistry.register_summary("max", pl.col("value").max())
MetricRegistry.register_summary("sum", pl.col("value").sum())
MetricRegistry.register_summary("sqrt", pl.col("value").sqrt())

# Register percentile summaries
percentiles = [1, 5, 25, 75, 90, 95, 99]
for p in percentiles:
    MetricRegistry.register_summary(f"p{p}", pl.col("value").quantile(p / 100))
