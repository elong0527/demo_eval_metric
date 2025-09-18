"""
Unified Expression Registry System

This module provides an extensible registry system for all types of expressions:
- Error expressions (for data preparation)
- Metric expressions (for aggregation)
- Summary expressions (for second-level aggregation)

Supports both scoped, injectable registries and the legacy singleton interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

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


@dataclass(slots=True)
class ExpressionRegistry:
    """Instance-based registry for metric, error, and summary expressions."""

    _errors: dict[str, Callable[..., pl.Expr]] = field(default_factory=dict)
    _metrics: dict[str, pl.Expr | Callable[[], pl.Expr]] = field(default_factory=dict)
    _summaries: dict[str, pl.Expr | Callable[[], pl.Expr]] = field(default_factory=dict)

    # ---------------------------- Error Expressions ----------------------------

    # ============ Error Expression Methods ============

    def register_error(self, name: str, func: Callable[..., pl.Expr]) -> None:
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
        self._errors[name] = func

    def get_error(
        self,
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
        if name not in self._errors:
            raise MetricNotFoundError(name, list(self._errors.keys()), "error")

        func = self._errors[name]
        return func(estimate, ground_truth, **params)

    def generate_error_columns(
        self,
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
            error_types = list(self._errors.keys())

        error_params = error_params or {}
        expressions = []

        for error_type in error_types:
            params = error_params.get(error_type, {})
            expr = self.get_error(error_type, estimate, ground_truth, **params)
            expressions.append(expr.alias(error_type))

        return expressions

    def list_errors(self) -> list[str]:
        """List all available error types."""
        return list(self._errors.keys())

    def has_error(self, name: str) -> bool:
        """Check if an error type is registered."""
        return name in self._errors

    # ============ Metric Expression Methods ============

    def register_metric(self, name: str, expr: pl.Expr | Callable[[], pl.Expr]) -> None:
        """
        Register a custom metric expression.

        Args:
            name: Name of the metric (e.g., 'mae', 'custom_accuracy')
            expr: Polars expression or callable that returns a Polars expression
                  The expression should typically reference error columns and produce a 'value' alias

        Example:
            MetricRegistry.register_metric('mae', pl.col('absolute_error').mean().alias('value'))
        """
        self._metrics[name] = expr

    def register_summary(
        self, name: str, expr: pl.Expr | Callable[[], pl.Expr]
    ) -> None:
        """
        Register a custom summary expression.

        Args:
            name: Name of the summary (e.g., 'mean', 'p90')
            expr: Polars expression or callable that returns a Polars expression
                  The expression should typically operate on 'value' column

        Example:
            MetricRegistry.register_summary('p90', pl.col('value').quantile(0.9, interpolation="linear"))
        """
        self._summaries[name] = expr

    def get_metric(self, name: str) -> pl.Expr:
        """
        Get a metric expression by name.

        Args:
            name: Name of the metric

        Returns:
            Polars expression for the metric

        Raises:
            ValueError: If the metric is not registered
        """
        if name not in self._metrics:
            raise MetricNotFoundError(name, list(self._metrics.keys()), "metric")

        expr = self._metrics[name]
        # If it's a callable, call it to get the expression
        if callable(expr):
            return expr()
        return expr

    def get_summary(self, name: str) -> pl.Expr:
        """
        Get a summary expression by name.

        Args:
            name: Name of the summary

        Returns:
            Polars expression for the summary

        Raises:
            ValueError: If the summary is not registered
        """
        if name not in self._summaries:
            raise MetricNotFoundError(name, list(self._summaries.keys()), "summary")

        expr = self._summaries[name]
        # If it's a callable, call it to get the expression
        if callable(expr):
            return expr()
        return expr

    def list_metrics(self) -> list[str]:
        """List all available metrics."""
        return list(self._metrics.keys())

    def list_summaries(self) -> list[str]:
        """List all available summaries."""
        return list(self._summaries.keys())

    def has_metric(self, name: str) -> bool:
        """Check if a metric is registered."""
        return name in self._metrics

    def has_summary(self, name: str) -> bool:
        """Check if a summary is registered."""
        return name in self._summaries

    # -------------------------- Registry Introspection -------------------------

    def snapshot(self) -> "ExpressionRegistry":
        """Create a shallow copy of the registry for safe mutation."""
        clone = ExpressionRegistry()
        clone._errors.update(self._errors)
        clone._metrics.update(self._metrics)
        clone._summaries.update(self._summaries)
        return clone

    def as_readonly(self) -> Mapping[str, Callable[..., pl.Expr]]:
        """Provide read-only access to error registry."""
        return dict(self._errors)

    def iter_metrics(self) -> Iterable[str]:
        """Iterate through registered metric names."""
        return self._metrics.keys()


class MetricRegistry:
    """Legacy singleton wrapper around an :class:`ExpressionRegistry`."""

    _registry: ExpressionRegistry = ExpressionRegistry()

    @classmethod
    def configure(cls, registry: ExpressionRegistry, *, copy: bool = False) -> None:
        """Configure the global registry used by legacy class methods."""

        cls._registry = registry.snapshot() if copy else registry

    @classmethod
    def get_registry(cls) -> ExpressionRegistry:
        """Return the currently configured registry instance."""

        return cls._registry

    @classmethod
    def reset_defaults(cls) -> None:
        """Restore the registry to the bundled built-in expressions."""

        cls._registry = create_builtin_registry()

    # Delegated API -----------------------------------------------------------

    @classmethod
    def register_error(cls, name: str, func: Callable[..., pl.Expr]) -> None:
        cls._registry.register_error(name, func)

    @classmethod
    def get_error(
        cls,
        name: str,
        estimate: str,
        ground_truth: str,
        **params: Any,
    ) -> pl.Expr:
        return cls._registry.get_error(name, estimate, ground_truth, **params)

    @classmethod
    def generate_error_columns(
        cls,
        estimate: str,
        ground_truth: str,
        error_types: list[str] | None = None,
        error_params: dict[str, dict[str, Any]] | None = None,
    ) -> list[pl.Expr]:
        return cls._registry.generate_error_columns(
            estimate, ground_truth, error_types, error_params
        )

    @classmethod
    def list_errors(cls) -> list[str]:
        return cls._registry.list_errors()

    @classmethod
    def has_error(cls, name: str) -> bool:
        return cls._registry.has_error(name)

    @classmethod
    def register_metric(cls, name: str, expr: pl.Expr | Callable[[], pl.Expr]) -> None:
        cls._registry.register_metric(name, expr)

    @classmethod
    def register_summary(cls, name: str, expr: pl.Expr | Callable[[], pl.Expr]) -> None:
        cls._registry.register_summary(name, expr)

    @classmethod
    def get_metric(cls, name: str) -> pl.Expr:
        return cls._registry.get_metric(name)

    @classmethod
    def get_summary(cls, name: str) -> pl.Expr:
        return cls._registry.get_summary(name)

    @classmethod
    def list_metrics(cls) -> list[str]:
        return cls._registry.list_metrics()

    @classmethod
    def list_summaries(cls) -> list[str]:
        return cls._registry.list_summaries()

    @classmethod
    def has_metric(cls, name: str) -> bool:
        return cls._registry.has_metric(name)

    @classmethod
    def has_summary(cls, name: str) -> bool:
        return cls._registry.has_summary(name)


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


def _register_builtin_errors(registry: ExpressionRegistry) -> None:
    registry.register_error("error", _error)
    registry.register_error("absolute_error", _absolute_error)
    registry.register_error("squared_error", _squared_error)
    registry.register_error("percent_error", _percent_error)
    registry.register_error("absolute_percent_error", _absolute_percent_error)


def _register_builtin_metrics(registry: ExpressionRegistry) -> None:
    registry.register_metric("me", pl.col("error").mean().alias("value"))

    registry.register_metric("mae", pl.col("absolute_error").mean().alias("value"))
    registry.register_metric("mse", pl.col("squared_error").mean().alias("value"))
    registry.register_metric(
        "rmse", pl.col("squared_error").mean().sqrt().alias("value")
    )
    registry.register_metric("mpe", pl.col("percent_error").mean().alias("value"))
    registry.register_metric(
        "mape", pl.col("absolute_percent_error").mean().alias("value")
    )

    registry.register_metric(
        "n_subject", pl.col("subject_id").n_unique().alias("value")
    )
    registry.register_metric(
        "n_visit", pl.struct(["subject_id", "visit_id"]).n_unique().alias("value")
    )
    registry.register_metric(
        "n_sample",
        pl.col("sample_index").n_unique().alias("value"),
    )

    # Metrics for subjects with data (non-null ground truth or estimates)
    registry.register_metric(
        "n_subject_with_data",
        pl.col("subject_id")
        .filter(pl.col("error").is_not_null())
        .n_unique()
        .alias("value"),
    )
    registry.register_metric(
        "pct_subject_with_data",
        (
            pl.col("subject_id").filter(pl.col("error").is_not_null()).n_unique()
            / pl.col("subject_id").n_unique()
            * 100
        ).alias("value"),
    )

    # Metrics for visits with data
    registry.register_metric(
        "n_visit_with_data",
        pl.struct(["subject_id", "visit_id"])
        .filter(pl.col("error").is_not_null())
        .n_unique()
        .alias("value"),
    )
    registry.register_metric(
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
    registry.register_metric(
        "n_sample_with_data", pl.col("error").is_not_null().sum().alias("value")
    )
    registry.register_metric(
        "pct_sample_with_data",
        (pl.col("error").is_not_null().mean() * 100).alias("value"),
    )


def _register_builtin_summaries(registry: ExpressionRegistry) -> None:
    registry.register_summary("mean", pl.col("value").mean())
    registry.register_summary("median", pl.col("value").median())
    registry.register_summary("std", pl.col("value").std())
    registry.register_summary("min", pl.col("value").min())
    registry.register_summary("max", pl.col("value").max())
    registry.register_summary("sum", pl.col("value").sum())
    registry.register_summary("sqrt", pl.col("value").sqrt())

    # Register percentile summaries
    percentiles = [1, 5, 25, 75, 90, 95, 99]
    for p in percentiles:
        registry.register_summary(
            f"p{p}", pl.col("value").quantile(p / 100, interpolation="linear")
        )


def create_builtin_registry() -> ExpressionRegistry:
    """Create a registry populated with built-in errors, metrics, and summaries."""

    registry = ExpressionRegistry()
    _register_builtin_errors(registry)
    _register_builtin_metrics(registry)
    _register_builtin_summaries(registry)
    return registry


# Initialize global registry with built-in expressions for backward compatibility
MetricRegistry.reset_defaults()
