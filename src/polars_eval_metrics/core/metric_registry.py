"""
Unified Expression Registry System

This module provides an extensible registry system for all types of expressions:
- Error expressions (for data preparation)
- Metric expressions (for aggregation)
- Selector expressions (for second-level aggregation)

Supports both global (class-level) and local (instance-level) registries.
"""

import polars as pl
from typing import Callable, Dict, Any, Optional
import copy


class MetricRegistry:
    """
    Unified registry for all expression types used in metric evaluation.

    Can be used as:
    1. Global registry (class methods) - shared across all evaluations
    2. Instance registry - isolated for specific evaluation contexts

    Instance registries inherit from the global registry but can override
    or extend with their own expressions.
    """

    # Global class-level registries
    _global_errors: Dict[str, Callable[[str, str, ...], pl.Expr]] = {}
    _global_metrics: Dict[str, pl.Expr | Callable[[], pl.Expr]] = {}
    _global_selectors: Dict[str, pl.Expr | Callable[[], pl.Expr]] = {}

    def __init__(self, inherit_global: bool = True):
        """
        Initialize a registry instance.

        Args:
            inherit_global: If True, inherits all global registrations.
                          If False, starts with empty registries.
        """
        if inherit_global:
            # Copy global registries to instance (not deep copy for functions)
            self._errors = dict(self._global_errors)
            self._metrics = dict(self._global_metrics)
            self._selectors = dict(self._global_selectors)
        else:
            # Start with empty registries
            self._errors = {}
            self._metrics = {}
            self._selectors = {}

    # ============ Error Expression Methods ============

    def register_error(self, name: str, func: Callable[..., pl.Expr]) -> None:
        """
        Register a custom error expression function in this instance.

        Args:
            name: Name of the error type (e.g., 'absolute_error', 'buffer_error')
            func: Function that takes (estimate, ground_truth, *args) and returns pl.Expr

        Example:
            registry = MetricRegistry()
            def buffer_error(estimate: str, ground_truth: str, threshold: float = 0.5):
                return (pl.col(estimate) - pl.col(ground_truth)).abs() <= threshold

            registry.register_error('buffer_error', buffer_error)
        """
        self._errors[name] = func

    @classmethod
    def register_error_global(cls, name: str, func: Callable[..., pl.Expr]) -> None:
        """
        Register a custom error expression function globally.

        Args:
            name: Name of the error type
            func: Function that takes (estimate, ground_truth, *args) and returns pl.Expr
        """
        cls._global_errors[name] = func

    def get_error(
        self, name: str, estimate: str, ground_truth: str, **params
    ) -> pl.Expr:
        """
        Get an error expression by name from this instance.

        Args:
            name: Name of the error type
            estimate: Estimate column name
            ground_truth: Ground truth column name
            **params: Additional parameters for parameterized error functions

        Returns:
            Polars expression that computes the error
        """
        if name not in self._errors:
            raise ValueError(
                f"Error type '{name}' not registered. Available: {list(self._errors.keys())}"
            )

        func = self._errors[name]
        return func(estimate, ground_truth, **params)

    @classmethod
    def get_error_global(
        cls, name: str, estimate: str, ground_truth: str, **params
    ) -> pl.Expr:
        """Get an error expression by name from global registry."""
        if name not in cls._global_errors:
            raise ValueError(
                f"Error type '{name}' not registered globally. Available: {list(cls._global_errors.keys())}"
            )

        func = cls._global_errors[name]
        return func(estimate, ground_truth, **params)

    def generate_error_columns(
        self,
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
            error_types: List of error types to generate. If None, uses ALL registered types.
            error_params: Dictionary mapping error types to their parameters

        Returns:
            List of Polars expressions with aliases matching the error_types names
        """
        if error_types is None:
            # Use all registered error types in this instance
            error_types = list(self._errors.keys())

        error_params = error_params or {}
        expressions = []

        for error_type in error_types:
            params = error_params.get(error_type, {})
            expr = self.get_error(error_type, estimate, ground_truth, **params)
            expressions.append(expr.alias(error_type))

        return expressions

    def list_errors(self) -> list[str]:
        """List all available error types in this instance."""
        return list(self._errors.keys())

    def has_error(self, name: str) -> bool:
        """Check if an error type is registered in this instance."""
        return name in self._errors

    # ============ Metric Expression Methods ============

    def register_metric(self, name: str, expr: pl.Expr | Callable[[], pl.Expr]) -> None:
        """
        Register a custom metric expression in this instance.

        Args:
            name: Name of the metric (e.g., 'mae', 'custom_accuracy')
            expr: Polars expression or callable that returns a Polars expression
                  The expression should typically reference error columns and produce a 'value' alias

        Example:
            registry = MetricRegistry()
            registry.register_metric('mae', pl.col('absolute_error').mean().alias('value'))
        """
        self._metrics[name] = expr

    @classmethod
    def register_metric_global(
        cls, name: str, expr: pl.Expr | Callable[[], pl.Expr]
    ) -> None:
        """Register a custom metric expression globally."""
        cls._global_metrics[name] = expr

    def register_selector(
        self, name: str, expr: pl.Expr | Callable[[], pl.Expr]
    ) -> None:
        """
        Register a custom selector expression in this instance.

        Args:
            name: Name of the selector (e.g., 'mean', 'p90')
            expr: Polars expression or callable that returns a Polars expression
                  The expression should typically operate on 'value' column

        Example:
            registry = MetricRegistry()
            registry.register_selector('p90', pl.col('value').quantile(0.9))
        """
        self._selectors[name] = expr

    @classmethod
    def register_selector_global(
        cls, name: str, expr: pl.Expr | Callable[[], pl.Expr]
    ) -> None:
        """Register a custom selector expression globally."""
        cls._global_selectors[name] = expr

    def get_metric(self, name: str) -> pl.Expr:
        """
        Get a metric expression by name from this instance.

        Args:
            name: Name of the metric

        Returns:
            Polars expression for the metric

        Raises:
            ValueError: If the metric is not registered
        """
        if name not in self._metrics:
            raise ValueError(
                f"Metric '{name}' not registered. Available metrics: {list(self._metrics.keys())}"
            )

        expr = self._metrics[name]
        # If it's a callable, call it to get the expression
        if callable(expr):
            return expr()
        return expr

    @classmethod
    def get_metric_global(cls, name: str) -> pl.Expr:
        """Get a metric expression by name from global registry."""
        if name not in cls._global_metrics:
            raise ValueError(
                f"Metric '{name}' not registered globally. Available metrics: {list(cls._global_metrics.keys())}"
            )

        expr = cls._global_metrics[name]
        if callable(expr):
            return expr()
        return expr

    def get_selector(self, name: str) -> pl.Expr:
        """
        Get a selector expression by name from this instance.

        Args:
            name: Name of the selector

        Returns:
            Polars expression for the selector

        Raises:
            ValueError: If the selector is not registered
        """
        if name not in self._selectors:
            raise ValueError(
                f"Selector '{name}' not registered. Available selectors: {list(self._selectors.keys())}"
            )

        expr = self._selectors[name]
        # If it's a callable, call it to get the expression
        if callable(expr):
            return expr()
        return expr

    @classmethod
    def get_selector_global(cls, name: str) -> pl.Expr:
        """Get a selector expression by name from global registry."""
        if name not in cls._global_selectors:
            raise ValueError(
                f"Selector '{name}' not registered globally. Available selectors: {list(cls._global_selectors.keys())}"
            )

        expr = cls._global_selectors[name]
        if callable(expr):
            return expr()
        return expr

    def list_metrics(self) -> list[str]:
        """List all available metrics in this instance."""
        return list(self._metrics.keys())

    def list_selectors(self) -> list[str]:
        """List all available selectors in this instance."""
        return list(self._selectors.keys())

    def has_metric(self, name: str) -> bool:
        """Check if a metric is registered in this instance."""
        return name in self._metrics

    def has_selector(self, name: str) -> bool:
        """Check if a selector is registered in this instance."""
        return name in self._selectors


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
MetricRegistry.register_error_global("error", _error)
MetricRegistry.register_error_global("absolute_error", _absolute_error)
MetricRegistry.register_error_global("squared_error", _squared_error)
MetricRegistry.register_error_global("percent_error", _percent_error)
MetricRegistry.register_error_global("absolute_percent_error", _absolute_percent_error)

# Register built-in metrics
MetricRegistry.register_metric_global(
    "mae", pl.col("absolute_error").mean().alias("value")
)
MetricRegistry.register_metric_global(
    "mse", pl.col("squared_error").mean().alias("value")
)
MetricRegistry.register_metric_global(
    "rmse", pl.col("squared_error").mean().sqrt().alias("value")
)
MetricRegistry.register_metric_global("bias", pl.col("error").mean().alias("value"))
MetricRegistry.register_metric_global(
    "mape", pl.col("absolute_percent_error").mean().alias("value")
)
MetricRegistry.register_metric_global(
    "n_subject", pl.col("subject_id").n_unique().alias("value")
)
MetricRegistry.register_metric_global(
    "n_visit", pl.struct(["subject_id", "visit_id"]).n_unique().alias("value")
)
MetricRegistry.register_metric_global("n_sample", pl.len().alias("value"))
MetricRegistry.register_metric_global(
    "total_subject", pl.col("subject_id").n_unique().alias("value")
)
MetricRegistry.register_metric_global(
    "total_visit", pl.struct(["subject_id", "visit_id"]).n_unique().alias("value")
)

# Register built-in selectors
MetricRegistry.register_selector_global("mean", pl.col("value").mean())
MetricRegistry.register_selector_global("median", pl.col("value").median())
MetricRegistry.register_selector_global("std", pl.col("value").std())
MetricRegistry.register_selector_global("min", pl.col("value").min())
MetricRegistry.register_selector_global("max", pl.col("value").max())
MetricRegistry.register_selector_global("sum", pl.col("value").sum())
MetricRegistry.register_selector_global("sqrt", pl.col("value").sqrt())
