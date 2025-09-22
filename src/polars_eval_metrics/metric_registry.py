"""
Unified Expression Registry System

This module provides an extensible registry system for all types of expressions:
- Error expressions (for data preparation)
- Metric expressions (for aggregation)
- Summary expressions (for second-level aggregation)

Supports both global (class-level) and local (instance-level) registries.
"""

from dataclasses import dataclass
from typing import Any, Callable, ClassVar, cast

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


@dataclass(frozen=True)
class MetricInfo:
    expr: pl.Expr
    value_kind: str = "float"
    format: str | None = None
    unit: str | None = None
    extras: dict[str, pl.Expr] | None = None


class MetricRegistry:
    """
    Unified registry for all expression types used in metric evaluation.

    This is a singleton-style class that provides global registration
    and retrieval of error types, metrics, and summary expressions.
    """

    # Class-level registries
    _errors: dict[str, Callable[..., pl.Expr]] = {}
    _metrics: dict[str, MetricInfo | Callable[[], MetricInfo]] = {}
    _summaries: dict[str, pl.Expr | Callable[[], pl.Expr]] = {}
    _REGISTRY_ATTRS: ClassVar[dict[str, str]] = {
        "error": "_errors",
        "metric": "_metrics",
        "summary": "_summaries",
    }

    @classmethod
    def _registry_store(cls, kind: str) -> dict[str, Any]:
        try:
            attr_name = cls._REGISTRY_ATTRS[kind]
        except KeyError as exc:
            raise ValueError(f"Unknown registry kind: {kind}") from exc
        return getattr(cls, attr_name)

    @classmethod
    def _registry_names(cls, kind: str) -> list[str]:
        return list(cls._registry_store(kind).keys())

    @classmethod
    def _registry_contains(cls, kind: str, name: str) -> bool:
        return name in cls._registry_store(kind)

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
        if not cls._registry_contains("error", name):
            raise MetricNotFoundError(name, cls.list_errors(), "error")

        func = cls._registry_store("error")[name]
        return func(estimate, ground_truth, **params)

    @classmethod
    def generate_error_columns(
        cls,
        estimate: str,
        ground_truth: str,
        error_types: list[str] | None = None,
        error_params: dict[str, dict[str, Any]] | None = None,
    ) -> list[pl.Expr]:
        """Generate error column expressions for specified error types."""
        error_types = error_types or cls.list_errors()
        error_params = error_params or {}

        return [
            cls.get_error(
                error_type, estimate, ground_truth, **error_params.get(error_type, {})
            ).alias(error_type)
            for error_type in error_types
        ]

    @classmethod
    def list_errors(cls) -> list[str]:
        """List all available error types."""
        return cls._registry_names("error")

    @classmethod
    def list_metrics(cls) -> list[str]:
        """List all available metrics."""
        return cls._registry_names("metric")

    @classmethod
    def list_summaries(cls) -> list[str]:
        """List all available summaries."""
        return cls._registry_names("summary")

    @classmethod
    def has_metric(cls, name: str) -> bool:
        """Check if a metric exists in the registry."""
        return cls._registry_contains("metric", name)

    @classmethod
    def has_summary(cls, name: str) -> bool:
        """Check if a summary/selector exists in the registry."""
        return cls._registry_contains("summary", name)

    @classmethod
    def has_error(cls, name: str) -> bool:
        """Check if an error type exists in the registry."""
        return cls._registry_contains("error", name)

    # ============ Metric Expression Methods ============

    @classmethod
    def register_metric(
        cls,
        name: str,
        expr: pl.Expr | Callable[[], pl.Expr] | MetricInfo | Callable[[], MetricInfo],
        *,
        value_kind: str | None = None,
        format: str | None = None,
        unit: str | None = None,
        extras: dict[str, pl.Expr] | None = None,
    ) -> None:
        """
        Register a custom metric expression.

        Args:
            name: Name of the metric (e.g., 'mae', 'custom_accuracy')
            expr: Polars expression/MetricInfo or callable returning one.
                  Expressions reference error columns and define the aggregated result.
            value_kind: Optional type hint (``float``, ``int``, ``struct`` ...) used to
                populate the ``stat`` struct. Defaults to ``float``.
            format: Optional string formatter applied when rendering ``value``.
            unit: Optional unit label recorded in the ``stat`` struct.
            extras: Optional mapping of additional aggregations (each expression should be
                compatible with the aggregation context). Extras are surfaced through
                ``stat['value_struct']``.

        Example:
            MetricRegistry.register_metric('mae', pl.col('absolute_error').mean().alias('value'))
        """
        if isinstance(expr, MetricInfo):
            cls._metrics[name] = expr
            return

        if callable(expr):
            callable_expr: Callable[[], MetricInfo | pl.Expr] = cast(
                Callable[[], MetricInfo | pl.Expr], expr
            )

            def factory() -> MetricInfo:
                result = callable_expr()
                if isinstance(result, MetricInfo):
                    return result
                return MetricInfo(
                    expr=result,
                    value_kind=value_kind or "float",
                    format=format,
                    unit=unit,
                    extras=extras,
                )

            cls._metrics[name] = factory
            return

        info = MetricInfo(
            expr=expr,
            value_kind=value_kind or "float",
            format=format,
            unit=unit,
            extras=extras,
        )
        cls._metrics[name] = info

    @classmethod
    def register_summary(cls, name: str, expr: pl.Expr | Callable[[], pl.Expr]) -> None:
        """
        Register a custom summary expression.

        Args:
            name: Name of the summary (e.g., 'mean', 'p90')
            expr: Polars expression or callable that returns a Polars expression
                  The expression should typically operate on 'value' column

        Example:
            MetricRegistry.register_summary('p90', pl.col('value').quantile(0.9, interpolation="linear"))
        """
        cls._summaries[name] = expr

    @classmethod
    def get_metric(cls, name: str) -> MetricInfo:
        """
        Get a metric expression by name.

        Args:
            name: Name of the metric

        Returns:
            Polars expression for the metric

        Raises:
            ValueError: If the metric is not registered
        """
        if not cls._registry_contains("metric", name):
            raise MetricNotFoundError(name, cls.list_metrics(), "metric")

        expr = cls._registry_store("metric")[name]
        # If it's a callable, call it to get the expression/info
        if callable(expr):
            expr = expr()
        if not isinstance(expr, MetricInfo):
            expr = MetricInfo(expr=expr)
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
        if not cls._registry_contains("summary", name):
            raise MetricNotFoundError(name, cls.list_summaries(), "summary")

        summary_entry = cls._registry_store("summary")[name]
        result = summary_entry() if callable(summary_entry) else summary_entry
        if not isinstance(result, pl.Expr):
            raise TypeError(
                "Summary registry returned a non-expression object; "
                "ensure summaries resolve to pl.Expr."
            )
        return result

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

_DEFAULT_ERRORS: dict[str, Callable[..., pl.Expr]] = {
    "error": _error,
    "absolute_error": _absolute_error,
    "squared_error": _squared_error,
    "percent_error": _percent_error,
    "absolute_percent_error": _absolute_percent_error,
}

for _name, _func in _DEFAULT_ERRORS.items():
    MetricRegistry.register_error(_name, _func)

_DEFAULT_METRICS: tuple[tuple[str, pl.Expr, dict[str, Any]], ...] = (
    ("me", pl.col("error").mean(), {}),
    ("mae", pl.col("absolute_error").mean(), {}),
    ("mse", pl.col("squared_error").mean(), {}),
    ("rmse", pl.col("squared_error").mean().sqrt(), {}),
    ("mpe", pl.col("percent_error").mean(), {}),
    ("mape", pl.col("absolute_percent_error").mean(), {}),
    ("n_subject", pl.col("subject_id").n_unique(), {"value_kind": "int"}),
    (
        "n_visit",
        pl.struct(["subject_id", "visit_id"]).n_unique(),
        {"value_kind": "int"},
    ),
    (
        "n_sample",
        pl.col("sample_index").n_unique(),
        {"value_kind": "int"},
    ),
    (
        "n_subject_with_data",
        pl.col("subject_id").filter(pl.col("error").is_not_null()).n_unique(),
        {"value_kind": "int"},
    ),
    (
        "pct_subject_with_data",
        (
            pl.col("subject_id").filter(pl.col("error").is_not_null()).n_unique()
            / pl.col("subject_id").n_unique()
            * 100
        ).alias("value"),
        {},
    ),
    (
        "n_visit_with_data",
        pl.struct(["subject_id", "visit_id"])
        .filter(pl.col("error").is_not_null())
        .n_unique(),
        {"value_kind": "int"},
    ),
    (
        "pct_visit_with_data",
        (
            pl.struct(["subject_id", "visit_id"])
            .filter(pl.col("error").is_not_null())
            .n_unique()
            / pl.struct(["subject_id", "visit_id"]).n_unique()
            * 100
        ).alias("value"),
        {},
    ),
    (
        "n_sample_with_data",
        pl.col("error").is_not_null().sum(),
        {"value_kind": "int"},
    ),
    (
        "pct_sample_with_data",
        (pl.col("error").is_not_null().mean() * 100).alias("value"),
        {},
    ),
)

for _name, _expr, _options in _DEFAULT_METRICS:
    MetricRegistry.register_metric(_name, _expr, **_options)

_DEFAULT_SUMMARIES: dict[str, pl.Expr] = {
    "mean": pl.col("value").mean(),
    "median": pl.col("value").median(),
    "std": pl.col("value").std(),
    "min": pl.col("value").min(),
    "max": pl.col("value").max(),
    "sum": pl.col("value").sum(),
    "sqrt": pl.col("value").sqrt(),
}
_DEFAULT_SUMMARIES.update(
    {
        f"p{_p}": pl.col("value").quantile(_p / 100, interpolation="linear")
        for _p in (1, 5, 25, 75, 90, 95, 99)
    }
)

for _name, _expr in _DEFAULT_SUMMARIES.items():
    MetricRegistry.register_summary(_name, _expr)
