"""Metric compilation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from .metric_registry import ExpressionRegistry, MetricRegistry

if TYPE_CHECKING:  # pragma: no cover
    from .metric_define import MetricDefine


@dataclass(slots=True)
class CompiledMetric:
    """Immutable representation of compiled metric expressions."""

    definition: "MetricDefine"
    within_exprs: list[pl.Expr]
    across_expr: pl.Expr | None

    def as_tuple(self) -> tuple[list[pl.Expr], pl.Expr | None]:
        """Return the legacy tuple representation."""

        return self.within_exprs, self.across_expr


class MetricCompiler:
    """Compile :class:`MetricDefine` objects into Polars expressions."""

    def __init__(self, registry: ExpressionRegistry | None = None) -> None:
        self._registry = registry or MetricRegistry.get_registry()

    def compile(self, metric: "MetricDefine") -> CompiledMetric:
        within_exprs, across_expr = self._compile(metric)
        return CompiledMetric(metric, within_exprs, across_expr)

    # Internal helpers -----------------------------------------------------

    def _compile(self, metric: "MetricDefine") -> tuple[list[pl.Expr], pl.Expr | None]:
        has_custom = metric.within_expr is not None or metric.across_expr is not None
        if has_custom:
            result = self._compile_custom(metric)
        else:
            result = self._compile_builtin(metric)

        # Special handling for across-sample metrics: treat single aggregation as selection
        metric_type = metric.type.value if metric.type else None
        if metric_type == "across_sample":
            agg_exprs, sel_expr = result
            if agg_exprs and sel_expr is None:
                expr = agg_exprs[0]
                return [], expr

        return result

    def _compile_custom(
        self, metric: "MetricDefine"
    ) -> tuple[list[pl.Expr], pl.Expr | None]:
        within_exprs: list[pl.Expr] = []

        if metric.within_expr is not None:
            for item in metric.within_expr:
                if isinstance(item, str):
                    try:
                        within_exprs.append(self._registry.get_metric(item))
                    except ValueError as exc:  # pragma: no cover - error path
                        raise ValueError(
                            f"Unknown built-in metric in within_expr: {item}"
                        ) from exc
                else:
                    within_exprs.append(item)

        across_pl_expr: pl.Expr | None = None
        if metric.across_expr is not None:
            across_expr = metric.across_expr
            if isinstance(across_expr, str):
                try:
                    across_pl_expr = self._registry.get_summary(across_expr)
                except ValueError as exc:  # pragma: no cover - error path
                    raise ValueError(
                        f"Unknown built-in selector in across_expr: {across_expr}"
                    ) from exc
            else:
                across_pl_expr = across_expr

        if not within_exprs and across_pl_expr is not None:
            return [across_pl_expr], None

        return within_exprs, across_pl_expr

    def _compile_builtin(
        self, metric: "MetricDefine"
    ) -> tuple[list[pl.Expr], pl.Expr | None]:
        parts = (metric.name + ":").split(":")[:2]
        agg_name = parts[0]
        select_name = parts[1] if parts[1] else None

        try:
            agg_expr = self._registry.get_metric(agg_name)
        except ValueError as exc:
            raise ValueError(f"Unknown built-in metric: {agg_name}") from exc

        if select_name:
            try:
                select_expr = self._registry.get_summary(select_name)
            except ValueError as exc:
                raise ValueError(f"Unknown built-in selector: {select_name}") from exc
            return [agg_expr], select_expr

        return [], agg_expr
