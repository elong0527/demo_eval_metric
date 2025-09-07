"""
Unified Metric Evaluation Pipeline

This module implements a simplified, unified evaluation pipeline for computing metrics
using Polars LazyFrames with comprehensive support for scopes, groups, and subgroups.
"""

from typing import Any
from dataclasses import dataclass

import polars as pl

from .metric_define import MetricDefine, MetricScope, MetricType
from .metric_registry import MetricRegistry


@dataclass
class EvaluationCombination:
    """Represents a single evaluation combination"""

    estimates: list[str]
    groups: dict[str, Any]
    subgroups: dict[str, Any]


class MetricEvaluator:
    """Unified metric evaluation pipeline"""

    # Instance attributes with type annotations
    df_raw: pl.LazyFrame
    metrics: list[MetricDefine]
    ground_truth: str
    estimates: list[str]
    group_by: list[str]
    subgroup_by: list[str]
    filter_expr: pl.Expr | None
    error_params: dict[str, dict[str, Any]]
    df: pl.LazyFrame

    def __init__(
        self,
        df: pl.DataFrame | pl.LazyFrame,
        metrics: MetricDefine | list[MetricDefine],
        ground_truth: str = "actual",
        estimates: str | list[str] | None = None,
        group_by: list[str] | None = None,
        subgroup_by: list[str] | None = None,
        filter_expr: pl.Expr | None = None,
        error_params: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Initialize evaluator with complete evaluation context"""
        # Store data as LazyFrame
        self.df_raw = df.lazy() if isinstance(df, pl.DataFrame) else df

        # Normalize inputs to lists
        self.metrics = [metrics] if isinstance(metrics, MetricDefine) else metrics
        self.ground_truth = ground_truth
        self.estimates = (
            [estimates] if isinstance(estimates, str) else (estimates or [])
        )
        self.group_by = group_by or []
        self.subgroup_by = subgroup_by or []
        self.filter_expr = filter_expr
        self.error_params = error_params or {}

        # Apply base filter once
        self.df = self._apply_base_filter()

    def _apply_base_filter(self) -> pl.LazyFrame:
        """Apply initial filter if provided"""
        if self.filter_expr is not None:
            return self.df_raw.filter(self.filter_expr)
        return self.df_raw

    def evaluate(
        self,
        metrics: MetricDefine | list[MetricDefine] | None = None,
        estimates: str | list[str] | None = None,
        collect: bool = True,
    ) -> pl.DataFrame | pl.LazyFrame:
        """
        Unified evaluation method for all metrics and combinations

        Args:
            metrics: Subset of metrics to evaluate (None = use all configured)
            estimates: Subset of estimates to evaluate (None = use all configured)
            collect: Whether to collect result to DataFrame

        Returns:
            Evaluation results
        """
        # Determine evaluation targets
        target_metrics = self._resolve_metrics(metrics)
        target_estimates = self._resolve_estimates(estimates)

        if not target_metrics or not target_estimates:
            raise ValueError("No metrics or estimates to evaluate")

        # Generate and evaluate all combinations
        results = []
        for metric in target_metrics:
            for combo in self._get_evaluation_combinations(metric, target_estimates):
                result = self._evaluate_combination(metric, combo)
                results.append(result)

        # Combine results
        combined = pl.concat(results, how="diagonal")

        # Format final result
        formatted = self._format_result(combined)

        return formatted.collect() if collect else formatted

    def _resolve_metrics(
        self, metrics: MetricDefine | list[MetricDefine] | None
    ) -> list[MetricDefine]:
        """Resolve which metrics to evaluate"""
        if metrics is None:
            return self.metrics

        metrics_list = [metrics] if isinstance(metrics, MetricDefine) else metrics
        configured_names = {m.name for m in self.metrics}

        for m in metrics_list:
            if m.name not in configured_names:
                raise ValueError(f"Metric '{m.name}' not in configured metrics")

        return metrics_list

    def _resolve_estimates(self, estimates: str | list[str] | None) -> list[str]:
        """Resolve which estimates to evaluate"""
        if estimates is None:
            return self.estimates

        estimates_list = [estimates] if isinstance(estimates, str) else estimates

        for e in estimates_list:
            if e not in self.estimates:
                raise ValueError(
                    f"Estimate '{e}' not in configured estimates: {self.estimates}"
                )

        return estimates_list

    def _get_evaluation_combinations(
        self, metric: MetricDefine, estimates: list[str]
    ) -> list[EvaluationCombination]:
        """
        Generate all evaluation combinations for a metric based on its scope
        """
        combinations = []

        # Get all group combinations (including empty if no groups)
        group_combinations = self._get_group_combinations()

        # Get all subgroup combinations
        subgroup_combinations = self._get_subgroup_combinations()

        if metric.scope == MetricScope.GLOBAL:
            # Single evaluation across all estimates, groups, and subgroups
            for subgroup_combo in subgroup_combinations:
                combinations.append(
                    EvaluationCombination(
                        estimates=estimates,
                        groups={},  # Global ignores groups
                        subgroups=subgroup_combo,
                    )
                )

        elif metric.scope == MetricScope.MODEL:
            # Per model, ignoring groups
            for estimate in estimates:
                for subgroup_combo in subgroup_combinations:
                    combinations.append(
                        EvaluationCombination(
                            estimates=[estimate],
                            groups={},  # Model scope ignores groups
                            subgroups=subgroup_combo,
                        )
                    )

        elif metric.scope == MetricScope.GROUP:
            # Per group, aggregating across models
            for group_combo in group_combinations:
                for subgroup_combo in subgroup_combinations:
                    combinations.append(
                        EvaluationCombination(
                            estimates=estimates,  # All estimates together
                            groups=group_combo,
                            subgroups=subgroup_combo,
                        )
                    )

        else:  # Default scope (None)
            # Cartesian product: estimate × group × subgroup
            for estimate in estimates:
                for group_combo in group_combinations:
                    for subgroup_combo in subgroup_combinations:
                        combinations.append(
                            EvaluationCombination(
                                estimates=[estimate],
                                groups=group_combo,
                                subgroups=subgroup_combo,
                            )
                        )

        return combinations

    def _get_group_combinations(self) -> list[dict[str, Any]]:
        """Get all unique combinations of group_by column values"""
        if not self.group_by:
            return [{}]  # Single empty combination

        # Get unique combinations of group values
        unique_groups = self.df.select(self.group_by).unique().collect()

        combinations = []
        for row in unique_groups.iter_rows(named=True):
            combinations.append(dict(row))

        return combinations

    def _get_subgroup_combinations(self) -> list[dict[str, Any]]:
        """Get marginal subgroup combinations - each subgroup variable analyzed separately"""
        if not self.subgroup_by:
            return [{}]  # Single empty combination

        combinations = []

        # For marginal analysis, generate one combination per subgroup variable per value
        for subgroup_col in self.subgroup_by:
            unique_values = self.df.select(subgroup_col).unique().collect()

            for row in unique_values.iter_rows(named=True):
                # Create combination with only this subgroup variable
                combinations.append({subgroup_col: row[subgroup_col]})

        return combinations

    def _evaluate_combination(
        self, metric: MetricDefine, combo: EvaluationCombination
    ) -> pl.LazyFrame:
        """Evaluate a single metric-combination pair"""

        # Filter data for this combination
        filtered_df = self._filter_for_combination(combo)

        # Prepare error columns for all estimates in this combination
        df_with_errors = self._prepare_error_columns(filtered_df, combo.estimates)

        # Compile metric expressions
        within_exprs, across_expr = metric.compile_expressions()

        # Determine grouping columns based on metric type and combination
        groups = self._get_grouping_for_combination(metric.type, combo)

        # Build evaluation pipeline
        result = self._build_pipeline(df_with_errors, within_exprs, across_expr, groups)

        # Add metadata
        return self._add_metadata(result, metric, combo)

    def _filter_for_combination(self, combo: EvaluationCombination) -> pl.LazyFrame:
        """Filter data for a specific combination"""
        df = self.df

        # Apply group filters
        for col, value in combo.groups.items():
            df = df.filter(pl.col(col) == value)

        # Apply subgroup filters
        for col, value in combo.subgroups.items():
            df = df.filter(pl.col(col) == value)

        return df

    def _prepare_error_columns(
        self, df: pl.LazyFrame, estimates: list[str]
    ) -> pl.LazyFrame:
        """Add error columns for all estimates in the combination"""
        df_with_errors = df

        for estimate in estimates:
            error_expressions = MetricRegistry.generate_error_columns(
                estimate=estimate,
                ground_truth=self.ground_truth,
                error_types=None,
                error_params=self.error_params,
            )
            df_with_errors = df_with_errors.with_columns(error_expressions)

        return df_with_errors

    def _get_grouping_for_combination(
        self, metric_type: MetricType, combo: EvaluationCombination
    ) -> tuple[list[str] | None, list[str] | None]:
        """Determine grouping columns for this metric type and combination"""

        # Base grouping columns (groups that should remain in final result)
        result_groups = list(combo.groups.keys()) + list(combo.subgroups.keys())

        # Aggregation grouping rules based on metric type
        if metric_type == MetricType.ACROSS_SAMPLES:
            return (None, result_groups if result_groups else None)

        elif metric_type == MetricType.WITHIN_SUBJECT:
            agg_groups = ["subject_id"] + result_groups
            return (agg_groups, None)

        elif metric_type == MetricType.ACROSS_SUBJECT:
            agg_groups = ["subject_id"] + result_groups
            return (agg_groups, result_groups if result_groups else None)

        elif metric_type == MetricType.WITHIN_VISIT:
            agg_groups = ["subject_id", "visit_id"] + result_groups
            return (agg_groups, None)

        elif metric_type == MetricType.ACROSS_VISIT:
            agg_groups = ["subject_id", "visit_id"] + result_groups
            return (agg_groups, result_groups if result_groups else None)

        else:
            raise ValueError(f"Unknown metric type: {metric_type}")

    def _build_pipeline(
        self,
        df: pl.LazyFrame,
        within_exprs: list[pl.Expr],
        across_expr: pl.Expr | None,
        groups: tuple[list[str] | None, list[str] | None],
    ) -> pl.LazyFrame:
        """Build evaluation pipeline"""
        agg_groups, select_groups = groups
        pipeline = df

        # First-level aggregation
        if agg_groups is not None and within_exprs:
            if agg_groups:
                pipeline = pipeline.group_by(agg_groups).agg(within_exprs)
            else:
                pipeline = pipeline.select(within_exprs)

        # Second-level aggregation or final selection
        if select_groups is not None:
            if across_expr is not None:
                value_expr = across_expr.alias("value").cast(pl.Float64)
            elif within_exprs:
                value_expr = within_exprs[0].alias("value").cast(pl.Float64)
            else:
                raise ValueError("No expression available")

            if select_groups:
                pipeline = pipeline.group_by(select_groups).agg(value_expr)
            else:
                pipeline = pipeline.select(value_expr)

        elif across_expr is not None:
            # Direct application of across expression
            if agg_groups:
                pipeline = pipeline.group_by(agg_groups).agg(
                    across_expr.alias("value").cast(pl.Float64)
                )
            else:
                pipeline = pipeline.select(across_expr.alias("value").cast(pl.Float64))

        return pipeline

    def _add_metadata(
        self, result: pl.LazyFrame, metric: MetricDefine, combo: EvaluationCombination
    ) -> pl.LazyFrame:
        """Add metadata columns to result"""

        # Determine estimate label
        if len(combo.estimates) == 1:
            estimate_label = combo.estimates[0]
        else:
            estimate_label = None  # Multiple estimates aggregated - use null

        metadata = [
            pl.lit(metric.name).alias("metric"),
            pl.lit(metric.label or metric.name).alias("label"),
            pl.lit(metric.type.value).alias("metric_type"),
        ]

        # Only add estimate column when it's meaningful (single estimate)
        if estimate_label is not None:
            metadata.insert(1, pl.lit(estimate_label).alias("estimate"))

        # Add group metadata
        for col, value in combo.groups.items():
            metadata.append(pl.lit(value).alias(col))

        # Add subgroup metadata
        if combo.subgroups:
            # For subgroups, we add special columns to indicate the subgroup analysis
            subgroup_names = list(combo.subgroups.keys())
            subgroup_values = list(combo.subgroups.values())

            if len(subgroup_names) == 1:
                metadata.extend(
                    [
                        pl.lit(subgroup_names[0]).alias("subgroup_name"),
                        pl.lit(str(subgroup_values[0])).alias("subgroup_value"),
                    ]
                )
            else:
                # Multiple subgroups - create combined name/value
                combined_name = "|".join(subgroup_names)
                combined_value = "|".join(str(v) for v in subgroup_values)
                metadata.extend(
                    [
                        pl.lit(combined_name).alias("subgroup_name"),
                        pl.lit(combined_value).alias("subgroup_value"),
                    ]
                )

        return result.with_columns(metadata)

    def _format_result(self, combined: pl.LazyFrame) -> pl.LazyFrame:
        """Format final result with proper column ordering and sorting"""

        # Determine available columns
        try:
            available_columns = combined.collect_schema().names()
        except Exception:
            available_columns = combined.limit(1).collect().columns

        # Define column order
        column_order = []

        # ID columns
        for col in ["subject_id", "visit_id"]:
            if col in available_columns:
                column_order.append(col)

        # Group columns
        for col in self.group_by:
            if col in available_columns and col not in column_order:
                column_order.append(col)

        # Subgroup columns
        if "subgroup_name" in available_columns:
            column_order.extend(["subgroup_name", "subgroup_value"])

        # Core result columns - only include estimate if it exists
        core_columns = []
        if "estimate" in available_columns:
            core_columns.append("estimate")
        core_columns.extend(["metric", "label", "value", "metric_type"])
        column_order.extend(core_columns)

        # Sort columns
        sort_cols = []
        potential_sort_cols = self.group_by + ["subgroup_name", "metric"]
        if "estimate" in available_columns:
            potential_sort_cols.append("estimate")

        for col in potential_sort_cols:
            if col in available_columns:
                sort_cols.append(col)

        # Apply formatting
        result = combined
        if sort_cols:
            result = result.sort(sort_cols)

        # Select columns in order (only those that exist)
        existing_order = [col for col in column_order if col in available_columns]
        result = result.select(existing_order)

        return result
