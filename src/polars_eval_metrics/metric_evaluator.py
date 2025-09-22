from __future__ import annotations

"""
Unified Metric Evaluation Pipeline

This module implements a simplified, unified evaluation pipeline for computing metrics
using Polars LazyFrames with comprehensive support for scopes, groups, and subgroups.
"""

from collections.abc import Collection
from typing import TYPE_CHECKING, Any, Sequence

# pyre-strict

import polars as pl

if TYPE_CHECKING:
    from polars.selectors import Selector

from .ard import ARD
from .metric_define import MetricDefine, MetricScope, MetricType
from .metric_registry import MetricRegistry, MetricInfo


class MetricEvaluator:
    """Unified metric evaluation pipeline"""

    # Instance attributes with type annotations
    df_raw: pl.LazyFrame
    metrics: list[MetricDefine]
    ground_truth: str
    estimates: dict[str, str]  # Maps estimate names to display labels
    group_by: dict[str, str]  # Maps group column names to display labels
    subgroup_by: dict[str, str]  # Maps subgroup column names to display labels
    filter_expr: pl.Expr | None
    error_params: dict[str, dict[str, Any]]
    df: pl.LazyFrame
    _evaluation_cache: dict[tuple[tuple[str, ...], tuple[str, ...]], ARD]
    _estimate_keys: list[str]
    _estimate_label_lookup: dict[str, str]
    _estimate_label_reverse: dict[str, str]
    _estimate_label_order: dict[str, int]
    _estimate_key_order: dict[str, int]
    _metric_label_order: dict[str, int]
    _metric_name_order: dict[str, int]
    _subgroup_categories: list[str]

    def __init__(
        self,
        df: pl.DataFrame | pl.LazyFrame,
        metrics: MetricDefine | list[MetricDefine],
        ground_truth: str = "actual",
        estimates: str | list[str] | dict[str, str] | None = None,
        group_by: list[str] | dict[str, str] | None = None,
        subgroup_by: list[str] | dict[str, str] | None = None,
        filter_expr: pl.Expr | None = None,
        error_params: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Initialize evaluator with complete evaluation context

        Args:
            df: Input data as DataFrame or LazyFrame
            metrics: Metric definitions to evaluate
            ground_truth: Column name containing ground truth values
            estimates: Estimate column names. Can be:
                - str: Single column name
                - list[str]: List of column names
                - dict[str, str]: Mapping from column names to display labels
            group_by: Columns to group by for analysis. Can be:
                - list[str]: List of column names
                - dict[str, str]: Mapping from column names to display labels
            subgroup_by: Columns for subgroup analysis. Can be:
                - list[str]: List of column names
                - dict[str, str]: Mapping from column names to display labels
            filter_expr: Optional filter expression
            error_params: Parameters for error calculations
        """
        # Store data as LazyFrame
        self.df_raw = df.lazy() if isinstance(df, pl.DataFrame) else df

        # Normalize inputs to lists
        self.metrics = [metrics] if isinstance(metrics, MetricDefine) else metrics
        self.ground_truth = ground_truth

        # Process inputs using dedicated methods
        self.estimates = self._process_estimates(estimates)
        # Preserve insertion order for deterministic display and sorting
        self._estimate_keys = list(self.estimates.keys())
        self._estimate_label_lookup = dict(self.estimates)
        self._estimate_label_reverse = {
            label: key for key, label in self._estimate_label_lookup.items()
        }
        self._estimate_label_order = {
            label: idx for idx, label in enumerate(self._estimate_label_lookup.values())
        }
        self._estimate_key_order = {
            key: idx for idx, key in enumerate(self._estimate_keys)
        }
        self._metric_label_order = {
            metric.label or metric.name: idx for idx, metric in enumerate(self.metrics)
        }
        self._metric_name_order = {
            metric.name: idx for idx, metric in enumerate(self.metrics)
        }
        self.group_by = self._process_grouping(group_by)
        self.subgroup_by = self._process_grouping(subgroup_by)
        self._subgroup_categories = self._compute_subgroup_categories()
        self.filter_expr = filter_expr
        self.error_params = error_params or {}

        # Apply base filter once
        self.df = self._apply_base_filter()

        # Initialize evaluation cache
        self._evaluation_cache = {}

    def _apply_base_filter(self) -> pl.LazyFrame:
        """Apply initial filter if provided"""
        if self.filter_expr is not None:
            return self.df_raw.filter(self.filter_expr)
        return self.df_raw

    def _get_cache_key(
        self,
        metrics: MetricDefine | list[MetricDefine] | None,
        estimates: str | list[str] | None,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Generate cache key for evaluation parameters"""
        target_metrics = self._resolve_metrics(metrics)
        target_estimates = self._resolve_estimates(estimates)

        # Create hashable key from metric names and estimates
        metric_names = tuple(sorted(m.name for m in target_metrics))
        estimate_names = tuple(sorted(target_estimates))

        return (metric_names, estimate_names)

    def _get_cached_evaluation(
        self,
        metrics: MetricDefine | list[MetricDefine] | None = None,
        estimates: str | list[str] | None = None,
    ) -> ARD:
        """Get cached evaluation result or compute and cache if not exists"""
        cache_key = self._get_cache_key(metrics, estimates)

        if cache_key not in self._evaluation_cache:
            if metrics is not None or estimates is not None:
                filtered_evaluator = self.filter(metrics=metrics, estimates=estimates)
                ard_result = filtered_evaluator._evaluate_ard(
                    metrics=metrics, estimates=estimates
                )
            else:
                ard_result = self._evaluate_ard(metrics=metrics, estimates=estimates)
            self._evaluation_cache[cache_key] = ard_result

        return self._evaluation_cache[cache_key]

    def clear_cache(self) -> None:
        """Clear the evaluation cache"""
        self._evaluation_cache.clear()

    def filter(
        self,
        *,
        metrics: MetricDefine | list[MetricDefine] | None = None,
        estimates: str | list[str] | None = None,
    ) -> "MetricEvaluator":
        """Return a new evaluator scoped to the requested metrics or estimates."""

        filtered_metrics = (
            self._resolve_metrics(metrics) if metrics is not None else self.metrics
        )
        filtered_estimate_keys = (
            self._resolve_estimates(estimates)
            if estimates is not None
            else list(self.estimates.keys())
        )
        filtered_estimates = {
            key: self.estimates[key] for key in filtered_estimate_keys
        }

        return MetricEvaluator(
            df=self.df,
            metrics=filtered_metrics,
            ground_truth=self.ground_truth,
            estimates=filtered_estimates,
            group_by=self.group_by,
            subgroup_by=self.subgroup_by,
            filter_expr=None,
            error_params=self.error_params,
        )

    def _evaluate_ard(
        self,
        metrics: MetricDefine | list[MetricDefine] | None = None,
        estimates: str | list[str] | None = None,
    ) -> ARD:
        """Internal helper that returns evaluation results as ARD."""

        target_metrics = self._resolve_metrics(metrics)
        target_estimates = self._resolve_estimates(estimates)

        if not target_metrics or not target_estimates:
            raise ValueError("No metrics or estimates to evaluate")

        combined = self._vectorized_evaluate(target_metrics, target_estimates)
        formatted = self._format_result(combined)
        return self._convert_to_ard(formatted)

    def evaluate(
        self,
        metrics: MetricDefine | list[MetricDefine] | None = None,
        estimates: str | list[str] | None = None,
        *,
        collect: bool = True,
    ) -> ARD | pl.LazyFrame | "EvaluationResult":
        """
        Unified evaluation method returning ARD format.

        Args:
            metrics: Subset of metrics to evaluate (None = use all configured)
            estimates: Subset of estimates to evaluate (None = use all configured)
            collect: When False, return a ``LazyFrame`` rather than an ``ARD`` instance

        Returns:
            :class:`EvaluationResult` (subclass of ``polars.DataFrame``), or a ``LazyFrame``
            when ``collect`` is False.
        """

        ard = self._evaluate_ard(metrics=metrics, estimates=estimates)

        if not collect:
            return ard.lazy

        return EvaluationResult(ard)

    def _convert_to_ard(self, result_lf: pl.LazyFrame) -> ARD:
        """Convert the evaluator output into the canonical ARD columns lazily."""

        schema = result_lf.collect_schema()

        # Groups -----------------------------------------------------------------
        group_cols = [col for col in self.group_by.keys() if col in schema]
        if group_cols:
            group_struct_dtype = pl.Struct(
                [pl.Field(col, schema[col]) for col in group_cols]
            )
            groups_expr = (
                pl.when(
                    pl.all_horizontal([pl.col(col).is_null() for col in group_cols])
                )
                .then(pl.lit(None, dtype=group_struct_dtype))
                .otherwise(pl.struct([pl.col(col).alias(col) for col in group_cols]))
                .alias("groups")
            )
        else:
            groups_expr = pl.lit(None).alias("groups")

        # Subgroups ---------------------------------------------------------------
        subgroup_labels = list(self.subgroup_by.values()) if self.subgroup_by else []
        has_subgroup_columns = "subgroup_name" in schema and "subgroup_value" in schema
        if subgroup_labels and has_subgroup_columns:
            subgroup_struct_dtype = pl.Struct(
                [pl.Field(label, pl.Utf8) for label in subgroup_labels]
            )
            subgroup_fields = [
                pl.when(pl.col("subgroup_name") == pl.lit(label))
                .then(pl.col("subgroup_value").cast(pl.Utf8))
                .otherwise(pl.lit(None, dtype=pl.Utf8))
                .alias(label)
                for label in subgroup_labels
            ]
            subgroups_expr = (
                pl.when(
                    pl.col("subgroup_name").is_null()
                    | pl.col("subgroup_value").is_null()
                )
                .then(pl.lit(None, dtype=subgroup_struct_dtype))
                .otherwise(pl.struct(subgroup_fields))
                .alias("subgroups")
            )
        else:
            subgroups_expr = pl.lit(None, dtype=pl.Struct([])).alias("subgroups")

        # Stat --------------------------------------------------------------------
        ard_frame = result_lf.with_columns(
            [
                self._expr_groups(schema),
                self._expr_subgroups(schema),
                self._expr_estimate(schema),
                self._expr_metric_enum(),
                self._expr_label_enum(),
                self._expr_stat_struct(schema),
                self._expr_context_struct(schema),
            ]
        )

        cleanup_cols = [
            "_value_kind",
            "_value_format",
            "_value_unit",
            "_value_float",
            "_value_int",
            "_value_bool",
            "_value_str",
            "_value_json",
            "_value_struct",
            "_vector_payload",
        ]
        schema_names = set(schema.names())
        drop_cols = [col for col in cleanup_cols if col in schema_names]
        if drop_cols:
            ard_frame = ard_frame.drop(drop_cols)

        return ARD(ard_frame)

    def pivot_by_group(
        self,
        metrics: MetricDefine | list[MetricDefine] | None = None,
        estimates: str | list[str] | None = None,
        column_order_by: str = "metrics",
        row_order_by: str = "group",
    ) -> pl.DataFrame:
        """
        Pivot results with groups as rows and model x metric as columns.

        Args:
            metrics: Subset of metrics to evaluate (None = use all configured)
            estimates: Subset of estimates to evaluate (None = use all configured)
            column_order_by: Column ordering strategy ("metrics" or "estimates")
            row_order_by: Row ordering strategy ("group" or "subgroup")

        Returns:
            DataFrame with group combinations as rows and metric columns
        """
        long_df = self._collect_long_dataframe(metrics=metrics, estimates=estimates)

        group_cols = [
            label for label in self.group_by.values() if label in long_df.columns
        ]
        subgroup_present = (
            "subgroup_name" in long_df.columns and "subgroup_value" in long_df.columns
        )

        index_cols: list[str]
        if row_order_by == "subgroup" and subgroup_present:
            index_cols = ["subgroup_name", "subgroup_value"] + group_cols
        else:
            index_cols = group_cols + (
                ["subgroup_name", "subgroup_value"] if subgroup_present else []
            )

        def pivot_default(df: pl.DataFrame) -> pl.DataFrame:
            default_df = df.filter(pl.col("scope").is_null())
            if default_df.is_empty():
                return (
                    pl.DataFrame({col: [] for col in index_cols})
                    if index_cols
                    else pl.DataFrame()
                )

            display_col = (
                "estimate_label"
                if "estimate_label" in default_df.columns
                else "estimate"
            )
            on_cols = [display_col, "label"]

            if index_cols:
                pivoted = default_df.pivot(
                    index=index_cols,
                    on=on_cols,
                    values="value",
                    aggregate_function="first",
                )
            else:
                with_idx = default_df.with_row_index("_idx")
                pivoted = with_idx.pivot(
                    index=["_idx"],
                    on=on_cols,
                    values="value",
                    aggregate_function="first",
                ).drop("_idx")

            return pivoted

        def pivot_scope(
            df: pl.DataFrame, scope_value: str, columns: list[str]
        ) -> tuple[pl.DataFrame, list[str]]:
            scoped = df.filter(pl.col("scope") == scope_value)
            if scoped.is_empty():
                return pl.DataFrame(), []

            if index_cols:
                pivoted = scoped.pivot(
                    index=index_cols,
                    on=columns,
                    values="value",
                    aggregate_function="first",
                )
            else:
                with_idx = scoped.with_row_index("_idx")
                pivoted = with_idx.pivot(
                    index=["_idx"],
                    on=columns,
                    values="value",
                    aggregate_function="first",
                ).drop("_idx")

            value_cols = [col for col in pivoted.columns if col not in index_cols]
            return pivoted, value_cols

        result = pivot_default(long_df)

        group_pivot, group_cols_created = pivot_scope(long_df, "group", ["label"])
        if not group_pivot.is_empty():
            if result.is_empty():
                result = group_pivot
            else:
                result = result.join(group_pivot, on=index_cols, how="left")

        global_pivot, global_cols_created = pivot_scope(long_df, "global", ["label"])
        if not global_pivot.is_empty():
            if not index_cols:
                if result.is_empty():
                    result = global_pivot
                else:
                    result = pl.concat([result, global_pivot], how="horizontal")
            elif index_cols and global_pivot.height == 1:
                broadcast_values = {
                    col: global_pivot[col][0] for col in global_cols_created
                }
                result = result.with_columns(
                    [
                        pl.lit(broadcast_values[col]).alias(col)
                        for col in broadcast_values
                    ]
                )
            elif result.is_empty():
                result = global_pivot
            else:
                result = result.join(global_pivot, on=index_cols, how="left")

        if result.is_empty():
            if index_cols:
                return pl.DataFrame({col: [] for col in index_cols})
            return pl.DataFrame()

        # Reorder columns: index -> global -> group -> default
        value_cols = [col for col in result.columns if col not in index_cols]
        default_cols = (
            [col for col in value_cols if col.startswith('{"') and col.endswith('"}')]
            if value_cols
            else []
        )

        estimate_order_lookup: dict[str, int] = self._estimate_label_order
        metric_label_order_lookup: dict[str, int] = self._metric_label_order
        metric_name_order_lookup: dict[str, int] = self._metric_name_order

        def metric_order(label: str) -> int:
            if label in metric_label_order_lookup:
                return metric_label_order_lookup[label]
            return metric_name_order_lookup.get(label, len(metric_label_order_lookup))

        def estimate_order(label: str) -> int:
            return estimate_order_lookup.get(label, len(estimate_order_lookup))

        def sort_default(columns: list[str]) -> list[str]:
            def parse(col: str) -> tuple[str, str]:
                inner = col[2:-2]
                parts = inner.split('","')
                return (parts[0], parts[1]) if len(parts) == 2 else (col, "")

            if column_order_by == "metrics":
                return sorted(
                    columns,
                    key=lambda c: (
                        metric_order(parse(c)[1]),
                        estimate_order(parse(c)[0]),
                    ),
                )
            return sorted(
                columns,
                key=lambda c: (
                    estimate_order(parse(c)[0]),
                    metric_order(parse(c)[1]),
                ),
            )

        ordered = (
            index_cols
            + global_cols_created
            + group_cols_created
            + sort_default(default_cols)
        )

        remaining = [col for col in value_cols if col not in ordered]
        ordered.extend(remaining)

        ordered = [col for col in ordered if col in result.columns]

        if "subgroup_value" in result.columns:
            if self._subgroup_categories and all(
                isinstance(cat, str) for cat in self._subgroup_categories
            ):
                result = result.with_columns(
                    pl.col("subgroup_value").cast(pl.Enum(self._subgroup_categories))
                )
            else:
                result = result.with_columns(pl.col("subgroup_value").cast(pl.Utf8))

        sort_columns: list[str] = []
        temp_sort_columns: list[str] = []
        subgroup_order_map = {
            label: idx for idx, label in enumerate(self.subgroup_by.values())
        }

        if row_order_by == "group":
            sort_columns.extend([col for col in group_cols if col in result.columns])
            if "subgroup_name" in result.columns and self.subgroup_by:
                result = result.with_columns(
                    pl.col("subgroup_name")
                    .replace(subgroup_order_map)
                    .fill_null(len(subgroup_order_map))
                    .cast(pl.Int32)
                    .alias("__subgroup_name_order")
                )
                sort_columns.append("__subgroup_name_order")
                temp_sort_columns.append("__subgroup_name_order")
            if "subgroup_value" in result.columns:
                sort_columns.append("subgroup_value")
        else:
            if "subgroup_name" in result.columns and self.subgroup_by:
                result = result.with_columns(
                    pl.col("subgroup_name")
                    .replace(subgroup_order_map)
                    .fill_null(len(subgroup_order_map))
                    .cast(pl.Int32)
                    .alias("__subgroup_name_order")
                )
                sort_columns.append("__subgroup_name_order")
                temp_sort_columns.append("__subgroup_name_order")
            if "subgroup_value" in result.columns:
                sort_columns.append("subgroup_value")
            sort_columns.extend([col for col in group_cols if col in result.columns])

        if sort_columns:
            result = result.sort(sort_columns)

        if temp_sort_columns:
            result = result.drop(temp_sort_columns)

        seen: set[str] = set()
        deduped: list[str] = []
        for col in ordered:
            if col not in seen:
                deduped.append(col)
                seen.add(col)

        result = result.select(deduped)

        return result

    def pivot_by_model(
        self,
        metrics: MetricDefine | list[MetricDefine] | None = None,
        estimates: str | list[str] | None = None,
        column_order_by: str = "estimates",
        row_order_by: str = "group",
    ) -> pl.DataFrame:
        """
        Pivot results with models as rows and group x metric as columns.

        Args:
            metrics: Subset of metrics to evaluate (None = use all configured)
            estimates: Subset of estimates to evaluate (None = use all configured)
            column_order_by: Column ordering strategy ("estimates" or "metrics")
            row_order_by: Row ordering strategy ("group" or "subgroup")

        Returns:
            DataFrame with model combinations as rows and group+metric columns
        """
        long_df = self._collect_long_dataframe(metrics=metrics, estimates=estimates)

        subgroup_present = (
            "subgroup_name" in long_df.columns and "subgroup_value" in long_df.columns
        )

        index_cols: list[str]
        if row_order_by == "subgroup" and subgroup_present:
            index_cols = ["estimate", "subgroup_name", "subgroup_value"]
        else:
            index_cols = ["estimate"] + (
                ["subgroup_name", "subgroup_value"] if subgroup_present else []
            )

        if "estimate" in index_cols:
            estimate_series = (
                long_df.get_column("estimate")
                if "estimate" in long_df.columns
                else None
            )
            if estimate_series is None or estimate_series.is_null().all():
                index_cols = [col for col in index_cols if col != "estimate"]

        default_cols_created: list[str] = []
        global_cols_created: list[str] = []
        group_cols_created: list[str] = []

        default_df = long_df.filter(pl.col("scope").is_null())
        if default_df.is_empty():
            default_pivot = (
                pl.DataFrame({col: [] for col in index_cols})
                if index_cols
                else pl.DataFrame()
            )
        else:
            on_cols = [label for label in self.group_by.values()] + ["label"]
            if index_cols:
                default_pivot = default_df.pivot(
                    index=index_cols,
                    on=on_cols,
                    values="value",
                    aggregate_function="first",
                )
            else:
                default_pivot = (
                    default_df.with_row_index("_idx")
                    .pivot(
                        index=["_idx"],
                        on=on_cols,
                        values="value",
                        aggregate_function="first",
                    )
                    .drop("_idx")
                )
            default_cols_created = [
                col for col in default_pivot.columns if col not in index_cols
            ]

        global_df = long_df.filter(pl.col("scope") == "global")
        if not global_df.is_empty():
            if index_cols:
                global_pivot = global_df.pivot(
                    index=index_cols,
                    on=["label"],
                    values="value",
                    aggregate_function="first",
                )
            else:
                global_pivot = (
                    global_df.with_row_index("_idx")
                    .pivot(
                        index=["_idx"],
                        on=["label"],
                        values="value",
                        aggregate_function="first",
                    )
                    .drop("_idx")
                )
            global_cols_created = [
                col for col in global_pivot.columns if col not in index_cols
            ]
            if not index_cols:
                if default_pivot.is_empty():
                    default_pivot = global_pivot
                else:
                    default_pivot = pl.concat(
                        [default_pivot, global_pivot], how="horizontal"
                    )
            elif default_pivot.is_empty():
                default_pivot = global_pivot
            else:
                default_pivot = default_pivot.join(
                    global_pivot, on=index_cols, how="left"
                )

        group_df = long_df.filter(pl.col("scope") == "group")
        if not group_df.is_empty():
            if index_cols:
                group_pivot = group_df.pivot(
                    index=index_cols,
                    on=[*self.group_by.values(), "label"],
                    values="value",
                    aggregate_function="first",
                )
            else:
                group_pivot = (
                    group_df.with_row_index("_idx")
                    .pivot(
                        index=["_idx"],
                        on=[*self.group_by.values(), "label"],
                        values="value",
                        aggregate_function="first",
                    )
                    .drop("_idx")
                )
            group_cols_created = [
                col for col in group_pivot.columns if col not in index_cols
            ]
            if not index_cols:
                if default_pivot.is_empty():
                    default_pivot = group_pivot
                else:
                    default_pivot = pl.concat(
                        [default_pivot, group_pivot], how="horizontal"
                    )
            elif default_pivot.is_empty():
                default_pivot = group_pivot
            else:
                default_pivot = default_pivot.join(
                    group_pivot, on=index_cols, how="left"
                )

        if "estimate" in default_pivot.columns:
            default_pivot = default_pivot.with_columns(
                pl.col("estimate")
                .map_elements(
                    lambda val, mapping=self._estimate_label_lookup: mapping.get(
                        val, val
                    ),
                    return_dtype=pl.Utf8,
                )
                .alias("estimate_label")
            )

        if "subgroup_value" in default_pivot.columns:
            if self._subgroup_categories and all(
                isinstance(cat, str) for cat in self._subgroup_categories
            ):
                default_pivot = default_pivot.with_columns(
                    pl.col("subgroup_value").cast(pl.Enum(self._subgroup_categories))
                )
            else:
                default_pivot = default_pivot.with_columns(
                    pl.col("subgroup_value").cast(pl.Utf8)
                )

        ordered = (
            [col for col in index_cols if col in default_pivot.columns]
            + [col for col in global_cols_created if col in default_pivot.columns]
            + [col for col in group_cols_created if col in default_pivot.columns]
            + [col for col in default_cols_created if col in default_pivot.columns]
        )

        remaining = [col for col in default_pivot.columns if col not in ordered]
        ordered.extend(remaining)

        default_pivot = default_pivot.select(ordered)

        sort_columns: list[str] = []
        if "subgroup_value" in default_pivot.columns:
            sort_columns.append("subgroup_value")
        if "estimate" in default_pivot.columns:
            sort_columns.append("estimate")
        for label in self.group_by.values():
            if label in default_pivot.columns:
                sort_columns.append(label)
        if sort_columns:
            default_pivot = default_pivot.sort(sort_columns)

        return default_pivot

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
            return list(self.estimates.keys())

        estimates_list = [estimates] if isinstance(estimates, str) else estimates

        for e in estimates_list:
            if e not in self.estimates:
                raise ValueError(
                    f"Estimate '{e}' not in configured estimates: {list(self.estimates.keys())}"
                )

        return estimates_list

    def _vectorized_evaluate(
        self, metrics: list[MetricDefine], estimates: list[str]
    ) -> pl.LazyFrame:
        """Vectorized evaluation using single Polars group_by operations"""

        # Step 1: Prepare data in long format with all estimates
        df_long = self._prepare_long_format_data(estimates)

        # Step 2: Generate all error columns for the melted data
        df_with_errors = self._add_error_columns_vectorized(df_long)

        # Step 3: Handle marginal subgroup analysis if needed
        if self.subgroup_by:
            return self._evaluate_with_marginal_subgroups(
                df_with_errors, metrics, estimates
            )
        else:
            return self._evaluate_without_subgroups(df_with_errors, metrics, estimates)

    def _evaluate_without_subgroups(
        self,
        df_with_errors: pl.LazyFrame,
        metrics: list[MetricDefine],
        estimates: list[str],
    ) -> pl.LazyFrame:
        """Evaluate metrics without subgroup analysis"""
        results = []
        for metric in metrics:
            metric_result = self._evaluate_metric_vectorized(
                df_with_errors, metric, estimates
            )
            results.append(metric_result)

        # Combine results (no schema harmonization needed with fixed evaluation structure)
        if results:
            return pl.concat(results, how="diagonal")
        else:
            return pl.DataFrame().lazy()

    def _evaluate_with_marginal_subgroups(
        self,
        df_with_errors: pl.LazyFrame,
        metrics: list[MetricDefine],
        estimates: list[str],
    ) -> pl.LazyFrame:
        """Evaluate metrics with marginal subgroup analysis using vectorized operations"""
        original_subgroup_by = self.subgroup_by

        # Create all subgroup combinations using vectorized unpivot
        subgroup_data = self._prepare_subgroup_data_vectorized(
            df_with_errors, original_subgroup_by
        )

        # Evaluate all metrics across all subgroups in a vectorized manner
        results = []
        for metric in metrics:
            # Temporarily clear subgroup_by to use the vectorized subgroup columns instead
            self.subgroup_by = {}
            try:
                metric_result = self._evaluate_metric_vectorized(
                    subgroup_data, metric, estimates
                )
                results.append(metric_result)
            finally:
                self.subgroup_by = original_subgroup_by

        # Combine results (no schema harmonization needed with fixed evaluation structure)
        return pl.concat(results, how="diagonal")

    def _prepare_subgroup_data_vectorized(
        self, df_with_errors: pl.LazyFrame, subgroup_by: dict[str, str]
    ) -> pl.LazyFrame:
        """Prepare subgroup data using vectorized unpivot operations"""
        schema_names = df_with_errors.collect_schema().names()
        subgroup_cols = list(subgroup_by.keys())
        id_vars = [col for col in schema_names if col not in subgroup_cols]

        # Use unpivot to create marginal subgroup analysis
        return df_with_errors.unpivot(
            index=id_vars,
            on=subgroup_cols,
            variable_name="subgroup_name",
            value_name="subgroup_value",
        ).with_columns(
            [
                # Replace subgroup column names with their display labels
                pl.col("subgroup_name").replace(subgroup_by)
            ]
        )

    def _prepare_long_format_data(self, estimates: list[str]) -> pl.LazyFrame:
        """Reshape data from wide to long format for vectorized processing"""

        # Add a row index to the original data to uniquely identify each sample
        # This must be done BEFORE unpivoting to avoid double counting
        df_with_index = self.df.with_row_index("sample_index")

        # Get all columns except estimates to preserve in melt
        schema_names = df_with_index.collect_schema().names()
        id_vars = [col for col in schema_names if col not in estimates]

        # Unpivot estimates into long format
        df_long = df_with_index.unpivot(
            index=id_vars,
            on=estimates,
            variable_name="estimate_name",
            value_name="estimate_value",
        )

        # Preserve canonical estimate key alongside optional display label
        df_long = (
            df_long.rename({self.ground_truth: "ground_truth"})
            .with_columns(
                [
                    pl.col("estimate_name").alias("estimate"),
                    pl.col("estimate_name")
                    .cast(pl.Utf8)
                    .map_elements(
                        lambda val, mapping=self._estimate_label_lookup: mapping.get(
                            val, val
                        ),
                        return_dtype=pl.Utf8,
                    )
                    .alias("estimate_label"),
                ]
            )
            .drop("estimate_name")
        )

        return df_long

    def _add_error_columns_vectorized(self, df_long: pl.LazyFrame) -> pl.LazyFrame:
        """Add error columns for the long-format data"""

        # Generate error expressions for the vectorized format
        # Use 'estimate_value' as the estimate column and 'ground_truth' as the renamed ground truth column
        error_expressions = MetricRegistry.generate_error_columns(
            estimate="estimate_value",
            ground_truth="ground_truth",
            error_types=None,
            error_params=self.error_params,
        )

        return df_long.with_columns(error_expressions)

    def _evaluate_metric_vectorized(
        self, df_with_errors: pl.LazyFrame, metric: MetricDefine, estimates: list[str]
    ) -> pl.LazyFrame:
        """Evaluate a single metric using vectorized operations"""

        # Determine grouping columns based on metric scope
        group_cols = self._get_vectorized_grouping_columns(metric, df_with_errors)

        # Compile metric expressions
        within_infos, across_info = metric.compile_expressions()

        # Apply metric-specific filtering if needed
        df_filtered = self._apply_metric_scope_filter(df_with_errors, metric, estimates)

        # Perform the evaluation with appropriate grouping
        if metric.type == MetricType.ACROSS_SAMPLE:
            if across_info is None:
                raise ValueError(
                    f"ACROSS_SAMPLE metric {metric.name} requires across_expr"
                )

            result_info = across_info
            agg_exprs = self._metric_agg_expressions(result_info)
            if group_cols:
                result = df_filtered.group_by(group_cols).agg(agg_exprs)
            else:
                result = df_filtered.select(*agg_exprs)

        elif metric.type in [MetricType.WITHIN_SUBJECT, MetricType.WITHIN_VISIT]:
            # Within-entity aggregation
            entity_groups = self._get_entity_grouping_columns(metric.type) + group_cols

            # Use the appropriate expression for within-entity aggregation
            if within_infos:
                result_info = within_infos[0]
            elif across_info is not None:
                result_info = across_info
            else:
                raise ValueError(f"No valid expression for metric {metric.name}")

            agg_exprs = self._metric_agg_expressions(result_info)
            result = df_filtered.group_by(entity_groups).agg(agg_exprs)

        elif metric.type in [MetricType.ACROSS_SUBJECT, MetricType.ACROSS_VISIT]:
            # Two-level aggregation: within entities, then across
            entity_groups = self._get_entity_grouping_columns(metric.type) + group_cols

            # First level: within entities
            if within_infos:
                base_info = within_infos[0]
            elif across_info is not None:
                base_info = across_info
            else:
                raise ValueError(
                    f"No valid expression for first level of metric {metric.name}"
                )

            intermediate = df_filtered.group_by(entity_groups).agg(
                self._metric_agg_expressions(base_info, include_extras=False)
            )

            # Second level: across entities
            if across_info is not None and within_infos:
                result_info = across_info
                agg_exprs = self._metric_agg_expressions(result_info)
                if group_cols:
                    result = intermediate.group_by(group_cols).agg(agg_exprs)
                else:
                    result = intermediate.select(*agg_exprs)
            else:
                result_info = base_info
                agg_exprs = [pl.col("value").mean().alias("value")]
                if group_cols:
                    result = intermediate.group_by(group_cols).agg(agg_exprs)
                else:
                    result = intermediate.select(*agg_exprs)

        else:
            raise ValueError(f"Unknown metric type: {metric.type}")

        # Add metadata columns and bundle entity-level vectors when needed
        result_with_metadata = self._add_metadata_vectorized(
            result, metric, result_info
        )
        return self._bundle_entity_vectors(result_with_metadata, metric)

    def _get_vectorized_grouping_columns(
        self, metric: MetricDefine, df: pl.LazyFrame | None = None
    ) -> list[str]:
        """Get grouping columns for vectorized evaluation based on metric scope"""
        group_cols = []

        # Check if we're in vectorized subgroup mode (data has subgroup_name/subgroup_value columns)
        if df is not None:
            schema_names = df.collect_schema().names()
            using_vectorized_subgroups = (
                "subgroup_name" in schema_names and "subgroup_value" in schema_names
            )
        else:
            using_vectorized_subgroups = False

        # Handle scope-based grouping
        if metric.scope == MetricScope.GLOBAL:
            if using_vectorized_subgroups:
                group_cols.extend(["subgroup_name", "subgroup_value"])
            else:
                group_cols.extend(list(self.subgroup_by.keys()))

        elif metric.scope == MetricScope.MODEL:
            group_cols.append("estimate")
            if using_vectorized_subgroups:
                group_cols.extend(["subgroup_name", "subgroup_value"])
            else:
                group_cols.extend(list(self.subgroup_by.keys()))

        elif metric.scope == MetricScope.GROUP:
            group_cols.extend(list(self.group_by.keys()))
            if using_vectorized_subgroups:
                group_cols.extend(["subgroup_name", "subgroup_value"])
            else:
                group_cols.extend(list(self.subgroup_by.keys()))

        else:
            # Default: estimate + groups + subgroups
            group_cols.append("estimate")
            group_cols.extend(list(self.group_by.keys()))
            if using_vectorized_subgroups:
                group_cols.extend(["subgroup_name", "subgroup_value"])
            else:
                group_cols.extend(list(self.subgroup_by.keys()))

        return group_cols

    def _apply_metric_scope_filter(
        self, df: pl.LazyFrame, metric: MetricDefine, estimates: list[str]
    ) -> pl.LazyFrame:
        """Apply any scope-specific filtering"""
        # For now, no additional filtering needed beyond grouping
        # Future: could add estimate filtering for specific scopes
        _ = metric, estimates  # Suppress unused parameter warnings
        return df

    @staticmethod
    def _metric_agg_expressions(
        info: MetricInfo, *, include_extras: bool = True
    ) -> list[pl.Expr]:
        expressions = [info.expr.alias("value")]
        if include_extras and info.extras:
            for name, expr in info.extras.items():
                expressions.append(expr.alias(f"_extra_{name}"))
        return expressions

    def _get_entity_grouping_columns(self, metric_type: MetricType) -> list[str]:
        """Get entity-level grouping columns (subject_id, visit_id)"""
        if metric_type in [MetricType.WITHIN_SUBJECT, MetricType.ACROSS_SUBJECT]:
            return ["subject_id"]
        elif metric_type in [MetricType.WITHIN_VISIT, MetricType.ACROSS_VISIT]:
            return ["subject_id", "visit_id"]
        else:
            return []

    def _add_metadata_vectorized(
        self, result: pl.LazyFrame, metric: MetricDefine, info: MetricInfo
    ) -> pl.LazyFrame:
        """Add metadata columns to vectorized result"""

        value_kind = (info.value_kind or "float").lower()

        metadata_columns = [
            pl.lit(metric.name).cast(pl.Utf8).alias("metric"),
            pl.lit(metric.label).cast(pl.Utf8).alias("label"),
            pl.lit(metric.type.value).cast(pl.Utf8).alias("metric_type"),
            pl.lit(metric.scope.value if metric.scope else None)
            .cast(pl.Utf8)
            .alias("scope"),
            pl.lit(value_kind).cast(pl.Utf8).alias("_value_kind"),
            pl.lit(info.format).cast(pl.Utf8).alias("_value_format"),
            pl.lit(info.unit).cast(pl.Utf8).alias("_value_unit"),
        ]

        result = result.with_columns(metadata_columns)

        # Canonical helper columns for stat construction
        result = result.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("_value_float"),
            pl.lit(None, dtype=pl.Int64).alias("_value_int"),
            pl.lit(None, dtype=pl.Boolean).alias("_value_bool"),
            pl.lit(None, dtype=pl.Utf8).alias("_value_str"),
            pl.lit(None, dtype=pl.Utf8).alias("_value_json"),
            pl.lit(None, dtype=pl.Struct([])).alias("_value_struct"),
        )

        # Normalise raw value column and populate helper fields
        if value_kind == "int":
            result = result.with_columns(
                pl.col("value").round(0).cast(pl.Int64).alias("_value_int"),
                pl.col("value").cast(pl.Float64).alias("_value_float"),
            )
            result = result.with_columns(pl.col("value").cast(pl.Float64))
        elif value_kind == "float":
            result = result.with_columns(
                pl.col("value").cast(pl.Float64).alias("_value_float"),
                pl.col("value").cast(pl.Float64),
            )
        elif value_kind == "bool":
            result = result.with_columns(
                pl.col("value").cast(pl.Boolean).alias("_value_bool"),
                pl.lit(None, dtype=pl.Float64).alias("value"),
            )
        elif value_kind == "string":
            result = result.with_columns(
                pl.col("value").cast(pl.Utf8).alias("_value_str"),
                pl.lit(None, dtype=pl.Float64).alias("value"),
            )
        elif value_kind == "json":
            result = result.with_columns(
                pl.col("value").cast(pl.Utf8).alias("_value_json"),
                pl.lit(None, dtype=pl.Float64).alias("value"),
            )
        elif value_kind == "struct":
            result = result.with_columns(
                pl.col("value").alias("_value_struct"),
                pl.lit(None, dtype=pl.Float64).alias("value"),
            )

        # Handle extras as structured payloads
        extra_cols: list[str] = []
        if info.extras:
            extra_cols = [f"_extra_{name}" for name in info.extras.keys()]
            struct_fields = [
                pl.col(col_name).alias(col_name.removeprefix("_extra_"))
                for col_name in extra_cols
            ]
            result = result.with_columns(
                pl.struct(struct_fields).alias("_value_struct")
            )
        if not info.extras:
            # Ensure consistent schema by keeping already-initialised Null struct
            result = result.with_columns(pl.col("_value_struct"))

        if extra_cols:
            result = result.drop(extra_cols)

        return result

    def _bundle_entity_vectors(
        self, result: pl.LazyFrame, metric: MetricDefine
    ) -> pl.LazyFrame:
        """Bundle entity-level results into vector payloads for within metrics."""

        entity_cols = self._get_entity_grouping_columns(metric.type)
        if not entity_cols:
            return result

        schema = result.collect_schema()
        schema_names = set(schema.names())
        if any(col not in schema_names for col in entity_cols):
            return result

        helper_value = "__value_list"
        helper_indices = {col: f"__index_{col}" for col in entity_cols}

        helper_columns = {
            "_value_float",
            "_value_int",
            "_value_bool",
            "_value_str",
            "_value_json",
            "_value_struct",
        }
        grouping_cols = [
            col
            for col in schema.names()
            if col not in {"value", *entity_cols} and col not in helper_columns
        ]

        base = result.sort(entity_cols)
        aggregate_exprs: list[pl.Expr] = [pl.col("value").alias(helper_value)]
        aggregate_exprs.extend(
            pl.col(col).alias(alias) for col, alias in helper_indices.items()
        )

        if grouping_cols:
            aggregated = base.group_by(grouping_cols).agg(aggregate_exprs)
        else:
            aggregated = (
                base.with_columns(pl.lit(0).alias("__group_marker"))
                .group_by("__group_marker")
                .agg(aggregate_exprs)
                .drop("__group_marker")
            )

        vector_struct = pl.struct(
            [
                pl.col(helper_value).alias("values"),
                pl.struct(
                    [pl.col(alias).alias(col) for col, alias in helper_indices.items()]
                ).alias("index"),
            ]
        )

        aggregated = aggregated.with_columns(
            [
                vector_struct.alias("_vector_payload"),
                pl.lit(None, dtype=pl.Float64).alias("value"),
            ]
        )
        aggregated = aggregated.drop([helper_value, *helper_indices.values()])

        aggregated = aggregated.with_columns(pl.lit("vector").alias("_value_kind"))

        return aggregated

    def _format_result(self, combined: pl.LazyFrame) -> pl.LazyFrame:
        """Minimal formatting - ARD handles all presentation concerns"""
        return combined

    # ------------------------------------------------------------------
    # Result shaping helpers
    # ------------------------------------------------------------------

    def _collect_long_dataframe(
        self,
        metrics: MetricDefine | list[MetricDefine] | None = None,
        estimates: str | list[str] | None = None,
    ) -> pl.DataFrame:
        """Collect evaluation results as a flat DataFrame for pivoting."""

        ard = self._get_cached_evaluation(metrics=metrics, estimates=estimates)
        lf = ard.lazy
        schema = lf.collect_schema()

        exprs: list[pl.Expr] = []

        # Group columns with display labels
        for col, label in self.group_by.items():
            if col in schema.names():
                exprs.append(pl.col(col).alias(label))

        # Subgroup columns
        if "subgroup_name" in schema.names():
            exprs.append(pl.col("subgroup_name"))
        if "subgroup_value" in schema.names():
            exprs.append(pl.col("subgroup_value"))

        # Estimate / metric / label columns
        if "estimate" in schema.names():
            exprs.append(pl.col("estimate").cast(pl.Utf8))
            exprs.append(
                pl.col("estimate")
                .cast(pl.Utf8)
                .map_elements(
                    lambda val, mapping=self._estimate_label_lookup: mapping.get(
                        val, val
                    ),
                    return_dtype=pl.Utf8,
                )
                .alias("estimate_label")
            )
        if "metric" in schema.names():
            exprs.append(pl.col("metric").cast(pl.Utf8))
        if "label" in schema.names():
            exprs.append(pl.col("label").cast(pl.Utf8))
        else:
            exprs.append(pl.col("metric").cast(pl.Utf8).alias("label"))

        # Formatted value and raw components sourced from ARD stat struct
        if "stat" in schema.names():
            exprs.append(pl.col("stat"))
            exprs.append(
                pl.col("stat")
                .map_elements(ARD._format_stat, return_dtype=pl.Utf8)
                .alias("value")
            )
        elif "value" in schema.names():
            # Legacy fallback when stat struct is unavailable
            exprs.append(
                pl.col("value")
                .cast(pl.Float64)
                .map_elements(
                    lambda v: "NULL" if v is None else f"{float(v):.1f}",
                    return_dtype=pl.Utf8,
                )
                .alias("value")
            )

        # Scope metadata
        if "metric_type" in schema.names():
            exprs.append(pl.col("metric_type").cast(pl.Utf8))
        if "scope" in schema.names():
            exprs.append(pl.col("scope").cast(pl.Utf8))

        return lf.select(exprs).collect()

    def _expr_groups(self, schema: pl.Schema) -> pl.Expr:
        group_cols = [col for col in self.group_by.keys() if col in schema.names()]
        if not group_cols:
            return pl.lit(None).alias("groups")

        dtype = pl.Struct([pl.Field(col, schema[col]) for col in group_cols])
        return (
            pl.when(pl.all_horizontal([pl.col(col).is_null() for col in group_cols]))
            .then(pl.lit(None, dtype=dtype))
            .otherwise(pl.struct([pl.col(col).alias(col) for col in group_cols]))
            .alias("groups")
        )

    def _expr_subgroups(self, schema: pl.Schema) -> pl.Expr:
        if (
            not self.subgroup_by
            or "subgroup_name" not in schema.names()
            or "subgroup_value" not in schema.names()
        ):
            return pl.lit(None).alias("subgroups")

        labels = list(self.subgroup_by.values())
        dtype = pl.Struct([pl.Field(label, pl.Utf8) for label in labels])
        fields = [
            pl.when(pl.col("subgroup_name") == pl.lit(label))
            .then(pl.col("subgroup_value").cast(pl.Utf8))
            .otherwise(pl.lit(None, dtype=pl.Utf8))
            .alias(label)
            for label in labels
        ]
        return (
            pl.when(
                pl.col("subgroup_name").is_null() | pl.col("subgroup_value").is_null()
            )
            .then(pl.lit(None, dtype=dtype))
            .otherwise(pl.struct(fields))
            .alias("subgroups")
        )

    def _expr_stat_struct(self, schema: pl.Schema) -> pl.Expr:
        null_utf8: pl.Expr = pl.lit(None, dtype=pl.Utf8)
        null_float: pl.Expr = pl.lit(None, dtype=pl.Float64)
        null_int: pl.Expr = pl.lit(None, dtype=pl.Int64)
        null_bool: pl.Expr = pl.lit(None, dtype=pl.Boolean)
        null_struct_expr: pl.Expr = pl.lit(None, dtype=pl.Struct([]))
        null_vector: pl.Expr = pl.lit(None, dtype=pl.List(pl.Float64))
        null_index: pl.Expr = pl.lit(None, dtype=pl.List(pl.Utf8))

        kind_expr: pl.Expr | None = (
            pl.col("_value_kind") if "_value_kind" in schema.names() else None
        )
        format_col: pl.Expr = (
            pl.col("_value_format") if "_value_format" in schema.names() else null_utf8
        )
        unit_col: pl.Expr = (
            pl.col("_value_unit") if "_value_unit" in schema.names() else null_utf8
        )

        value_col: pl.Expr = (
            pl.col("value") if "value" in schema.names() else null_float
        )

        float_col: pl.Expr = (
            pl.col("_value_float")
            if "_value_float" in schema.names()
            else value_col.cast(pl.Float64)
        )
        int_col: pl.Expr = (
            pl.col("_value_int")
            if "_value_int" in schema.names()
            else value_col.round(0).cast(pl.Int64)
        )
        bool_col: pl.Expr = (
            pl.col("_value_bool")
            if "_value_bool" in schema.names()
            else value_col.cast(pl.Boolean)
        )
        str_col: pl.Expr = (
            pl.col("_value_str")
            if "_value_str" in schema.names()
            else value_col.cast(pl.Utf8)
        )
        json_col: pl.Expr = (
            pl.col("_value_json")
            if "_value_json" in schema.names()
            else value_col.cast(pl.Utf8)
        )
        struct_col: pl.Expr = (
            pl.col("_value_struct")
            if "_value_struct" in schema.names()
            else null_struct_expr
        )

        has_vector_payload = "_vector_payload" in schema.names()
        vector_payload_col: pl.Expr = (
            pl.col("_vector_payload")
            if has_vector_payload
            else pl.lit(None, dtype=pl.Struct([]))
        )

        payload_dtype = schema.get("_vector_payload") if has_vector_payload else None
        index_field_names: tuple[str, ...] = tuple()
        if isinstance(payload_dtype, pl.Struct):
            for field in payload_dtype.fields:
                if field.name == "index" and isinstance(field.dtype, pl.Struct):
                    index_field_names = tuple(
                        subfield.name for subfield in field.dtype.fields
                    )
                    break

        if has_vector_payload:
            values_expr: pl.Expr = (
                pl.when(vector_payload_col.is_null())
                .then(null_vector)
                .otherwise(
                    vector_payload_col.struct.field("values").cast(
                        pl.List(pl.Float64), strict=False
                    )
                )
            )

            def encode_index(
                payload: Any, field_order: tuple[str, ...] | None = index_field_names
            ) -> list[str] | None:
                order = field_order if field_order else None
                return ARD._encode_index_vector(payload, order)

            index_expr: pl.Expr = (
                pl.when(vector_payload_col.is_null())
                .then(null_index)
                .otherwise(
                    vector_payload_col.struct.field("index").map_elements(
                        encode_index,
                        return_dtype=pl.List(pl.Utf8),
                    )
                )
            )
        else:
            values_expr = null_vector
            index_expr = null_index

        def build_struct(
            type_label: str,
            *,
            value_float: pl.Expr = null_float,
            value_int: pl.Expr = null_int,
            value_bool: pl.Expr = null_bool,
            value_str: pl.Expr = null_utf8,
            value_json: pl.Expr = null_utf8,
            value_struct_expr: pl.Expr | None = None,
            value_vector_expr: pl.Expr = null_vector,
            index_vector_expr: pl.Expr = null_index,
        ) -> pl.Expr:
            struct_expr = (
                value_struct_expr if value_struct_expr is not None else struct_col
            )
            vector_expr = value_vector_expr
            index_vector = index_vector_expr
            if type_label == "float":
                type_is_null = value_float.is_null()
            elif type_label == "int":
                type_is_null = value_int.is_null()
            elif type_label == "bool":
                type_is_null = value_bool.is_null()
            elif type_label == "string":
                type_is_null = value_str.is_null()
            elif type_label == "json":
                type_is_null = value_json.is_null()
            elif type_label == "struct":
                type_is_null = struct_expr.is_null()
            elif type_label == "vector":
                type_is_null = vector_expr.is_null()
            else:
                type_is_null = value_json.is_null()

            return pl.struct(
                [
                    pl.when(type_is_null)
                    .then(null_utf8)
                    .otherwise(pl.lit(type_label))
                    .alias("type"),
                    value_float.alias("value_float"),
                    value_int.alias("value_int"),
                    value_bool.alias("value_bool"),
                    value_str.alias("value_str"),
                    value_json.alias("value_json"),
                    struct_expr.alias("value_struct"),
                    vector_expr.alias("value_vector"),
                    index_vector.alias("index_vector"),
                    format_col.alias("format"),
                    unit_col.alias("unit"),
                ]
            )

        float_struct = build_struct("float", value_float=float_col)
        int_struct = build_struct("int", value_int=int_col)
        bool_struct = build_struct("bool", value_bool=bool_col)
        string_struct = build_struct("string", value_str=str_col)
        json_struct = build_struct("json", value_json=json_col)
        struct_struct = build_struct("struct", value_struct_expr=struct_col)
        vector_struct = build_struct(
            "vector", value_vector_expr=values_expr, index_vector_expr=index_expr
        )

        if kind_expr is not None:
            return (
                pl.when(kind_expr == "vector")
                .then(vector_struct)
                .when(kind_expr == "struct")
                .then(struct_struct)
                .when(kind_expr == "float")
                .then(float_struct)
                .when(kind_expr == "int")
                .then(int_struct)
                .when(kind_expr == "bool")
                .then(bool_struct)
                .when(kind_expr == "string")
                .then(string_struct)
                .when(kind_expr == "json")
                .then(json_struct)
                .otherwise(json_struct)
            ).alias("stat")

        if has_vector_payload:
            return vector_struct.alias("stat")

        value_dtype = schema.get("value", pl.Float64)
        if value_dtype in (pl.Float32, pl.Float64):
            return float_struct.alias("stat")
        if value_dtype in {
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
        }:
            return int_struct.alias("stat")
        if value_dtype == pl.Boolean:
            return bool_struct.alias("stat")
        if value_dtype == pl.Utf8:
            return string_struct.alias("stat")

        value_struct_dtype = schema.get("_value_struct")
        if isinstance(value_struct_dtype, pl.Struct):
            return struct_struct.alias("stat")

        return json_struct.alias("stat")

    def _expr_context_struct(self, schema: pl.Schema) -> pl.Expr:
        null_utf8 = pl.lit(None, dtype=pl.Utf8)
        fields = []
        for field in ("metric_type", "scope", "label"):
            if field in schema.names():
                fields.append(pl.col(field).cast(pl.Utf8).alias(field))
            else:
                fields.append(null_utf8.alias(field))
        if "estimate" in schema.names():
            fields.append(
                pl.col("estimate")
                .cast(pl.Utf8)
                .map_elements(
                    lambda val, mapping=self._estimate_label_lookup: mapping.get(
                        val, val
                    ),
                    return_dtype=pl.Utf8,
                )
                .alias("estimate_label")
            )
        else:
            fields.append(null_utf8.alias("estimate_label"))
        return pl.struct(fields).alias("context")

    def _expr_estimate(self, schema: pl.Schema) -> pl.Expr:
        null_utf8 = pl.lit(None, dtype=pl.Utf8)
        if "estimate" not in schema.names():
            return null_utf8.alias("estimate")

        estimate_names = list(self.estimates.keys())
        if estimate_names:
            return (
                pl.col("estimate")
                .cast(pl.Utf8)
                .replace({name: name for name in estimate_names})
                .cast(pl.Enum(estimate_names))
                .alias("estimate")
            )

        return pl.col("estimate").cast(pl.Utf8).alias("estimate")

    def _expr_metric_enum(self) -> pl.Expr:
        metric_categories = list(dict.fromkeys(metric.name for metric in self.metrics))
        return (
            pl.col("metric")
            .cast(pl.Utf8)
            .replace({name: name for name in metric_categories})
            .cast(pl.Enum(metric_categories))
            .alias("metric")
        )

    def _expr_label_enum(self) -> pl.Expr:
        label_categories = [metric.label or metric.name for metric in self.metrics]
        unique_labels = list(dict.fromkeys(label_categories))
        return pl.col("label").cast(pl.Enum(unique_labels)).alias("label")

    # ========================================
    # INPUT PROCESSING METHODS - Pure Logic
    # ========================================

    @staticmethod
    def _process_estimates(
        estimates: str | list[str] | dict[str, str] | None,
    ) -> dict[str, str]:
        """Pure transformation: normalize estimates to dict format"""
        if isinstance(estimates, str):
            return {estimates: estimates}
        elif isinstance(estimates, dict):
            return estimates
        elif isinstance(estimates, list):
            return {est: est for est in (estimates or [])}
        else:
            return {}

    @staticmethod
    def _process_grouping(
        grouping: list[str] | dict[str, str] | None,
    ) -> dict[str, str]:
        """Pure transformation: normalize grouping to dict format"""
        if isinstance(grouping, dict):
            return grouping
        elif isinstance(grouping, list):
            return {col: col for col in (grouping or [])}
        else:
            return {}

    def _compute_subgroup_categories(self) -> list[str]:
        if not self.subgroup_by:
            return []

        categories: list[str] = []
        seen: set[str] = set()
        schema = self.df_raw.collect_schema()

        for column in self.subgroup_by.keys():
            if column not in schema.names():
                continue

            dtype = schema[column]

            if isinstance(dtype, pl.Enum):
                for value in dtype.categories.to_list():
                    if value not in seen:
                        seen.add(value)
                        categories.append(value)
                continue

            series = self.df_raw.select(pl.col(column)).collect()[column]
            non_null = [value for value in series.to_list() if value is not None]

            if not non_null:
                continue

            if dtype.is_numeric():
                ordered_values = sorted(set(non_null))
            else:
                ordered_values = sorted({str(value) for value in non_null})

            for value in ordered_values:
                if value in seen:
                    continue
                seen.add(value)
                categories.append(value)

        return categories

    # ========================================
    # VALIDATION METHODS - Centralized Logic
    # ========================================

    def _validate_inputs(self) -> None:
        """Validate all inputs after processing"""
        if not self.estimates:
            raise ValueError("No estimates provided")

        if not self.metrics:
            raise ValueError("No metrics provided")

        # Validate that required columns exist
        schema_names = self.df_raw.collect_schema().names()

        if self.ground_truth not in schema_names:
            raise ValueError(
                f"Ground truth column '{self.ground_truth}' not found in data"
            )

        missing_estimates = [
            est for est in self.estimates.keys() if est not in schema_names
        ]
        if missing_estimates:
            raise ValueError(f"Estimate columns not found in data: {missing_estimates}")

        missing_groups = [
            col for col in self.group_by.keys() if col not in schema_names
        ]
        if missing_groups:
            raise ValueError(f"Group columns not found in data: {missing_groups}")

        missing_subgroups = [
            col for col in self.subgroup_by.keys() if col not in schema_names
        ]
        if missing_subgroups:
            raise ValueError(f"Subgroup columns not found in data: {missing_subgroups}")


class EvaluationResult(pl.DataFrame):
    """DataFrame-like wrapper around ARD results with convenience accessors."""

    def __init__(self, ard: ARD) -> None:
        long_df = ard.to_long()

        group_sort_cols = list(ard._group_fields)
        subgroup_struct_cols = list(ard._subgroup_fields)
        sort_cols: list[str] = []

        for col in group_sort_cols:
            if col in long_df.columns:
                sort_cols.append(col)

        for col in ("subgroup_name", "subgroup_value"):
            if col in long_df.columns:
                sort_cols.append(col)

        for col in subgroup_struct_cols:
            if col in long_df.columns and col not in sort_cols:
                sort_cols.append(col)

        for col in ("metric", "estimate"):
            if col in long_df.columns:
                sort_cols.append(col)

        if sort_cols:
            long_df = long_df.sort(sort_cols)

        super().__init__(long_df)
        self._ard = ard

    def collect(self) -> pl.DataFrame:
        """Return the canonical ARD table with struct columns."""
        return self._ard.collect()

    def to_ard(self) -> ARD:
        """Expose the underlying ARD object."""
        return self._ard

    def to_long(self) -> pl.DataFrame:
        """Return a copy of the flattened DataFrame representation."""
        return pl.DataFrame(self)

    def unnest(
        self,
        columns: str | "Selector" | Collection[str | "Selector"],
        *args: Any,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """Delegate unnest operations to the structured ARD representation."""
        df = self._ard.collect()
        if isinstance(columns, str):
            cols: list[str | "Selector"] = [columns]
        elif isinstance(columns, Collection):
            cols = list(columns)
        else:
            cols = [columns]
        safe_cols: list[str | "Selector"] = []
        schema = df.schema
        for col in cols:
            if isinstance(col, str):
                dtype = schema.get(col)
                if dtype is None or dtype == pl.Null:
                    continue
            safe_cols.append(col)

        if safe_cols:
            return df.unnest(safe_cols, *args, **kwargs)
        return df
