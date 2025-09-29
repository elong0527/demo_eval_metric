from __future__ import annotations

"""
Unified Metric Evaluation Pipeline

This module implements a simplified, unified evaluation pipeline for computing metrics
using Polars LazyFrames with comprehensive support for scopes, groups, and subgroups.
"""

from collections.abc import Iterable, Mapping
from typing import Any, Sequence

# pyre-strict

import polars as pl

from .ard import ARD
from .evaluation_context import EstimateCatalog, FormatterContext, MetricCatalog
from .metric_define import MetricDefine, MetricScope, MetricType
from .metric_registry import MetricRegistry, MetricInfo
from .result_formatter import (
    build_group_pivot,
    build_model_pivot,
    format_compact_frame,
    format_verbose_frame,
    convert_to_ard as format_to_ard,
)


class MetricEvaluator:
    """Unified metric evaluation pipeline"""

    # Instance attributes with type annotations
    df_raw: pl.LazyFrame
    ground_truth: str
    group_by: dict[str, str]  # Maps group column names to display labels
    subgroup_by: dict[str, str]  # Maps subgroup column names to display labels
    filter_expr: pl.Expr | None
    error_params: dict[str, dict[str, Any]]
    df: pl.LazyFrame
    _evaluation_cache: dict[tuple[tuple[str, ...], tuple[str, ...]], ARD]
    _metric_catalog: MetricCatalog
    _estimate_catalog: EstimateCatalog
    _formatter_context: FormatterContext
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

        metric_list = [metrics] if isinstance(metrics, MetricDefine) else list(metrics)
        self._metric_catalog = MetricCatalog(tuple(metric_list))
        self.ground_truth = ground_truth

        # Process inputs using dedicated methods
        self._estimate_catalog = EstimateCatalog.build(
            self._process_estimates(estimates)
        )
        self.group_by = self._process_grouping(group_by)
        self.subgroup_by = self._process_grouping(subgroup_by)
        self._subgroup_categories = self._compute_subgroup_categories()
        self._formatter_context = FormatterContext(
            group_by=self.group_by,
            subgroup_by=self.subgroup_by,
            estimate_catalog=self._estimate_catalog,
            metric_catalog=self._metric_catalog,
            subgroup_categories=tuple(self._subgroup_categories),
        )
        self.filter_expr = filter_expr
        self.error_params = error_params or {}

        # Apply base filter once
        self.df = self._apply_base_filter()

        # Initialize evaluation cache
        self._evaluation_cache = {}

        # Validate configuration eagerly so errors surface early
        self._validate_inputs()

    @property
    def metrics(self) -> tuple[MetricDefine, ...]:
        return self._metric_catalog.entries

    @property
    def estimates(self) -> Mapping[str, str]:
        return self._estimate_catalog.key_to_label

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

        if metrics is not None:
            filtered_metrics = self._resolve_metrics(metrics)
        else:
            filtered_metrics = list(self.metrics)
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
        verbose: bool = False,
    ) -> pl.LazyFrame | pl.DataFrame:
        """
        Unified evaluation method returning ARD format.

        Args:
            metrics: Subset of metrics to evaluate (None = use all configured)
            estimates: Subset of estimates to evaluate (None = use all configured)
            collect: When False, return a ``LazyFrame`` rather than materialising results.
            verbose: When True, include struct columns (``id``, ``groups``, ``subgroups``)
                and diagnostic fields (``stat``, ``stat_fmt``, ``context``, ``warning``,
                ``error``). When False, struct values are flattened and diagnostics are
                dropped for a compact table view.

        Returns:
            ``polars.DataFrame`` when ``collect`` is True (verbose controls struct
            flattening), or a ``LazyFrame`` when ``collect`` is False.
        """

        ard = self._evaluate_ard(metrics=metrics, estimates=estimates)

        if not collect:
            return ard.lazy

        if verbose:
            return format_verbose_frame(ard)
        return format_compact_frame(ard)

    def _convert_to_ard(self, result_lf: pl.LazyFrame) -> ARD:
        return format_to_ard(result_lf, self._formatter_context)

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
        return build_group_pivot(
            long_df,
            self._formatter_context,
            column_order_by=column_order_by,
            row_order_by=row_order_by,
        )

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
        return build_model_pivot(
            long_df,
            self._formatter_context,
            column_order_by=column_order_by,
            row_order_by=row_order_by,
        )

    def _resolve_metrics(
        self, metrics: MetricDefine | list[MetricDefine] | None
    ) -> list[MetricDefine]:
        """Resolve which metrics to evaluate"""
        if metrics is None:
            return list(self.metrics)

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
        # Create all subgroup combinations using vectorized unpivot
        subgroup_data = self._prepare_subgroup_data_vectorized(
            df_with_errors, self.subgroup_by
        )

        # Evaluate all metrics across all subgroups in a vectorized manner
        results = []
        for metric in metrics:
            metric_result = self._evaluate_metric_vectorized(
                subgroup_data, metric, estimates
            )
            results.append(metric_result)

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
        label_mapping = self._estimate_catalog.key_to_label
        df_long = (
            df_long.rename({self.ground_truth: "ground_truth"})
            .with_columns(
                [
                    pl.col("estimate_name").alias("estimate"),
                    pl.col("estimate_name")
                    .cast(pl.Utf8)
                    .map_elements(
                        lambda val, mapping=label_mapping: mapping.get(val, val),
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

        group_cols = self._get_vectorized_grouping_columns(metric, df_with_errors)
        within_infos, across_info = metric.compile_expressions()
        df_filtered = self._apply_metric_scope_filter(df_with_errors, metric, estimates)

        handlers = {
            MetricType.ACROSS_SAMPLE: self._evaluate_across_sample_metric,
            MetricType.WITHIN_SUBJECT: self._evaluate_within_entity_metric,
            MetricType.WITHIN_VISIT: self._evaluate_within_entity_metric,
            MetricType.ACROSS_SUBJECT: self._evaluate_two_stage_metric,
            MetricType.ACROSS_VISIT: self._evaluate_two_stage_metric,
        }

        handler = handlers.get(metric.type)
        if handler is None:
            raise ValueError(f"Unknown metric type: {metric.type}")

        try:
            result, result_info = handler(
                df_filtered,
                metric,
                group_cols,
                within_infos,
                across_info,
            )
            return self._add_metadata_vectorized(result, metric, result_info)
        except Exception as exc:
            fallback_info = self._fallback_metric_info(
                metric, within_infos, across_info
            )
            placeholder = self._prepare_error_lazyframe(group_cols)
            return self._add_metadata_vectorized(
                placeholder,
                metric,
                fallback_info,
                warnings_list=[],
                errors_list=[self._format_exception_message(exc, metric.name)],
            )

    def _evaluate_across_sample_metric(
        self,
        df: pl.LazyFrame,
        metric: MetricDefine,
        group_cols: list[str],
        _within_infos: Sequence[MetricInfo] | None,
        across_info: MetricInfo | None,
    ) -> tuple[pl.LazyFrame, MetricInfo]:
        if across_info is None:
            raise ValueError(f"ACROSS_SAMPLE metric {metric.name} requires across_expr")

        agg_exprs = self._metric_agg_expressions(across_info)
        result = self._aggregate_lazyframe(df, group_cols, agg_exprs)
        return result, across_info

    def _evaluate_within_entity_metric(
        self,
        df: pl.LazyFrame,
        metric: MetricDefine,
        group_cols: list[str],
        within_infos: Sequence[MetricInfo] | None,
        across_info: MetricInfo | None,
    ) -> tuple[pl.LazyFrame, MetricInfo]:
        entity_groups = self._merge_group_columns(
            self._get_entity_grouping_columns(metric.type), group_cols
        )

        result_info = self._resolve_metric_info(
            metric,
            primary=within_infos,
            fallback=across_info,
            error_message=f"No valid expression for metric {metric.name}",
        )

        agg_exprs = self._metric_agg_expressions(result_info)
        result = self._aggregate_lazyframe(df, entity_groups, agg_exprs)
        return result, result_info

    def _evaluate_two_stage_metric(
        self,
        df: pl.LazyFrame,
        metric: MetricDefine,
        group_cols: list[str],
        within_infos: Sequence[MetricInfo] | None,
        across_info: MetricInfo | None,
    ) -> tuple[pl.LazyFrame, MetricInfo]:
        entity_groups = self._merge_group_columns(
            self._get_entity_grouping_columns(metric.type), group_cols
        )

        base_info = self._resolve_metric_info(
            metric,
            primary=within_infos,
            fallback=across_info,
            error_message=(
                f"No valid expression for first level of metric {metric.name}"
            ),
        )

        intermediate = self._aggregate_lazyframe(
            df,
            entity_groups,
            self._metric_agg_expressions(base_info),
        )

        if across_info is not None and within_infos:
            result_info = across_info
            agg_exprs = self._metric_agg_expressions(result_info)
        else:
            result_info = base_info
            agg_exprs = [pl.col("value").mean().alias("value")]

        result = self._aggregate_lazyframe(intermediate, group_cols, agg_exprs)
        return result, result_info

    @staticmethod
    def _merge_group_columns(
        *column_groups: Sequence[str],
    ) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for columns in column_groups:
            for col in columns:
                if col not in seen:
                    seen.add(col)
                    ordered.append(col)
        return ordered

    def _resolve_metric_info(
        self,
        metric: MetricDefine,
        *,
        primary: Sequence[MetricInfo] | None,
        fallback: MetricInfo | None,
        error_message: str,
    ) -> MetricInfo:
        if primary:
            return primary[0]
        if fallback is not None:
            return fallback
        raise ValueError(error_message)

    @staticmethod
    def _aggregate_lazyframe(
        df: pl.LazyFrame, group_cols: Sequence[str], agg_exprs: Sequence[pl.Expr]
    ) -> pl.LazyFrame:
        columns = [col for col in group_cols if col]
        if columns:
            return df.group_by(columns).agg(agg_exprs)
        return df.select(*agg_exprs)

    def _get_vectorized_grouping_columns(
        self, metric: MetricDefine, df: pl.LazyFrame | None = None
    ) -> list[str]:
        """Get grouping columns for vectorized evaluation based on metric scope"""

        schema_names: set[str]
        if df is not None:
            schema_names = set(df.collect_schema().names())
        else:
            schema_names = set(self.df.collect_schema().names())

        using_vectorized_subgroups = {
            "subgroup_name",
            "subgroup_value",
        }.issubset(schema_names)

        def existing(columns: Iterable[str]) -> list[str]:
            return [col for col in columns if col in schema_names]

        group_cols: list[str] = []

        if metric.scope == MetricScope.GLOBAL:
            subgroup_cols = (
                ["subgroup_name", "subgroup_value"]
                if using_vectorized_subgroups
                else existing(self.subgroup_by.keys())
            )
            group_cols.extend(subgroup_cols)
        elif metric.scope == MetricScope.MODEL:
            model_cols = existing(["estimate"])
            subgroup_cols = (
                ["subgroup_name", "subgroup_value"]
                if using_vectorized_subgroups
                else existing(self.subgroup_by.keys())
            )
            group_cols.extend(model_cols + subgroup_cols)
        elif metric.scope == MetricScope.GROUP:
            group_cols.extend(existing(self.group_by.keys()))
            subgroup_cols = (
                ["subgroup_name", "subgroup_value"]
                if using_vectorized_subgroups
                else existing(self.subgroup_by.keys())
            )
            group_cols.extend(subgroup_cols)
        else:
            group_cols.extend(existing(["estimate"]))
            group_cols.extend(existing(self.group_by.keys()))
            subgroup_cols = (
                ["subgroup_name", "subgroup_value"]
                if using_vectorized_subgroups
                else existing(self.subgroup_by.keys())
            )
            group_cols.extend(subgroup_cols)

        return self._merge_group_columns(group_cols)

    def _apply_metric_scope_filter(
        self, df: pl.LazyFrame, metric: MetricDefine, estimates: list[str]
    ) -> pl.LazyFrame:
        """Apply any scope-specific filtering"""
        # For now, no additional filtering needed beyond grouping
        # Future: could add estimate filtering for specific scopes
        _ = metric, estimates  # Suppress unused parameter warnings
        return df

    def _metric_agg_expressions(self, info: MetricInfo) -> list[pl.Expr]:
        return [info.expr.alias("value")]

    def _get_entity_grouping_columns(self, metric_type: MetricType) -> list[str]:
        """Get entity-level grouping columns (subject_id, visit_id)"""
        if metric_type in [MetricType.WITHIN_SUBJECT, MetricType.ACROSS_SUBJECT]:
            return ["subject_id"]
        elif metric_type in [MetricType.WITHIN_VISIT, MetricType.ACROSS_VISIT]:
            return ["subject_id", "visit_id"]
        else:
            return []

    def _add_metadata_vectorized(
        self,
        result: pl.LazyFrame,
        metric: MetricDefine,
        info: MetricInfo,
        *,
        warnings_list: Sequence[str] | None = None,
        errors_list: Sequence[str] | None = None,
    ) -> pl.LazyFrame:
        """Add metadata columns to vectorized result"""

        schema = result.collect_schema()
        value_dtype = schema.get("value") if "value" in schema.names() else None
        value_kind = (
            (info.value_kind or "").lower()
            if info.value_kind
            else self._infer_value_kind_from_dtype(value_dtype)
        )
        if not value_kind:
            value_kind = "float"

        metadata_columns = [
            pl.lit(metric.name).cast(pl.Utf8).alias("metric"),
            pl.lit(metric.label).cast(pl.Utf8).alias("label"),
            pl.lit(metric.type.value).cast(pl.Utf8).alias("metric_type"),
            pl.lit(metric.scope.value if metric.scope else None)
            .cast(pl.Utf8)
            .alias("scope"),
            pl.lit(value_kind).cast(pl.Utf8).alias("_value_kind"),
            pl.lit(info.format).cast(pl.Utf8).alias("_value_format"),
        ]

        result = result.with_columns(metadata_columns)

        diagnostics_columns = [
            pl.lit(list(warnings_list or []), dtype=pl.List(pl.Utf8)).alias(
                "_diagnostic_warning"
            ),
            pl.lit(list(errors_list or []), dtype=pl.List(pl.Utf8)).alias(
                "_diagnostic_error"
            ),
        ]

        result = result.with_columns(diagnostics_columns)

        # Attach entity identifiers for within-entity metrics
        result = self._attach_entity_identifier(result, metric)

        # Helper columns for stat struct construction
        helper_columns = [
            pl.lit(None, dtype=pl.Float64).alias("_value_float"),
            pl.lit(None, dtype=pl.Int64).alias("_value_int"),
            pl.lit(None, dtype=pl.Boolean).alias("_value_bool"),
            pl.lit(None, dtype=pl.Utf8).alias("_value_str"),
            pl.lit(None).alias("_value_struct"),
        ]
        result = result.with_columns(helper_columns)

        if value_kind == "int":
            result = result.with_columns(
                pl.col("value").cast(pl.Int64, strict=False).alias("_value_int"),
                pl.col("value").cast(pl.Float64, strict=False).alias("_value_float"),
                pl.col("value").cast(pl.Float64, strict=False),
            )
        elif value_kind == "float":
            result = result.with_columns(
                pl.col("value").cast(pl.Float64, strict=False).alias("_value_float"),
                pl.col("value").cast(pl.Float64, strict=False),
            )
        elif value_kind == "bool":
            result = result.with_columns(
                pl.col("value").cast(pl.Boolean, strict=False).alias("_value_bool"),
                pl.lit(None, dtype=pl.Float64).alias("value"),
            )
        elif value_kind == "string":
            result = result.with_columns(
                pl.col("value").cast(pl.Utf8, strict=False).alias("_value_str"),
                pl.lit(None, dtype=pl.Float64).alias("value"),
            )
        elif value_kind == "struct":
            result = result.with_columns(
                pl.col("value").alias("_value_struct"),
                pl.lit(None, dtype=pl.Float64).alias("value"),
            )
        else:
            result = result.with_columns(
                pl.col("value").cast(pl.Utf8, strict=False).alias("_value_str"),
                pl.lit(None, dtype=pl.Float64).alias("value"),
            )

        return result

    @staticmethod
    def _format_exception_message(exc: Exception, metric_name: str) -> str:
        return f"{metric_name}: {type(exc).__name__}: {exc}".strip()

    @staticmethod
    def _fallback_metric_info(
        metric: MetricDefine,
        within_infos: Sequence[MetricInfo] | None,
        across_info: MetricInfo | None,
    ) -> MetricInfo:
        if within_infos:
            return within_infos[0]
        if across_info is not None:
            return across_info
        return MetricInfo(expr=pl.lit(None).alias("value"), value_kind="float")

    @staticmethod
    def _prepare_error_lazyframe(group_cols: Sequence[str]) -> pl.LazyFrame:
        data: dict[str, list[Any]] = {}
        for col in group_cols:
            data[col] = [None]
        data["value"] = [None]
        return pl.DataFrame(data, strict=False).lazy()

    @staticmethod
    def _infer_value_kind_from_dtype(dtype: pl.DataType | None) -> str:
        """Map Polars dtypes to MetricInfo value_kind labels."""

        if dtype is None or dtype == pl.Null:
            return "float"
        if dtype == pl.Struct:
            return "struct"
        if dtype == pl.Boolean:
            return "bool"
        if dtype == pl.Utf8:
            return "string"
        if hasattr(dtype, "is_numeric") and dtype.is_numeric():
            return "int" if dtype.is_integer() else "float"
        return "string"

    def _attach_entity_identifier(
        self, result: pl.LazyFrame, metric: MetricDefine
    ) -> pl.LazyFrame:
        """Attach a canonical id struct for within-entity metrics."""

        entity_cols = self._get_entity_grouping_columns(metric.type)
        if not entity_cols:
            if "id" in result.collect_schema().names():
                return result
            return result.with_columns(pl.lit(None).alias("id"))

        schema = result.collect_schema()
        available = set(schema.names())
        present = [col for col in entity_cols if col in available]

        if not present:
            if "id" in available:
                return result
            return result.with_columns(pl.lit(None).alias("id"))

        id_struct = pl.struct([pl.col(col).alias(col) for col in present]).alias("id")
        cleaned = result.with_columns(id_struct)

        # Drop entity columns to avoid polluting downstream schema
        return cleaned.drop(present)

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
        estimate_label_map = self._estimate_catalog.key_to_label

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
                    lambda val, mapping=estimate_label_map: mapping.get(val, val),
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

        if "stat" in schema.names():
            exprs.append(pl.col("stat"))
            if "stat_fmt" in schema.names():
                exprs.append(pl.col("stat_fmt"))
                exprs.append(
                    pl.when(pl.col("stat_fmt").is_null())
                    .then(
                        pl.col("stat").map_elements(
                            ARD._format_stat, return_dtype=pl.Utf8
                        )
                    )
                    .otherwise(pl.col("stat_fmt"))
                    .alias("value")
                )
            else:
                exprs.append(
                    pl.col("stat")
                    .map_elements(ARD._format_stat, return_dtype=pl.Utf8)
                    .alias("value")
                )

        if "id" in schema.names():
            exprs.append(pl.col("id"))

        # Scope metadata
        if "metric_type" in schema.names():
            exprs.append(pl.col("metric_type").cast(pl.Utf8))
        if "scope" in schema.names():
            exprs.append(pl.col("scope").cast(pl.Utf8))

        return lf.select(exprs).collect()

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

        categories: list[Any] = []
        seen: set[Any] = set()
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

            ordered_values = self._collect_unique_subgroup_values(column, dtype)
            for value in ordered_values:
                if value in seen:
                    continue
                seen.add(value)
                categories.append(value)

        return categories

    def _collect_unique_subgroup_values(
        self, column: str, dtype: pl.DataType
    ) -> list[Any]:
        """Collect sorted unique subgroup values using lazy execution."""

        expr = pl.col(column).drop_nulls()

        if not dtype.is_numeric():
            expr = expr.cast(pl.Utf8)

        lazy_unique = (
            self.df_raw.select(expr.alias(column)).unique(subset=[column]).sort(column)
        )

        df_unique = lazy_unique.collect(engine="streaming")
        values = df_unique[column].to_list()

        if dtype.is_numeric():
            return values

        return [str(value) for value in values]

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

        overlap = set(self.group_by.keys()) & set(self.subgroup_by.keys())
        if overlap:
            raise ValueError(
                "Group and subgroup columns must be distinct; found duplicates: "
                f"{sorted(overlap)}"
            )
