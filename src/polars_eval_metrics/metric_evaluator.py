"""
Unified Metric Evaluation Pipeline

This module implements a simplified, unified evaluation pipeline for computing metrics
using Polars LazyFrames with comprehensive support for scopes, groups, and subgroups.
"""

from typing import Any

# pyre-strict

import polars as pl

from .ard import ARD
from .metric_define import MetricDefine, MetricScope, MetricType
from .metric_registry import MetricRegistry


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
        self.group_by = self._process_grouping(group_by)
        self.subgroup_by = self._process_grouping(subgroup_by)
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
            # Compute and cache the result using new ARD method
            # Create temporary evaluator with filtered metrics/estimates for caching
            if metrics is not None or estimates is not None:
                filtered_evaluator = self.filter(metrics=metrics, estimates=estimates)
                result = filtered_evaluator.evaluate()
            else:
                result = self.evaluate()
            self._evaluation_cache[cache_key] = result

        return self._evaluation_cache[cache_key]

    def clear_cache(self) -> None:
        """Clear the evaluation cache"""
        self._evaluation_cache.clear()

    def evaluate(
        self,
        metrics: MetricDefine | list[MetricDefine] | None = None,
        estimates: str | list[str] | None = None,
    ) -> ARD:
        """
        Unified evaluation method returning ARD format.

        Args:
            metrics: Subset of metrics to evaluate (None = use all configured)
            estimates: Subset of estimates to evaluate (None = use all configured)

        Returns:
            ARD object with evaluation results
        """
        # Determine evaluation targets
        target_metrics = self._resolve_metrics(metrics)
        target_estimates = self._resolve_estimates(estimates)

        if not target_metrics or not target_estimates:
            raise ValueError("No metrics or estimates to evaluate")

        # Vectorized evaluation - single Polars operation
        combined = self._vectorized_evaluate(target_metrics, target_estimates)

        # Format final result
        formatted = self._format_result(combined)

        # Convert to ARD format
        return self._convert_to_ard(formatted)

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
                pl.when(pl.all_horizontal([pl.col(col).is_null() for col in group_cols]))
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
        value_dtype = schema.get("value", pl.Float64)
        value_col = pl.col("value")
        null_utf8 = pl.lit(None, dtype=pl.Utf8)
        null_float = pl.lit(None, dtype=pl.Float64)
        null_int = pl.lit(None, dtype=pl.Int64)
        null_bool = pl.lit(None, dtype=pl.Boolean)

        if value_dtype in (pl.Float32, pl.Float64):
            stat_expr = pl.struct(
                [
                    pl.when(value_col.is_null())
                    .then(null_utf8)
                    .otherwise(pl.lit("float"))
                    .alias("type"),
                    value_col.cast(pl.Float64).alias("value_float"),
                    null_int.alias("value_int"),
                    null_bool.alias("value_bool"),
                    null_utf8.alias("value_str"),
                    null_utf8.alias("value_json"),
                    null_utf8.alias("format"),
                    null_utf8.alias("unit"),
                ]
            ).alias("stat")
        elif value_dtype in (
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
        ):
            stat_expr = pl.struct(
                [
                    pl.when(value_col.is_null())
                    .then(null_utf8)
                    .otherwise(pl.lit("int"))
                    .alias("type"),
                    null_float.alias("value_float"),
                    value_col.cast(pl.Int64).alias("value_int"),
                    null_bool.alias("value_bool"),
                    null_utf8.alias("value_str"),
                    null_utf8.alias("value_json"),
                    null_utf8.alias("format"),
                    null_utf8.alias("unit"),
                ]
            ).alias("stat")
        elif value_dtype == pl.Boolean:
            stat_expr = pl.struct(
                [
                    pl.when(value_col.is_null())
                    .then(null_utf8)
                    .otherwise(pl.lit("bool"))
                    .alias("type"),
                    null_float.alias("value_float"),
                    null_int.alias("value_int"),
                    value_col.alias("value_bool"),
                    null_utf8.alias("value_str"),
                    null_utf8.alias("value_json"),
                    null_utf8.alias("format"),
                    null_utf8.alias("unit"),
                ]
            ).alias("stat")
        elif value_dtype == pl.Utf8:
            stat_expr = pl.struct(
                [
                    pl.when(value_col.is_null())
                    .then(null_utf8)
                    .otherwise(pl.lit("string"))
                    .alias("type"),
                    null_float.alias("value_float"),
                    null_int.alias("value_int"),
                    null_bool.alias("value_bool"),
                    value_col.alias("value_str"),
                    null_utf8.alias("value_json"),
                    null_utf8.alias("format"),
                    null_utf8.alias("unit"),
                ]
            ).alias("stat")
        else:
            stat_expr = pl.struct(
                [
                    pl.when(value_col.is_null())
                    .then(null_utf8)
                    .otherwise(pl.lit("json"))
                    .alias("type"),
                    null_float.alias("value_float"),
                    null_int.alias("value_int"),
                    null_bool.alias("value_bool"),
                    null_utf8.alias("value_str"),
                    value_col.map_elements(ARD._encode_json, return_dtype=pl.Utf8).alias("value_json"),
                    null_utf8.alias("format"),
                    null_utf8.alias("unit"),
                ]
            ).alias("stat")

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

        group_cols = [self.group_by[col] for col in self.group_by]
        subgroup_present = "subgroup_name" in long_df.columns and "subgroup_value" in long_df.columns

        if row_order_by == "subgroup" and subgroup_present:
            index_cols = ["subgroup_name", "subgroup_value"] + group_cols
        else:
            index_cols = group_cols + (["subgroup_name", "subgroup_value"] if subgroup_present else [])

        def pivot_default(df: pl.DataFrame) -> pl.DataFrame:
            default_df = df.filter(pl.col("scope").is_null())
            if default_df.is_empty():
                return pl.DataFrame()

            on_cols = ["estimate", "label"]
            if column_order_by == "metrics":
                on_cols = ["label", "estimate"]

            if index_cols:
                pivoted = default_df.pivot(
                    index=index_cols,
                    columns=on_cols,
                    values="value",
                    aggregate_function="first",
                )
            else:
                with_idx = default_df.with_row_index("_idx")
                pivoted = with_idx.pivot(
                    index=["_idx"],
                    columns=on_cols,
                    values="value",
                    aggregate_function="first",
                ).drop("_idx")

            return pivoted

        def pivot_scope(df: pl.DataFrame, scope_value: str, columns: list[str]) -> tuple[pl.DataFrame, list[str]]:
            scoped = df.filter(pl.col("scope") == scope_value)
            if scoped.is_empty():
                return pl.DataFrame(), []

            if index_cols:
                pivoted = scoped.pivot(
                    index=index_cols,
                    columns=columns,
                    values="value",
                    aggregate_function="first",
                )
            else:
                with_idx = scoped.with_row_index("_idx")
                pivoted = with_idx.pivot(
                    index=["_idx"],
                    columns=columns,
                    values="value",
                    aggregate_function="first",
                ).drop("_idx")

            value_cols = [col for col in pivoted.columns if col not in index_cols]
            return pivoted, value_cols

        result = pivot_default(long_df)

        group_pivot, group_cols_created = pivot_scope(long_df, "group", ["label"])
        if not group_pivot.is_empty():
            result = result.join(group_pivot, on=index_cols, how="left")

        global_pivot, global_cols_created = pivot_scope(long_df, "global", ["label"])
        if not global_pivot.is_empty():
            result = result.join(global_pivot, on=index_cols, how="left")

        if result.is_empty():
            if index_cols:
                return pl.DataFrame({col: [] for col in index_cols})
            return pl.DataFrame()

        # Reorder columns: index -> global -> group -> default
        value_cols = [col for col in result.columns if col not in index_cols]
        default_cols = [col for col in value_cols if col.startswith('{"') and col.endswith('"}')] if value_cols else []

        def sort_default(columns: list[str]) -> list[str]:
            def parse(col: str) -> tuple[str, str]:
                inner = col[2:-2]
                parts = inner.split('","')
                return (parts[0], parts[1]) if len(parts) == 2 else (col, "")

            if column_order_by == "metrics":
                return sorted(columns, key=lambda c: (parse(c)[1], parse(c)[0]))
            return sorted(columns, key=lambda c: (parse(c)[0], parse(c)[1]))

        ordered = (
            index_cols
            + global_cols_created
            + group_cols_created
            + sort_default(default_cols)
        )

        remaining = [col for col in value_cols if col not in ordered]
        ordered.extend(remaining)

        ordered = [col for col in ordered if col in result.columns]

        seen: set[str] = set()
        deduped: list[str] = []
        for col in ordered:
            if col not in seen:
                deduped.append(col)
                seen.add(col)

        return result.select(deduped)

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

        subgroup_present = "subgroup_name" in long_df.columns and "subgroup_value" in long_df.columns

        if row_order_by == "subgroup" and subgroup_present:
            index_cols = ["estimate", "subgroup_name", "subgroup_value"]
        else:
            index_cols = ["estimate"] + (["subgroup_name", "subgroup_value"] if subgroup_present else [])

        default_df = long_df.filter(pl.col("scope").is_null())
        if default_df.is_empty():
            default_pivot = pl.DataFrame({col: [] for col in index_cols})
        else:
            on_cols = [label for label in self.group_by.values()] + ["label"]
            default_pivot = default_df.pivot(
                index=index_cols,
                columns=on_cols,
                values="value",
                aggregate_function="first",
            )

        global_df = long_df.filter(pl.col("scope") == "global")
        if not global_df.is_empty():
            global_pivot = global_df.pivot(
                index=index_cols,
                columns=["label"],
                values="value",
                aggregate_function="first",
            )
            default_pivot = default_pivot.join(global_pivot, on=index_cols, how="left")

        group_df = long_df.filter(pl.col("scope") == "group")
        if not group_df.is_empty():
            group_pivot = group_df.pivot(
                index=index_cols,
                columns=[label for label in self.group_by.values()],
                values="value",
                aggregate_function="first",
            )
            default_pivot = default_pivot.join(group_pivot, on=index_cols, how="left")

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

        # Replace estimate names with their display labels and rename columns
        df_long = df_long.with_columns(
            [pl.col("estimate_name").replace(self.estimates).alias("estimate")]
        ).rename({self.ground_truth: "ground_truth"})

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
        within_exprs, across_expr = metric.compile_expressions()

        # Apply metric-specific filtering if needed
        df_filtered = self._apply_metric_scope_filter(df_with_errors, metric, estimates)

        # Perform the evaluation with appropriate grouping
        if metric.type == MetricType.ACROSS_SAMPLE:
            if across_expr is None:
                raise ValueError(
                    f"ACROSS_SAMPLE metric {metric.name} requires across_expr"
                )

            if group_cols:
                result = df_filtered.group_by(group_cols).agg(
                    across_expr.alias("value").cast(pl.Float64)
                )
            else:
                result = df_filtered.select(across_expr.alias("value").cast(pl.Float64))

        elif metric.type in [MetricType.WITHIN_SUBJECT, MetricType.WITHIN_VISIT]:
            # Within-entity aggregation
            entity_groups = self._get_entity_grouping_columns(metric.type) + group_cols

            # Use the appropriate expression for within-entity aggregation
            if within_exprs:
                agg_expr = within_exprs[0].alias("value").cast(pl.Float64)
            elif across_expr is not None:
                agg_expr = across_expr.alias("value").cast(pl.Float64)
            else:
                raise ValueError(f"No valid expression for metric {metric.name}")

            result = df_filtered.group_by(entity_groups).agg(agg_expr)

        elif metric.type in [MetricType.ACROSS_SUBJECT, MetricType.ACROSS_VISIT]:
            # Two-level aggregation: within entities, then across
            entity_groups = self._get_entity_grouping_columns(metric.type) + group_cols

            # First level: within entities
            if within_exprs:
                first_level_expr = within_exprs[0].alias("value")
            elif across_expr is not None:
                first_level_expr = across_expr.alias("value")
            else:
                raise ValueError(
                    f"No valid expression for first level of metric {metric.name}"
                )

            intermediate = df_filtered.group_by(entity_groups).agg(first_level_expr)

            # Second level: across entities
            if across_expr is not None and within_exprs:
                # True two-level case - use the across_expr directly since we named the column 'value'
                second_level_expr = across_expr.alias("value").cast(pl.Float64)
            else:
                # Default aggregation across entities when no across_expr specified
                second_level_expr = (
                    pl.col("value").mean().alias("value").cast(pl.Float64)
                )

            if group_cols:
                result = intermediate.group_by(group_cols).agg(second_level_expr)
            else:
                result = intermediate.select(second_level_expr)

        else:
            raise ValueError(f"Unknown metric type: {metric.type}")

        # Add metadata columns
        return self._add_metadata_vectorized(result, metric)

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

    def _get_entity_grouping_columns(self, metric_type: MetricType) -> list[str]:
        """Get entity-level grouping columns (subject_id, visit_id)"""
        if metric_type in [MetricType.WITHIN_SUBJECT, MetricType.ACROSS_SUBJECT]:
            return ["subject_id"]
        elif metric_type in [MetricType.WITHIN_VISIT, MetricType.ACROSS_VISIT]:
            return ["subject_id", "visit_id"]
        else:
            return []

    def _add_metadata_vectorized(
        self, result: pl.LazyFrame, metric: MetricDefine
    ) -> pl.LazyFrame:
        """Add metadata columns to vectorized result"""

        metadata = [
            pl.lit(metric.name).alias("metric"),
            pl.lit(metric.label).alias("label"),
            pl.lit(metric.type.value).alias("metric_type"),
            pl.lit(metric.scope.value if metric.scope else None).alias("scope"),
        ]

        # Add metadata columns
        result_with_metadata = result.with_columns(metadata)

        return result_with_metadata

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
            exprs.append(pl.col("subgroup_name").cast(pl.Utf8))
        if "subgroup_value" in schema.names():
            exprs.append(pl.col("subgroup_value").cast(pl.Utf8))

        # Estimate / metric / label columns
        if "estimate" in schema.names():
            exprs.append(pl.col("estimate").cast(pl.Utf8))
        if "metric" in schema.names():
            exprs.append(pl.col("metric").cast(pl.Utf8))
        if "label" in schema.names():
            exprs.append(pl.col("label").cast(pl.Utf8))
        else:
            exprs.append(pl.col("metric").cast(pl.Utf8).alias("label"))

        # Numeric value
        if "value" in schema.names():
            exprs.append(pl.col("value").cast(pl.Float64))
        else:
            exprs.append(pl.col("stat").struct.field("value_float").alias("value"))

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
        if not self.subgroup_by or "subgroup_name" not in schema.names() or "subgroup_value" not in schema.names():
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
                pl.col("subgroup_name").is_null()
                | pl.col("subgroup_value").is_null()
            )
            .then(pl.lit(None, dtype=dtype))
            .otherwise(pl.struct(fields))
            .alias("subgroups")
        )

    def _expr_stat_struct(self, schema: pl.Schema) -> pl.Expr:
        value_dtype = schema.get("value", pl.Float64)
        value_col = pl.col("value")
        null_utf8 = pl.lit(None, dtype=pl.Utf8)
        null_float = pl.lit(None, dtype=pl.Float64)
        null_int = pl.lit(None, dtype=pl.Int64)
        null_bool = pl.lit(None, dtype=pl.Boolean)

        if value_dtype in (pl.Float32, pl.Float64):
            return pl.struct(
                [
                    pl.when(value_col.is_null()).then(null_utf8).otherwise(pl.lit("float")).alias("type"),
                    value_col.cast(pl.Float64).alias("value_float"),
                    null_int.alias("value_int"),
                    null_bool.alias("value_bool"),
                    null_utf8.alias("value_str"),
                    null_utf8.alias("value_json"),
                    null_utf8.alias("format"),
                    null_utf8.alias("unit"),
                ]
            ).alias("stat")

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
            return pl.struct(
                [
                    pl.when(value_col.is_null()).then(null_utf8).otherwise(pl.lit("int")).alias("type"),
                    null_float.alias("value_float"),
                    value_col.cast(pl.Int64).alias("value_int"),
                    null_bool.alias("value_bool"),
                    null_utf8.alias("value_str"),
                    null_utf8.alias("value_json"),
                    null_utf8.alias("format"),
                    null_utf8.alias("unit"),
                ]
            ).alias("stat")

        if value_dtype == pl.Boolean:
            return pl.struct(
                [
                    pl.when(value_col.is_null()).then(null_utf8).otherwise(pl.lit("bool")).alias("type"),
                    null_float.alias("value_float"),
                    null_int.alias("value_int"),
                    value_col.alias("value_bool"),
                    null_utf8.alias("value_str"),
                    null_utf8.alias("value_json"),
                    null_utf8.alias("format"),
                    null_utf8.alias("unit"),
                ]
            ).alias("stat")

        if value_dtype == pl.Utf8:
            return pl.struct(
                [
                    pl.when(value_col.is_null()).then(null_utf8).otherwise(pl.lit("string")).alias("type"),
                    null_float.alias("value_float"),
                    null_int.alias("value_int"),
                    null_bool.alias("value_bool"),
                    value_col.alias("value_str"),
                    null_utf8.alias("value_json"),
                    null_utf8.alias("format"),
                    null_utf8.alias("unit"),
                ]
            ).alias("stat")

        return pl.struct(
            [
                pl.when(value_col.is_null()).then(null_utf8).otherwise(pl.lit("json")).alias("type"),
                null_float.alias("value_float"),
                null_int.alias("value_int"),
                null_bool.alias("value_bool"),
                null_utf8.alias("value_str"),
                value_col.map_elements(ARD._encode_json, return_dtype=pl.Utf8).alias("value_json"),
                null_utf8.alias("format"),
                null_utf8.alias("unit"),
            ]
        ).alias("stat")

    def _expr_context_struct(self, schema: pl.Schema) -> pl.Expr:
        null_utf8 = pl.lit(None, dtype=pl.Utf8)
        fields = []
        for field in ("metric_type", "scope", "label"):
            if field in schema.names():
                fields.append(pl.col(field).cast(pl.Utf8).alias(field))
            else:
                fields.append(null_utf8.alias(field))
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
        label_lookup = {metric.name: metric.label or metric.name for metric in self.metrics}
        label_categories = list(dict.fromkeys(label_lookup.values()))
        return (
            pl.col("metric")
            .cast(pl.Utf8)
            .replace(label_lookup)
            .cast(pl.Enum(label_categories))
            .alias("label")
        )

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
