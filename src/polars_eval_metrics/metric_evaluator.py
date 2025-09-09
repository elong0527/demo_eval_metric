"""
Unified Metric Evaluation Pipeline

This module implements a simplified, unified evaluation pipeline for computing metrics
using Polars LazyFrames with comprehensive support for scopes, groups, and subgroups.
"""

from typing import Any

# pyre-strict

import polars as pl

from .metric_define import MetricDefine, MetricScope, MetricType
from .metric_registry import MetricRegistry


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
    _evaluation_cache: dict[tuple[tuple[str, ...], tuple[str, ...]], pl.DataFrame]

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
    ) -> pl.DataFrame:
        """Get cached evaluation result or compute and cache if not exists"""
        cache_key = self._get_cache_key(metrics, estimates)

        if cache_key not in self._evaluation_cache:
            # Compute and cache the result
            result_data = self.evaluate(
                metrics=metrics, estimates=estimates, collect=True
            )
            # Ensure we have a DataFrame for caching
            result = (
                result_data
                if isinstance(result_data, pl.DataFrame)
                else result_data.collect()
            )
            self._evaluation_cache[cache_key] = result

        return self._evaluation_cache[cache_key]

    def clear_cache(self) -> None:
        """Clear the evaluation cache"""
        self._evaluation_cache.clear()

    def _build_index_cols(
        self, long_df: pl.DataFrame, include_estimate: bool = False
    ) -> list[str]:
        """Build index columns for pivot operations"""
        index_cols = []
        if not include_estimate and self.group_by:
            index_cols.extend(self.group_by)
        if "subgroup_name" in long_df.columns:
            index_cols.extend(["subgroup_name", "subgroup_value"])
        if include_estimate:
            index_cols.append("estimate")
        return index_cols

    def _add_group_columns(
        self, result: pl.DataFrame, group_data: pl.DataFrame
    ) -> pl.DataFrame:
        """Add group scope columns by broadcasting group x metric combinations"""
        if group_data.is_empty():
            return result

        # Create group combination column
        if self.group_by:
            group_expr = pl.concat_str(
                [pl.col(col).cast(pl.Utf8) for col in self.group_by], separator="_"
            )
            group_data = group_data.with_columns(group_expr.alias("group_combination"))
        else:
            group_data = group_data.with_columns(
                pl.lit("ALL").alias("group_combination")
            )

        # Add group x metric combinations
        for row in group_data.iter_rows(named=True):
            col_name = f"{row['group_combination']}_{row['label']}"
            value = row["value"]
            result = result.with_columns(pl.lit(value).alias(col_name))

        return result

    def _add_global_columns(
        self, result: pl.DataFrame, global_data: pl.DataFrame
    ) -> pl.DataFrame:
        """Add global scope columns by broadcasting values"""
        if global_data.is_empty():
            return result
        for row in global_data.iter_rows(named=True):
            col_name = row["label"]
            value = row["value"]
            result = result.with_columns(pl.lit(value).alias(col_name))
        return result

    def _reorder_columns(
        self,
        result: pl.DataFrame,
        index_cols: list[str],
        global_data: pl.DataFrame,
        group_data: pl.DataFrame,
    ) -> pl.DataFrame:
        """Reorder columns: index -> global -> group -> default"""
        if result.is_empty():
            return result

        all_cols = result.columns
        global_cols = []
        group_cols = []
        default_cols = []

        # Get scope information from the original data
        global_labels = []
        if not global_data.is_empty():
            global_labels = global_data.select("label").unique().to_series().to_list()

        group_labels = []
        if not group_data.is_empty():
            group_labels = group_data.select("label").unique().to_series().to_list()

        for col in all_cols:
            if col in index_cols:
                continue
            elif col in global_labels:
                global_cols.append(col)
            elif any(col.endswith(f"_{label}") for label in group_labels):
                group_cols.append(col)
            else:
                default_cols.append(col)

        # Build ordered column list
        ordered_cols = (
            index_cols + sorted(global_cols) + sorted(group_cols) + sorted(default_cols)
        )

        # Only reorder if we have all columns (safety check)
        if len(ordered_cols) == len(all_cols):
            result = result.select(ordered_cols)

        return result

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

        # Vectorized evaluation - single Polars operation
        combined = self._vectorized_evaluate(target_metrics, target_estimates)

        # Format final result
        formatted = self._format_result(combined)

        return formatted.collect() if collect else formatted

    def pivot_by_group(
        self,
        metrics: MetricDefine | list[MetricDefine] | None = None,
        estimates: str | list[str] | None = None,
    ) -> pl.DataFrame:
        """
        Pivot results with groups as rows and model x metric as columns.

        Args:
            metrics: Subset of metrics to evaluate (None = use all configured)
            estimates: Subset of estimates to evaluate (None = use all configured)

        Returns:
            DataFrame with group combinations as rows
        """
        long_df = self._get_cached_evaluation(metrics=metrics, estimates=estimates)

        # Build index columns
        index_cols = self._build_index_cols(long_df, include_estimate=False)

        # Separate data by scope
        default_data = long_df.filter(pl.col("scope").is_null())
        group_data = long_df.filter(pl.col("scope") == "group")
        global_data = long_df.filter(pl.col("scope") == "global")

        # Handle each scope separately
        result_parts = []

        # 1. Default scope: model x metric columns
        if not default_data.is_empty():
            pivot_col = (
                pl.col("estimate").cast(pl.Utf8) + "_" + pl.col("label").cast(pl.Utf8)
            ).alias("pivot_col")
            default_pivot = default_data.with_columns(pivot_col)
            if index_cols:
                default_result = default_pivot.pivot(
                    on="pivot_col", values="value", index=index_cols
                )
            else:
                default_result = default_pivot.pivot(
                    on="pivot_col", values="value", index=pl.lit(1).alias("_row")
                )
                if "_row" in default_result.columns:
                    default_result = default_result.drop("_row")
            result_parts.append(default_result)

        # 2. Group scope: metric label as column
        if not group_data.is_empty():
            if index_cols:
                group_result = group_data.pivot(
                    on="label", values="value", index=index_cols
                )
            else:
                group_result = group_data.pivot(
                    on="label", values="value", index=pl.lit(1).alias("_row")
                )
                if "_row" in group_result.columns:
                    group_result = group_result.drop("_row")
            result_parts.append(group_result)

        # Combine results
        if result_parts:
            result = result_parts[0]
            for part in result_parts[1:]:
                if index_cols:
                    result = result.join(part, on=index_cols, how="full")
                    # Clean up duplicate columns from join
                    for col in index_cols:
                        if f"{col}_right" in result.columns:
                            result = result.drop(f"{col}_right")
                else:
                    # Add columns that don't already exist
                    for col in part.columns:
                        if col not in result.columns:
                            result = result.with_columns(part.select(col))
        else:
            # Create result with proper index structure
            if index_cols:
                # For global scope only, get unique combinations from original data
                available_cols = [
                    col
                    for col in index_cols
                    if col in self.df_raw.collect_schema().names()
                ]
                if available_cols:
                    # Get unique combinations from the original data
                    result = self.df_raw.select(available_cols).unique().collect()
                else:
                    # If no group columns available, create single row
                    result = pl.DataFrame({col: [None] for col in index_cols})
            else:
                result = pl.DataFrame({})

        # Add global columns and reorder
        result = self._add_global_columns(result, global_data)
        result = self._reorder_columns(result, index_cols, global_data, group_data)

        return result

    def pivot_by_model(
        self,
        metrics: MetricDefine | list[MetricDefine] | None = None,
        estimates: str | list[str] | None = None,
    ) -> pl.DataFrame:
        """
        Pivot results with models as rows and group x metric as columns.

        Args:
            metrics: Subset of metrics to evaluate (None = use all configured)
            estimates: Subset of estimates to evaluate (None = use all configured)

        Returns:
            DataFrame with models as rows
        """
        long_df = self._get_cached_evaluation(metrics=metrics, estimates=estimates)

        # Build index columns
        index_cols = self._build_index_cols(long_df, include_estimate=True)

        # Separate data by scope
        model_default_data = long_df.filter(
            (pl.col("scope").is_in(["model", "default"])) | (pl.col("scope").is_null())
        )
        global_data = long_df.filter(pl.col("scope") == "global")
        group_data = long_df.filter(pl.col("scope") == "group")

        # Create pivot result from model/default data
        if not model_default_data.is_empty():
            # Create group x metric columns
            if self.group_by:
                group_combo = pl.concat_str(
                    [pl.col(col).cast(pl.Utf8) for col in self.group_by], separator="_"
                )
            else:
                group_combo = pl.lit("ALL")

            pivot_col = (group_combo + "_" + pl.col("metric").cast(pl.Utf8)).alias(
                "pivot_col"
            )
            pivot_df = model_default_data.with_columns(pivot_col)
            result = pivot_df.pivot(on="pivot_col", values="value", index=index_cols)
        else:
            # Create empty result with index columns
            result = pl.DataFrame().with_columns(
                [pl.lit(None, dtype=pl.Utf8).alias(col) for col in index_cols]
            )

        # Add global and group columns, then reorder
        result = self._add_global_columns(result, global_data)
        result = self._add_group_columns(result, group_data)
        result = self._reorder_columns(result, index_cols, global_data, group_data)

        return result

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

        # Harmonize schemas before combining
        if results:
            harmonized_results = self._harmonize_result_schemas(results)
            return pl.concat(harmonized_results, how="diagonal")
        else:
            return pl.DataFrame().lazy()

    def _evaluate_with_marginal_subgroups(
        self,
        df_with_errors: pl.LazyFrame,
        metrics: list[MetricDefine],
        estimates: list[str],
    ) -> pl.LazyFrame:
        """Evaluate metrics with marginal subgroup analysis"""
        results = []

        # For each metric, evaluate across all marginal subgroup combinations
        for metric in metrics:
            # For each subgroup variable, create separate analyses
            for subgroup_col in self.subgroup_by:
                # Temporarily set subgroup_by to single column for this analysis
                original_subgroup_by = self.subgroup_by
                self.subgroup_by = [subgroup_col]

                try:
                    # Evaluate metric for this single subgroup
                    metric_result = self._evaluate_metric_vectorized(
                        df_with_errors, metric, estimates
                    )

                    # Add subgroup metadata columns
                    metric_result = metric_result.with_columns(
                        [
                            pl.lit(subgroup_col).alias("subgroup_name"),
                            pl.col(subgroup_col).cast(pl.Utf8).alias("subgroup_value"),
                        ]
                    )

                    # Remove the original subgroup column to avoid duplication
                    available_cols = metric_result.collect_schema().names()
                    if subgroup_col in available_cols:
                        cols_to_keep = [
                            col for col in available_cols if col != subgroup_col
                        ]
                        metric_result = metric_result.select(cols_to_keep)

                    results.append(metric_result)

                finally:
                    # Restore original subgroup_by
                    self.subgroup_by = original_subgroup_by

        # Harmonize schemas before combining
        if results:
            harmonized_results = self._harmonize_result_schemas(results)
            return pl.concat(harmonized_results, how="diagonal")
        else:
            return pl.DataFrame().lazy()

    def _prepare_long_format_data(self, estimates: list[str]) -> pl.LazyFrame:
        """Reshape data from wide to long format for vectorized processing"""

        # Get all columns except estimates to preserve in melt
        id_vars = []
        for col in self.df.collect_schema().names():
            if col not in estimates:
                id_vars.append(col)

        # Unpivot estimates into long format
        df_long = self.df.unpivot(
            index=id_vars,
            on=estimates,
            variable_name="estimate_name",
            value_name="estimate_value",
        )

        return df_long

    def _add_error_columns_vectorized(self, df_long: pl.LazyFrame) -> pl.LazyFrame:
        """Add error columns for the long-format data"""

        # Generate error expressions for the vectorized format
        # Use 'estimate_value' as the estimate column and ground_truth as before
        error_expressions = MetricRegistry.generate_error_columns(
            estimate="estimate_value",
            ground_truth=self.ground_truth,
            error_types=None,
            error_params=self.error_params,
        )

        return df_long.with_columns(error_expressions)

    def _evaluate_metric_vectorized(
        self, df_with_errors: pl.LazyFrame, metric: MetricDefine, estimates: list[str]
    ) -> pl.LazyFrame:
        """Evaluate a single metric using vectorized operations"""

        # Determine grouping columns based on metric scope
        group_cols = self._get_vectorized_grouping_columns(metric)

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
                first_level_expr = within_exprs[0].alias("intermediate_value")
            elif across_expr is not None:
                first_level_expr = across_expr.alias("intermediate_value")
            else:
                raise ValueError(
                    f"No valid expression for first level of metric {metric.name}"
                )

            intermediate = df_filtered.group_by(entity_groups).agg(first_level_expr)

            # Second level: across entities
            if across_expr is not None and within_exprs:
                # True two-level case - substitute 'value' with 'intermediate_value' in the expression
                # For now, use the across expression directly but this may need column name mapping
                # TODO: More sophisticated expression rewriting to handle column references
                second_level_expr = (
                    pl.col("intermediate_value").mean().alias("value").cast(pl.Float64)
                )
            else:
                # Default aggregation across entities
                second_level_expr = (
                    pl.col("intermediate_value").mean().alias("value").cast(pl.Float64)
                )

            if group_cols:
                result = intermediate.group_by(group_cols).agg(second_level_expr)
            else:
                result = intermediate.select(second_level_expr)

        else:
            raise ValueError(f"Unknown metric type: {metric.type}")

        # Add metadata columns
        return self._add_metadata_vectorized(result, metric)

    def _get_vectorized_grouping_columns(self, metric: MetricDefine) -> list[str]:
        """Get grouping columns for vectorized evaluation based on metric scope"""

        group_cols = []

        # Handle scope-based grouping
        if metric.scope == MetricScope.GLOBAL:
            # Global: only subgroups
            group_cols.extend(self.subgroup_by)

        elif metric.scope == MetricScope.MODEL:
            # Model: estimate + subgroups
            group_cols.append("estimate_name")
            group_cols.extend(self.subgroup_by)

        elif metric.scope == MetricScope.GROUP:
            # Group: groups + subgroups (no estimate separation)
            group_cols.extend(self.group_by)
            group_cols.extend(self.subgroup_by)

        else:
            # Default: estimate + groups + subgroups
            group_cols.append("estimate_name")
            group_cols.extend(self.group_by)
            group_cols.extend(self.subgroup_by)

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

        # Rename estimate_name to estimate for compatibility
        result_with_metadata = result.with_columns(metadata)

        # Check if estimate_name column exists and rename it
        schema = result_with_metadata.collect_schema()
        if "estimate_name" in schema.names():
            result_with_metadata = result_with_metadata.rename(
                {"estimate_name": "estimate"}
            )

        return result_with_metadata

    def _format_result(self, combined: pl.LazyFrame) -> pl.LazyFrame:
        """Format final result with proper column ordering and sorting"""

        # Determine available columns
        try:
            available_columns = combined.collect_schema().names()
        except Exception:
            available_columns = combined.limit(1).collect().columns

        # Extract ordered values for enum columns from definitions
        ordered_metrics = []
        ordered_labels = []
        for metric in self.metrics:
            if metric.name not in ordered_metrics:
                ordered_metrics.append(metric.name)
            label = metric.label
            if label not in ordered_labels:
                ordered_labels.append(label)

        # Convert metric column to enum with definition-based ordering
        if "metric" in available_columns and ordered_metrics:
            metric_enum = pl.Enum(ordered_metrics)
            combined = combined.with_columns(pl.col("metric").cast(metric_enum))

        # Convert label column to enum with definition-based ordering
        if "label" in available_columns and ordered_labels:
            label_enum = pl.Enum(ordered_labels)
            combined = combined.with_columns(pl.col("label").cast(label_enum))

        # Convert estimate column to enum with input-based ordering
        if "estimate" in available_columns and self.estimates:
            estimate_enum = pl.Enum(self.estimates)
            combined = combined.with_columns(pl.col("estimate").cast(estimate_enum))

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
        core_columns.extend(["metric", "label", "value", "metric_type", "scope"])
        column_order.extend(core_columns)

        # Sort columns
        sort_cols = []
        potential_sort_cols = self.group_by + ["subgroup_name", "label"]
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

    def _harmonize_result_schemas(
        self, results: list[pl.LazyFrame]
    ) -> list[pl.LazyFrame]:
        """
        Harmonize schemas of result DataFrames to ensure they can be concatenated.

        This addresses the issue where different scopes produce different column structures
        (e.g., some have 'estimate' column, others don't; some have string 'scope', others null).
        """
        if not results:
            return results

        # Collect all unique columns across all results
        all_columns = set()
        schemas = []

        for result in results:
            try:
                schema = result.collect_schema()
                schemas.append(schema)
                all_columns.update(schema.names())
            except Exception:
                # Fallback: collect a small sample to get schema
                sample = result.limit(1).collect()
                schema = sample.schema
                schemas.append(schema)
                all_columns.update(schema.names())

        # Define the target column order and types
        target_columns = []

        # Standard columns that should always be present
        standard_cols = [
            "subject_id",
            "visit_id",  # Entity columns
        ]
        # Add dynamic group_by columns
        standard_cols.extend(self.group_by)
        # Add remaining standard columns
        standard_cols.extend(
            [
                "estimate",  # Model column (may be missing for some scopes)
                "metric",
                "label",
                "value",
                "metric_type",
                "scope",  # Core metric columns
                "subgroup_name",
                "subgroup_value",  # Subgroup columns (may be missing)
            ]
        )

        # Add columns in preferred order
        for col in standard_cols:
            if col in all_columns:
                target_columns.append(col)

        # Add any remaining columns (group_by columns that aren't standard)
        for col in sorted(all_columns):
            if col not in target_columns:
                target_columns.append(col)

        # Harmonize each result
        harmonized = []
        for i, result in enumerate(results):
            schema = schemas[i]

            # Add missing columns with appropriate null values
            exprs = []
            for col in target_columns:
                if col in schema.names():
                    # Handle scope column type conversion
                    if col == "scope" and schema[col] == pl.Null:
                        exprs.append(pl.col(col).cast(pl.Utf8))
                    else:
                        exprs.append(pl.col(col))
                else:
                    # Add missing column with null value of appropriate type
                    if col == "estimate":
                        exprs.append(pl.lit(None, dtype=pl.Utf8).alias(col))
                    elif col == "scope":
                        exprs.append(pl.lit(None, dtype=pl.Utf8).alias(col))
                    elif col in ["subject_id", "visit_id"]:
                        exprs.append(pl.lit(None, dtype=pl.Int64).alias(col))
                    elif col == "value":
                        exprs.append(pl.lit(None, dtype=pl.Float64).alias(col))
                    else:
                        exprs.append(pl.lit(None, dtype=pl.Utf8).alias(col))

            harmonized_result = result.select(exprs)
            harmonized.append(harmonized_result)

        return harmonized
