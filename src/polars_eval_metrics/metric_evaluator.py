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

        # Vectorized evaluation - single Polars operation
        combined = self._vectorized_evaluate(target_metrics, target_estimates)

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
            return self._evaluate_with_marginal_subgroups(df_with_errors, metrics, estimates)
        else:
            return self._evaluate_without_subgroups(df_with_errors, metrics, estimates)

    def _evaluate_without_subgroups(
        self, df_with_errors: pl.LazyFrame, metrics: list[MetricDefine], estimates: list[str]
    ) -> pl.LazyFrame:
        """Evaluate metrics without subgroup analysis"""
        results = []
        for metric in metrics:
            metric_result = self._evaluate_metric_vectorized(df_with_errors, metric, estimates)
            results.append(metric_result)
        
        # Combine all metric results
        return pl.concat(results, how="diagonal")

    def _evaluate_with_marginal_subgroups(
        self, df_with_errors: pl.LazyFrame, metrics: list[MetricDefine], estimates: list[str]
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
                    metric_result = self._evaluate_metric_vectorized(df_with_errors, metric, estimates)
                    
                    # Add subgroup metadata columns
                    metric_result = metric_result.with_columns([
                        pl.lit(subgroup_col).alias("subgroup_name"),
                        pl.col(subgroup_col).cast(pl.Utf8).alias("subgroup_value")
                    ])
                    
                    # Remove the original subgroup column to avoid duplication
                    available_cols = metric_result.collect_schema().names()
                    if subgroup_col in available_cols:
                        cols_to_keep = [col for col in available_cols if col != subgroup_col]
                        metric_result = metric_result.select(cols_to_keep)
                    
                    results.append(metric_result)
                    
                finally:
                    # Restore original subgroup_by
                    self.subgroup_by = original_subgroup_by
        
        # Combine all results
        return pl.concat(results, how="diagonal")

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
            value_name="estimate_value"
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
                raise ValueError(f"ACROSS_SAMPLE metric {metric.name} requires across_expr")
                
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
                raise ValueError(f"No valid expression for first level of metric {metric.name}")
                
            intermediate = df_filtered.group_by(entity_groups).agg(first_level_expr)
            
            # Second level: across entities  
            if across_expr is not None and within_exprs:
                # True two-level case - substitute 'value' with 'intermediate_value' in the expression
                # For now, use the across expression directly but this may need column name mapping
                # TODO: More sophisticated expression rewriting to handle column references
                second_level_expr = pl.col("intermediate_value").mean().alias("value").cast(pl.Float64)
            else:
                # Default aggregation across entities
                second_level_expr = pl.col("intermediate_value").mean().alias("value").cast(pl.Float64)
            
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

    def _add_metadata_vectorized(self, result: pl.LazyFrame, metric: MetricDefine) -> pl.LazyFrame:
        """Add metadata columns to vectorized result"""
        
        metadata = [
            pl.lit(metric.name).alias("metric"),
            pl.lit(metric.label or metric.name).alias("label"), 
            pl.lit(metric.type.value).alias("metric_type"),
            pl.lit(metric.scope.value if metric.scope else None).alias("scope"),
        ]
        
        # Rename estimate_name to estimate for compatibility
        result_with_metadata = result.with_columns(metadata)
        
        # Check if estimate_name column exists and rename it
        schema = result_with_metadata.collect_schema()
        if "estimate_name" in schema.names():
            result_with_metadata = result_with_metadata.rename({"estimate_name": "estimate"})
        
        return result_with_metadata

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
        core_columns.extend(["metric", "label", "value", "metric_type", "scope"])
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