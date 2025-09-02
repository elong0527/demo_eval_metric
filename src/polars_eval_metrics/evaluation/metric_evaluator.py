"""
Lazy Evaluation Pipeline for Metrics

This module implements the lazy evaluation pipeline for computing metrics
using Polars LazyFrames, following the architecture design.
"""

import polars as pl
from ..core import MetricDefine, MetricType, MetricScope


class MetricEvaluator:
    """Metric evaluation pipeline with complete context at initialization"""
    
    def __init__(self, 
                 df: pl.DataFrame | pl.LazyFrame,
                 metrics: MetricDefine | list[MetricDefine],
                 ground_truth: str = "actual",
                 estimates: str | list[str] = None,
                 group_by: list[str] = None,
                 subgroup_by: list[str] = None,
                 filter_expr: pl.Expr = None):
        """
        Initialize evaluator with complete evaluation context
        
        Args:
            df: Input data (stored as LazyFrame)
            metrics: Single metric or list of metrics to evaluate
            ground_truth: Name of ground truth column
            estimates: Single estimate column or list of estimate columns
            group_by: Grouping columns for analysis
            subgroup_by: Additional subgrouping columns (combined with group_by)
            filter_expr: Optional filter expression
        """
        # Store data as LazyFrame
        self.df_raw = df.lazy() if isinstance(df, pl.DataFrame) else df
        
        # Store configuration - normalize to lists
        self.metrics = [metrics] if isinstance(metrics, MetricDefine) else metrics
        self.ground_truth = ground_truth
        self.estimates = [estimates] if isinstance(estimates, str) else (estimates or [])
        self.group_by = group_by or []
        self.subgroup_by = subgroup_by or []
        self.filter_expr = filter_expr
        
        # Prepare data once with filter
        self.df = self._prepare_base_data()
    
    def _prepare_base_data(self) -> pl.LazyFrame:
        """Apply filter if provided"""
        if self.filter_expr is not None:
            return self.df_raw.filter(self.filter_expr)
        return self.df_raw
    
    def _prepare_error_columns(self, df: pl.LazyFrame, estimate: str) -> pl.LazyFrame:
        """Add error columns for a specific estimate"""
        error = pl.col(estimate) - pl.col(self.ground_truth)
        
        return df.with_columns([
            error.alias("error"),
            error.abs().alias("absolute_error"),
            (error ** 2).alias("squared_error"),
            pl.when(pl.col(self.ground_truth) != 0)
                .then(error / pl.col(self.ground_truth) * 100)
                .otherwise(None)
                .alias("percent_error"),
            pl.when(pl.col(self.ground_truth) != 0)
                .then((error / pl.col(self.ground_truth) * 100).abs())
                .otherwise(None)
                .alias("absolute_percent_error"),
        ])
    
    def evaluate_single(self, metric: MetricDefine, estimate: str) -> pl.LazyFrame:
        """
        Evaluate a single metric for one estimate
        
        Args:
            metric: Metric to evaluate
            estimate: Estimate column
            
        Returns:
            LazyFrame with evaluation result
        """
        if not metric or not estimate:
            raise ValueError("Metric and estimate must be provided")
        
        # Prepare data with error columns
        df_prep = self._prepare_error_columns(self.df, estimate)
        
        # Get metric expressions
        agg_exprs, select_expr = metric.compile_expressions()
        
        # Determine grouping based on metric type and scope
        agg_groups, select_groups = self._get_grouping_columns(
            metric.type, metric.scope, estimate
        )
        
        # Build pipeline
        pipeline = self._build_pipeline(df_prep, agg_exprs, select_expr, 
                                       agg_groups, select_groups)
        
        # Add metadata
        return pipeline.with_columns([
            pl.lit(metric.name).alias("metric"),
            pl.lit(estimate).alias("estimate"),
            pl.lit(metric.label).alias("label"),
            pl.lit(metric.type.value).alias("metric_type")
        ])
    
    def _evaluate_single_with_subgroup(self, metric: MetricDefine, estimate: str, subgroup_var: str) -> pl.LazyFrame:
        """
        Evaluate a single metric for one estimate with a specific subgroup variable
        
        Args:
            metric: Metric to evaluate
            estimate: Estimate column
            subgroup_var: Subgroup variable to include in grouping
            
        Returns:
            LazyFrame with evaluation results including subgroup_name and subgroup_value columns
        """
        if not metric or not estimate:
            raise ValueError("Metric and estimate must be provided")
        
        # Prepare data with error columns
        df_prep = self._prepare_error_columns(self.df, estimate)
        
        # Get metric expressions
        agg_exprs, select_expr = metric.compile_expressions()
        
        # Determine grouping based on metric type and scope, but with the subgroup
        # Temporarily modify group_by to include the subgroup for this calculation
        original_group_by = self.group_by
        try:
            self.group_by = self.group_by + [subgroup_var]
            agg_groups, select_groups = self._get_grouping_columns(
                metric.type, metric.scope, estimate
            )
        finally:
            # Restore original group_by
            self.group_by = original_group_by
        
        # Build pipeline
        pipeline = self._build_pipeline(df_prep, agg_exprs, select_expr, 
                                       agg_groups, select_groups)
        
        # Add metadata including subgroup information
        return pipeline.with_columns([
            pl.lit(metric.name).alias("metric"),
            pl.lit(estimate).alias("estimate"),
            pl.lit(metric.label).alias("label"),
            pl.lit(metric.type.value).alias("metric_type"),
            pl.lit(subgroup_var).alias("subgroup_name"),
            pl.col(subgroup_var).alias("subgroup_value")
        ])
    
    def evaluate(self, 
                 metrics: MetricDefine | list[MetricDefine] | None = None,
                 estimates: str | list[str] | None = None,
                 collect: bool = True) -> pl.DataFrame | pl.LazyFrame:
        """
        Evaluate metrics for estimates
        
        Args:
            metrics: Optional subset of metrics to evaluate. If None, uses all configured metrics.
            estimates: Optional subset of estimates to evaluate. If None, uses all configured estimates.
            collect: Whether to collect the LazyFrame into a DataFrame. Default is True for backward compatibility.
        
        Returns:
            DataFrame if collect=True, LazyFrame if collect=False
        """
        # Determine which metrics to use
        if metrics is None:
            metrics_to_eval = self.metrics
        else:
            # Normalize to list
            metrics_list = [metrics] if isinstance(metrics, MetricDefine) else metrics
            # Validate that requested metrics are in configured metrics
            configured_names = {m.name for m in self.metrics}
            for m in metrics_list:
                if m.name not in configured_names:
                    raise ValueError(f"Metric '{m.name}' not in configured metrics")
            metrics_to_eval = metrics_list
        
        # Determine which estimates to use
        if estimates is None:
            estimates_to_eval = self.estimates
        else:
            # Normalize to list
            estimates_list = [estimates] if isinstance(estimates, str) else estimates
            # Validate that requested estimates are in configured estimates
            for e in estimates_list:
                if e not in self.estimates:
                    raise ValueError(f"Estimate '{e}' not in configured estimates: {self.estimates}")
            estimates_to_eval = estimates_list
        
        if not metrics_to_eval or not estimates_to_eval:
            raise ValueError("No metrics or estimates to evaluate")
        
        # Handle subgroup_by by creating separate evaluations
        if self.subgroup_by:
            # For now, collect all results separately and then concatenate
            all_results = []
            
            # Create evaluation for each subgroup variable
            for subgroup_var in self.subgroup_by:
                for estimate in estimates_to_eval:
                    for metric in metrics_to_eval:
                        # Create a modified evaluation with the subgroup
                        result = self._evaluate_single_with_subgroup(metric, estimate, subgroup_var)
                        # Collect immediately to avoid schema issues
                        collected_result = result.collect()
                        all_results.append(collected_result)
            
            # Combine all collected DataFrames with schema alignment
            if all_results:
                # Ensure all DataFrames have the same columns by adding missing ones with null
                all_columns = set()
                for df in all_results:
                    all_columns.update(df.columns)
                
                # Standardize all DataFrames to have the same columns
                standardized_dfs = []
                for df in all_results:
                    missing_cols = all_columns - set(df.columns)
                    if missing_cols:
                        # Add missing columns as null with proper types
                        for col in missing_cols:
                            # Assume string type for grouping columns
                            df = df.with_columns(pl.lit(None, dtype=pl.String).alias(col))
                    # Ensure consistent column order
                    df = df.select(sorted(all_columns))
                    standardized_dfs.append(df)
                
                combined_results = pl.concat([df.lazy() for df in standardized_dfs])
            else:
                raise ValueError("No results generated")
        else:
            # Original logic when no subgroup_by
            results = []
            for estimate in estimates_to_eval:
                for metric in metrics_to_eval:
                    result = self.evaluate_single(metric, estimate)
                    results.append(result)
            
            # Combine all lazy frames
            combined_results = pl.concat(results)
        
        # Use a safer approach to determine available columns
        try:
            available_columns = combined_results.collect_schema().names()
        except:
            # Fallback: collect a small sample to get column names
            available_columns = combined_results.limit(1).collect().columns
        
        sort_cols = []
        
        # Add grouping columns that exist in the result
        potential_group_cols = self.group_by
        for col in potential_group_cols:
            if col in available_columns:
                sort_cols.append(col)
        
        # Add subgroup columns if they exist
        if "subgroup_name" in available_columns:
            sort_cols.extend(["subgroup_name", "subgroup_value"])
        
        sort_cols.extend(["metric", "estimate"])
        
        # Sort by available grouping columns, then metric, then estimate
        sorted_results = combined_results.sort(sort_cols)
        
        # Arrange columns in logical order - dynamically determine grouping columns
        column_order = []
        
        # First: add grouping columns that exist in the result
        for col in potential_group_cols:
            if col in available_columns:
                column_order.append(col)
        
        # Add subgroup columns if they exist
        if "subgroup_name" in available_columns:
            column_order.extend(["subgroup_name", "subgroup_value"])
        
        # Then: estimate and metric identifiers
        column_order.extend(["estimate", "metric", "label"])
        
        # Finally: the value and metadata
        column_order.extend(["value", "metric_type"])
        
        # Select columns in the desired order
        result = sorted_results.select(column_order)
        
        # Return DataFrame or LazyFrame based on collect parameter
        return result.collect() if collect else result
    
    
    def _build_pipeline(self, df: pl.LazyFrame, 
                       agg_exprs: list[pl.Expr], 
                       select_expr: pl.Expr,
                       agg_groups: list[str] | None,
                       select_groups: list[str] | None) -> pl.LazyFrame:
        """Build the evaluation pipeline"""
        pipeline = df
        
        # First-level aggregation if needed
        if agg_groups is not None and agg_exprs:
            if agg_groups:
                pipeline = pipeline.group_by(agg_groups).agg(agg_exprs)
            else:
                pipeline = pipeline.select(agg_exprs)
        
        # Second-level selection if needed
        if select_groups is not None:
            if select_expr is not None:
                value_expr = select_expr.alias("value").cast(pl.Float64)
            elif agg_exprs:
                value_expr = agg_exprs[0].alias("value").cast(pl.Float64)
            else:
                raise ValueError("No expression available for selection")
            
            if select_groups:
                pipeline = pipeline.group_by(select_groups).agg(value_expr)
            else:
                pipeline = pipeline.select(value_expr)
        
        return pipeline
    
    def _get_grouping_columns(self, 
                             metric_type: MetricType,
                             scope: MetricScope | None,
                             estimate: str) -> tuple[list[str] | None, list[str] | None]:
        """
        Determine grouping columns based on metric type and scope
        
        Args:
            metric_type: Type of metric aggregation
            scope: Metric calculation scope (GLOBAL, MODEL, GROUP, or None)
            estimate: Current estimate/model column name
            
        Returns:
            Tuple of (first_level_groups, second_level_groups)
        """
        # Apply MetricScope to modify group_by
        effective_group_by = self._apply_metric_scope(scope)
        
        # Define grouping rules as a mapping
        grouping_rules = {
            MetricType.ACROSS_SAMPLES: (None, effective_group_by),
            MetricType.WITHIN_SUBJECT: (["subject_id"] + effective_group_by, None),
            MetricType.ACROSS_SUBJECT: (["subject_id"] + effective_group_by, effective_group_by),
            MetricType.WITHIN_VISIT: (["subject_id", "visit_id"] + effective_group_by, None),
            MetricType.ACROSS_VISIT: (["subject_id", "visit_id"] + effective_group_by, effective_group_by),
        }
        
        # Get the rule for the metric type
        rule = grouping_rules.get(metric_type)
        
        if rule is None:
            raise ValueError(f"Unknown metric type: {metric_type}")
        
        return rule
    
    def _apply_metric_scope(self, scope: MetricScope | None) -> list[str]:
        """
        Apply MetricScope to determine effective grouping columns
        
        Args:
            scope: Metric calculation scope (GLOBAL, MODEL, or GROUP)
            
        Returns:
            Effective group_by columns
        """
        if scope == MetricScope.GLOBAL:
            # No grouping - aggregate across everything
            return []
        elif scope == MetricScope.MODEL:
            # No grouping - per model metrics ignore group_by columns
            # The model separation happens at evaluate_single level
            return []
        elif scope == MetricScope.GROUP:
            # Group-only scope - this will be handled specially in evaluation
            # to aggregate across models for each group
            return self.group_by
        elif scope is None:
            # Default: use configured group_by columns (per model-group)
            return self.group_by
        else:
            # Unexpected scope value
            raise ValueError(f"Unknown scope: {scope}")