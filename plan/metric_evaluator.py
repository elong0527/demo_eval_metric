"""
Lazy Evaluation Pipeline for Metrics

This module implements the lazy evaluation pipeline for computing metrics
using Polars LazyFrames, following the architecture design.
"""

import polars as pl
from metric_data import MetricData, MetricType, SharedType
from metric_compiler import MetricCompiler


class MetricEvaluator:
    """Metric evaluation pipeline with complete context at initialization"""
    
    def __init__(self, 
                 df: pl.DataFrame | pl.LazyFrame,
                 metrics: list[MetricData],
                 ground_truth: str = "actual",
                 estimates: list[str] = None,
                 group_by: list[str] = None,
                 filter_expr: pl.Expr = None,
                 compiler: MetricCompiler = None):
        """
        Initialize evaluator with complete evaluation context
        
        Args:
            df: Input data (stored as LazyFrame)
            metrics: List of metrics to evaluate
            ground_truth: Name of ground truth column
            estimates: List of estimate/model columns to evaluate
            group_by: Grouping columns for analysis
            filter_expr: Optional filter expression
            compiler: Metric compiler (creates default if not provided)
        """
        # Store data as LazyFrame
        self.df_raw = df.lazy() if isinstance(df, pl.DataFrame) else df
        
        # Store configuration
        self.metrics = metrics
        self.ground_truth = ground_truth
        self.estimates = estimates or []
        self.group_by = group_by or []
        self.filter_expr = filter_expr
        
        # Initialize expression compiler
        self.compiler = compiler or MetricCompiler()
        
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
    
    def evaluate_single(self, metric: MetricData, estimate: str) -> pl.LazyFrame:
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
        
        # Get metric expressions using the compiler
        agg_exprs, select_expr = self.compiler.compile_expressions(
            metric.name, metric.agg_expr, metric.select_expr
        )
        
        # Determine grouping based on metric type and shared_by
        agg_groups, select_groups = self._get_grouping_columns(
            metric.type, metric.shared_by, estimate
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
    
    def evaluate_all(self) -> pl.DataFrame:
        """
        Evaluate all configured metrics for all estimates
        
        Returns:
            Combined DataFrame with all results
        """
        if not self.metrics or not self.estimates:
            raise ValueError("Metrics and estimates must be set in initialization")
        
        results = []
        for estimate in self.estimates:
            for metric in self.metrics:
                result = self.evaluate_single(metric, estimate)
                results.append(result)
        
        return pl.concat(results).collect()
    
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
                             shared_by: SharedType | None,
                             estimate: str) -> tuple[list[str] | None, list[str] | None]:
        """
        Determine grouping columns based on metric type and shared_by
        
        Args:
            metric_type: Type of metric aggregation
            shared_by: How the metric is shared (ALL, GROUP, or None)
            estimate: Current estimate/model column name
            
        Returns:
            Tuple of (first_level_groups, second_level_groups)
        """
        # Apply SharedType to modify group_by
        effective_group_by = self._apply_shared_type(shared_by)
        
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
    
    def _apply_shared_type(self, shared_by: SharedType | None) -> list[str]:
        """
        Apply SharedType to determine effective grouping columns
        
        Args:
            shared_by: How the metric is shared
            
        Returns:
            Effective group_by columns
        """
        if shared_by == SharedType.ALL:
            # No grouping - aggregate across everything
            return []
        elif shared_by == SharedType.GROUP:
            # Group by group columns only (same value for all models)
            return self.group_by
        elif shared_by == SharedType.MODEL:
            # This doesn't make sense since we calculate per model already
            # Could either raise error or treat as default
            return self.group_by
        else:
            # Default: use configured group_by
            return self.group_by