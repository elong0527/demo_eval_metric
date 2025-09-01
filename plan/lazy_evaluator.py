"""
Lazy Evaluation Pipeline for Metrics

This module implements the lazy evaluation pipeline for computing metrics
using Polars LazyFrames, following the architecture design.
"""

import polars as pl
from metric import Metric, MetricType


class LazyEvaluator:
    """Lazy evaluation pipeline with complete context at initialization"""
    
    def __init__(self, 
                 df: pl.DataFrame | pl.LazyFrame,
                 metrics: list[Metric],
                 ground_truth: str = "actual",
                 estimates: list[str] = None,
                 group_by: list[str] = None,
                 filter_expr: pl.Expr = None):
        """
        Initialize evaluator with complete evaluation context
        
        Args:
            df: Input data (stored as LazyFrame)
            metrics: List of metrics to evaluate
            ground_truth: Name of ground truth column
            estimates: List of estimate/model columns to evaluate
            group_by: Grouping columns for analysis
            filter_expr: Optional filter expression
        """
        # Store data as LazyFrame
        self.df_raw = df.lazy() if isinstance(df, pl.DataFrame) else df
        
        # Store configuration
        self.metrics = metrics
        self.ground_truth = ground_truth
        self.estimates = estimates or []
        self.group_by = group_by or []
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
    
    def evaluate_single(self, metric: Metric, estimate: str) -> pl.LazyFrame:
        """
        Evaluate a single metric for one estimate
        
        Args:
            metric: Metric to evaluate (uses first from self.metrics if None)
            estimate: Estimate column (uses first from self.estimates if None)
            group_by: Override grouping columns (uses self.group_by if None)
        
        Returns:
            LazyFrame with evaluation result
        """
        group_by = self.group_by
        
        if not metric or not estimate:
            raise ValueError("Metric and estimate must be provided or set in initialization")
        
        # Prepare data with error columns
        df_prep = self._prepare_error_columns(self.df, estimate)
        
        # Get metric expressions
        agg_exprs, select_expr = metric.get_polars_expressions()
        
        # Determine grouping based on metric type
        agg_groups, select_groups = self._get_grouping_columns(metric.type, group_by)
        
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
    
    def _get_grouping_columns(self, metric_type: MetricType, 
                             group_by: list[str]) -> tuple[list[str] | None, list[str] | None]:
        """Determine grouping columns based on metric type"""
        if metric_type == MetricType.ACROSS_SAMPLES:
            return None, group_by
        elif metric_type == MetricType.WITHIN_SUBJECT:
            return ["subject_id"] + group_by, None
        elif metric_type == MetricType.ACROSS_SUBJECT:
            return ["subject_id"] + group_by, group_by
        elif metric_type == MetricType.WITHIN_VISIT:
            return ["subject_id", "visit_id"] + group_by, None
        elif metric_type == MetricType.ACROSS_VISIT:
            return ["subject_id", "visit_id"] + group_by, group_by
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")
    
