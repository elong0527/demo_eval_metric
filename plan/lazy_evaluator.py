"""
Lazy Evaluation Pipeline for Metrics

This module implements the lazy evaluation pipeline for computing metrics
using Polars LazyFrames, following the architecture design.
"""

import polars as pl
from metric import Metric, MetricType


class LazyEvaluator:
    """Lazy evaluation pipeline for metrics using Polars"""
    
    def __init__(self, group_cols: list[str] = None, subgroup_cols: list[str] = None):
        """
        Initialize evaluator with column configurations
        
        Args:
            group_cols: Group columns for analysis
            subgroup_cols: Subgroup columns for analysis
        """
        self.group_cols = group_cols or []
        self.subgroup_cols = subgroup_cols or []
        self.analysis_cols = self.group_cols + self.subgroup_cols
    
    def prepare_data(self, df: pl.DataFrame | pl.LazyFrame, 
                    ground_truth: str, estimate: str,
                    filter_expr: pl.Expr = None) -> pl.LazyFrame:
        """
        Stage 1: Data preparation with error columns and optional filtering
        
        Args:
            df: Input data
            ground_truth: Name of ground truth column
            estimate: Name of estimate/prediction column
            filter_expr: Optional filter expression
        
        Returns:
            LazyFrame with computed error columns
        """
        lf = df.lazy() if isinstance(df, pl.DataFrame) else df
        
        # Add error columns for metric computation
        error = pl.col(estimate) - pl.col(ground_truth)
        lf = lf.with_columns([
            error.alias("error"),
            error.abs().alias("absolute_error"),
            (error ** 2).alias("squared_error"),
            pl.when(pl.col(ground_truth) != 0)
                .then(error / pl.col(ground_truth) * 100)
                .otherwise(None)
                .alias("percent_error"),
            pl.when(pl.col(ground_truth) != 0)
                .then((error / pl.col(ground_truth) * 100).abs())
                .otherwise(None)
                .alias("absolute_percent_error"),
        ])
        
        # Apply filter if provided
        if filter_expr is not None:
            lf = lf.filter(filter_expr)
        
        return lf
    
    def aggregate_first_level(self, lf: pl.LazyFrame, 
                             agg_exprs: list[pl.Expr],
                             agg_groups: list[str]) -> pl.LazyFrame:
        """
        Stage 2: First-level aggregation
        
        Args:
            lf: Input LazyFrame
            agg_exprs: Aggregation expressions
            agg_groups: Grouping columns for aggregation
        
        Returns:
            Aggregated LazyFrame
        """
        if agg_groups:
            return lf.group_by(agg_groups).agg(agg_exprs)
        return lf.select(agg_exprs)
    
    def select_final(self, lf: pl.LazyFrame,
                    select_expr: pl.Expr,
                    select_groups: list[str]) -> pl.LazyFrame:
        """
        Stage 3: Final selection/aggregation
        
        Args:
            lf: Input LazyFrame
            select_expr: Selection expression
            select_groups: Grouping columns for final result
        
        Returns:
            LazyFrame with final metric value
        """
        # Ensure value is always Float64 for consistency
        value_expr = select_expr.alias("value").cast(pl.Float64)
        
        if select_groups:
            return lf.group_by(select_groups).agg(value_expr)
        else:
            return lf.select(value_expr)
    
    def add_metadata(self, lf: pl.LazyFrame,
                    metric_name: str,
                    estimate_name: str) -> pl.LazyFrame:
        """
        Stage 4: Add metadata columns
        
        Args:
            lf: Input LazyFrame
            metric_name: Name of the metric
            estimate_name: Name of the estimate/model
        
        Returns:
            LazyFrame with metadata columns
        """
        return lf.with_columns([
            pl.lit(metric_name).alias("metric"),
            pl.lit(estimate_name).alias("estimate"),
            pl.lit(metric_name).alias("label"),  # Can be customized later
        ])
    
    def evaluate_pipeline(self, df: pl.DataFrame | pl.LazyFrame,
                         metric: Metric,
                         ground_truth: str,
                         estimate: str,
                         use_group: bool = True,
                         use_subgroup: bool = True,
                         filter_expr: pl.Expr = None) -> pl.LazyFrame:
        """
        Complete evaluation pipeline for a single metric-model combination
        
        Args:
            df: Input data
            metric: Metric definition
            ground_truth: Ground truth column name
            estimate: Estimate/model column name
            use_group: Whether to use group columns
            use_subgroup: Whether to use subgroup columns
            filter_expr: Optional filter expression
        
        Returns:
            LazyFrame with evaluation results
        """
        # Determine analysis columns
        analysis_cols = []
        if use_subgroup:
            analysis_cols.extend(self.subgroup_cols)
        if use_group:
            analysis_cols.extend(self.group_cols)
        
        # Get expressions from metric
        agg_exprs, select_expr = metric.get_polars_expressions()
        
        # Determine grouping based on metric type
        agg_groups, select_groups = self._get_grouping_columns(metric.type, analysis_cols)
        
        # Build pipeline
        pipeline = self.prepare_data(df, ground_truth, estimate, filter_expr)
        
        # Apply first-level aggregation if needed
        if agg_groups is not None and agg_exprs:
            pipeline = self.aggregate_first_level(pipeline, agg_exprs, agg_groups)
        
        # Apply final selection
        if select_groups is not None:
            # Apply second-level aggregation
            if select_expr is not None:
                pipeline = self.select_final(pipeline, select_expr, select_groups)
            elif agg_exprs:
                # If no select_expr but have agg_exprs, use them directly
                pipeline = self.select_final(pipeline, agg_exprs[0], select_groups)
        # else: No second-level aggregation for within_subject/within_visit
        
        # Add metadata
        pipeline = self.add_metadata(pipeline, metric.name, estimate)
        
        # Add metric type for reference
        pipeline = pipeline.with_columns(
            pl.lit(metric.type.value).alias("metric_type")
        )
        
        return pipeline
    
    def _get_grouping_columns(self, metric_type: MetricType, 
                             analysis_cols: list[str]) -> tuple[list[str] | None, list[str]]:
        """
        Determine grouping columns based on metric type
        
        Args:
            metric_type: Type of metric
            analysis_cols: Analysis columns (group + subgroup)
        
        Returns:
            Tuple of (agg_groups, select_groups)
        """
        if metric_type == MetricType.ACROSS_SAMPLES:
            # No first-level aggregation, only final grouping
            return None, analysis_cols
        
        elif metric_type == MetricType.WITHIN_SUBJECT:
            # Group by subject in first level, output keeps subject_id
            # No second-level aggregation
            return ["subject_id"] + analysis_cols, None
        
        elif metric_type == MetricType.ACROSS_SUBJECT:
            # Two-level: aggregate by subject, then across subjects
            return ["subject_id"] + analysis_cols, analysis_cols
        
        elif metric_type == MetricType.WITHIN_VISIT:
            # Group by subject and visit in first level, output keeps both
            # No second-level aggregation
            return ["subject_id", "visit_id"] + analysis_cols, None
        
        elif metric_type == MetricType.ACROSS_VISIT:
            # Two-level: aggregate by visit, then across visits
            return ["subject_id", "visit_id"] + analysis_cols, analysis_cols
        
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")
    
    def evaluate_multiple(self, df: pl.DataFrame | pl.LazyFrame,
                         metrics: list[Metric],
                         ground_truth: str,
                         estimates: list[str],
                         **kwargs) -> pl.DataFrame:
        """
        Evaluate multiple metrics for multiple models
        
        Args:
            df: Input data
            metrics: List of metric definitions
            ground_truth: Ground truth column name
            estimates: List of estimate/model column names
            **kwargs: Additional arguments for evaluate_pipeline
        
        Returns:
            Combined DataFrame with all results
        """
        results = []
        
        for estimate in estimates:
            for metric in metrics:
                result = self.evaluate_pipeline(
                    df, metric, ground_truth, estimate, **kwargs
                )
                results.append(result)
        
        # Combine and collect all results
        return pl.concat(results).collect()