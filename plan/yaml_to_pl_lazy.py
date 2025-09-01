"""
YAML to Polars Lazy Expression Translator - Simplified

With expression building moved to Metric class, the translator becomes much simpler.
"""

import polars as pl
import yaml
from metric import Metric, MetricType


class YAMLToPolarsLazyTranslator:
    """Translates YAML metric definitions to Polars lazy expressions"""
    
    # Base columns for each metric type
    BASE_COLUMNS = {
        MetricType.ACROSS_SAMPLES: [],
        MetricType.WITHIN_SUBJECT: ["subject_id"],
        MetricType.ACROSS_SUBJECT: ["subject_id"],
        MetricType.WITHIN_VISIT: ["subject_id", "visit_id"],
        MetricType.ACROSS_VISIT: ["subject_id", "visit_id"],
    }
    
    def __init__(self, yaml_config: str | dict):
        """Initialize translator with YAML configuration"""
        # Load configuration
        self.config = self._load_config(yaml_config)
        
        # Extract column mappings
        columns = self.config.get('columns', {})
        self.group_cols = columns.get('group', [])
        self.subgroup_cols = columns.get('subgroup', [])
        
        # Parse metrics (Metric class handles all normalization)
        self.metrics = [Metric.from_yaml(config) for config in self.config.get('metrics', [])]
        
        # Initialize aggregation strategies
        self._init_strategies()
    
    def _load_config(self, yaml_config: str | dict) -> dict:
        """Load YAML configuration"""
        if isinstance(yaml_config, str):
            with open(yaml_config, 'r') as f:
                return yaml.safe_load(f)
        return yaml_config
    
    def _init_strategies(self):
        """Initialize aggregation strategies using dispatch table"""
        # Single-level aggregation strategy
        def single_level(lf, metric, analysis_cols):
            # Get expressions from metric
            agg_exprs, select_expr = metric.get_polars_expressions()
            
            # Determine grouping columns
            group_cols = analysis_cols + self.BASE_COLUMNS[metric.type]
            
            # Perform aggregation (Metric already handles 'value' alias)
            return self._aggregate(lf, group_cols, agg_exprs)
        
        # Two-level aggregation strategy
        def two_level(lf, metric, analysis_cols):
            # Get expressions from metric
            agg_exprs, select_expr = metric.get_polars_expressions()
            
            # Base columns for this metric type
            base_cols = self.BASE_COLUMNS[metric.type]
            
            # First level: aggregate by subject/visit
            first_agg = self._aggregate(lf, analysis_cols + base_cols, agg_exprs)
            
            final_expr = select_expr.alias("value")
            return self._aggregate(first_agg, analysis_cols, [final_expr])
        
        # Dispatch table
        self.aggregation_strategies = {
            MetricType.ACROSS_SAMPLES: single_level,
            MetricType.WITHIN_SUBJECT: single_level,
            MetricType.WITHIN_VISIT: single_level,
            MetricType.ACROSS_SUBJECT: two_level,
            MetricType.ACROSS_VISIT: two_level,
        }
    
    def prepare_dataframe(self, df: pl.DataFrame | pl.LazyFrame, 
                         ground_truth: str, estimate: str) -> pl.LazyFrame:
        """Add error columns needed for metrics calculation"""
        lf = df.lazy() if isinstance(df, pl.DataFrame) else df
        error = pl.col(estimate) - pl.col(ground_truth)
        
        return lf.with_columns([
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
    
    def _aggregate(self, lf: pl.LazyFrame, group_cols: list[str], 
                  agg_exprs: list[pl.Expr]) -> pl.LazyFrame:
        """Perform aggregation (handles both grouped and ungrouped)"""
        if not agg_exprs:
            raise ValueError("No aggregation expressions provided")
        
        return lf.group_by(group_cols).agg(agg_exprs) if group_cols else lf.select(agg_exprs)
    
    def get_lazy_expression(self, lf: pl.LazyFrame, metric: Metric,
                           use_group: bool = False, use_subgroup: bool = False) -> pl.LazyFrame:
        """Generate the Polars lazy expression for a metric
        
        This is now much simpler since Metric handles expression building.
        """
        # Collect analysis columns
        analysis_cols = []
        if use_subgroup:
            analysis_cols.extend(self.subgroup_cols)
        if use_group:
            analysis_cols.extend(self.group_cols)
        
        # Apply aggregation strategy based on metric type
        strategy = self.aggregation_strategies[metric.type]
        result = strategy(lf, metric, analysis_cols)
        
        # Add metadata
        return result.with_columns([
            pl.lit(metric.name).alias("metric"),
            pl.lit(metric.label).alias("label"),
            pl.lit(metric.type.value).alias("metric_type")
        ])