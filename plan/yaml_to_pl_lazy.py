"""
YAML to Polars Lazy Expression Translator

Minimal implementation focused solely on translating YAML metric definitions 
into Polars lazy expressions.
"""

import polars as pl
from dataclasses import dataclass
from enum import Enum
import yaml


class MetricType(Enum):
    """Metric aggregation types"""
    ACROSS_SAMPLES = "across_samples"
    ACROSS_SUBJECT = "across_subject"
    WITHIN_SUBJECT = "within_subject"
    ACROSS_VISIT = "across_visit"
    WITHIN_VISIT = "within_visit"


class SharedType(Enum):
    """Shared semantics for metric calculation optimization"""
    MODEL = "model"
    ALL = "all"
    GROUP = "group"


@dataclass
class MetricDefinition:
    """Parsed metric definition from YAML"""
    name: str
    label: str
    type: MetricType
    shared_by: SharedType | None = None
    agg_expr: list[str] | None = None
    select_expr: str | None = None


class YAMLToPolarsLazyTranslator:
    """Translates YAML metric definitions to Polars lazy expressions"""
    
    # Built-in metric expressions
    BUILTIN_METRICS = {
        "mae": pl.col("absolute_error").mean(),
        "mse": pl.col("squared_error").mean(),
        "rmse": pl.col("squared_error").mean().sqrt(),
        "bias": pl.col("error").mean(),
        "mape": pl.col("absolute_percent_error").mean(),
        "n_subject": pl.col("subject_id").n_unique(),
        "n_visit": pl.struct(["subject_id", "visit_id"]).n_unique(),
        "n_sample": pl.len(),
        "total_subject": pl.col("subject_id").n_unique(),
        "total_visit": pl.struct(["subject_id", "visit_id"]).n_unique(),
    }
    
    # Built-in selection expressions (all use _value column)
    BUILTIN_SELECTORS = {
        "mean": pl.col("_value").mean(),
        "median": pl.col("_value").median(),
        "std": pl.col("_value").std(),
        "min": pl.col("_value").min(),
        "max": pl.col("_value").max(),
        "sum": pl.col("_value").sum(),
        "sqrt": pl.col("_value").sqrt(),
    }
    
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
        if isinstance(yaml_config, str):
            with open(yaml_config, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = yaml_config
        
        # Extract column mappings
        columns = self.config.get('columns', {})
        self.group_cols = columns.get('group', [])
        self.subgroup_cols = columns.get('subgroup', [])
        
        # Parse metrics
        self.metrics = self._parse_metrics()
    
    def _parse_metrics(self) -> list[MetricDefinition]:
        """Parse metric definitions from configuration"""
        metrics = []
        
        for metric_config in self.config.get('metrics', []):
            # Parse type
            metric_type = MetricType(metric_config.get('type', 'across_samples'))
            
            # Parse shared_by
            shared_by_value = metric_config.get('shared_by')
            shared_type = None
            if shared_by_value:
                shared_map = {'model': SharedType.MODEL, 'all': SharedType.ALL, 'group': SharedType.GROUP}
                shared_type = shared_map.get(shared_by_value)
            
            # Parse expressions (normalize to list)
            agg_config = metric_config.get('agg', {})
            agg_expr = None
            if isinstance(agg_config, dict) and 'expr' in agg_config:
                raw_expr = agg_config['expr']
                agg_expr = [raw_expr] if isinstance(raw_expr, str) else raw_expr
            
            select_config = metric_config.get('select', {})
            select_expr = select_config.get('expr') if isinstance(select_config, dict) else None
            
            metrics.append(MetricDefinition(
                name=metric_config['name'],
                label=metric_config.get('label', metric_config['name']),
                type=metric_type,
                shared_by=shared_type,
                agg_expr=agg_expr,
                select_expr=select_expr
            ))
        
        return metrics
    
    def prepare_dataframe(self, df: pl.DataFrame | pl.LazyFrame, 
                         ground_truth: str, estimate: str) -> pl.LazyFrame:
        """Add error columns needed for metrics calculation"""
        if isinstance(df, pl.DataFrame):
            df = df.lazy()
        
        error = pl.col(estimate) - pl.col(ground_truth)
        
        return df.with_columns([
            error.alias("error"),
            error.abs().alias("absolute_error"),
            (error ** 2).alias("squared_error"),
            
            # Percentage errors with division by zero handling
            pl.when(pl.col(ground_truth) != 0)
            .then(error / pl.col(ground_truth) * 100)
            .otherwise(None)
            .alias("percent_error"),
            
            pl.when(pl.col(ground_truth) != 0)
            .then((error / pl.col(ground_truth) * 100).abs())
            .otherwise(None)
            .alias("absolute_percent_error"),
        ])
    
    def _evaluate_expression(self, expr_str: str) -> pl.Expr:
        """Evaluate a Polars expression string"""
        namespace = {
            'pl': pl,
            'col': pl.col,
            'lit': pl.lit,
            'len': pl.len,
            'struct': pl.struct,
            **self.BUILTIN_METRICS  # Include built-in metrics as shortcuts
        }
        
        try:
            return eval(expr_str, namespace, {})
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression '{expr_str}': {e}")
    
    def _build_expressions(self, metric: MetricDefinition) -> tuple[list[pl.Expr], pl.Expr | None]:
        """Build Polars expressions for a metric"""
        # Handle custom expressions
        if metric.agg_expr or metric.select_expr:
            # Parse aggregation expressions
            agg_exprs = [self._evaluate_expression(expr) for expr in metric.agg_expr] if metric.agg_expr else []
            
            # Parse selection expression
            select_expr = self._evaluate_expression(metric.select_expr) if metric.select_expr else None
            
            # Handle case where only select_expr is provided
            if not agg_exprs and select_expr is not None:
                return [select_expr.alias("value")], None
            
            # Add value alias for single expression without select
            if len(agg_exprs) == 1 and select_expr is None:
                agg_exprs = [agg_exprs[0].alias("value")]
            
            return agg_exprs, select_expr
        
        # Handle built-in metric names
        parts = metric.name.split(':', 1)
        agg_name = parts[0]
        select_name = parts[1] if len(parts) > 1 else None
        
        # Get aggregation expression
        agg_expr = self.BUILTIN_METRICS.get(agg_name)
        if agg_expr is None:
            raise ValueError(f"Unknown metric: {agg_name}")
        
        # Get selection expression
        select_expr = self.BUILTIN_SELECTORS.get(select_name) if select_name else None
        
        # Add value alias if no selection expression
        if select_expr is None:
            agg_expr = agg_expr.alias("value")
        
        return [agg_expr], select_expr
    
    def _aggregate(self, lf: pl.LazyFrame, group_cols: list[str], 
                  agg_exprs: list[pl.Expr]) -> pl.LazyFrame:
        """Perform aggregation"""
        if not agg_exprs:
            raise ValueError("No aggregation expressions provided")
        
        if group_cols:
            return lf.group_by(group_cols).agg(agg_exprs)
        else:
            return lf.select(agg_exprs)
    
    def get_lazy_expression(self, lf: pl.LazyFrame, metric: MetricDefinition,
                           use_group: bool = False, use_subgroup: bool = False) -> pl.LazyFrame:
        """Generate the Polars lazy expression for a metric"""
        # Build expressions
        agg_exprs, select_expr = self._build_expressions(metric)
        
        # Get grouping columns
        analysis_cols = []
        if use_subgroup:
            analysis_cols.extend(self.subgroup_cols)
        if use_group:
            analysis_cols.extend(self.group_cols)
        
        base_cols = self.BASE_COLUMNS[metric.type]
        
        # Apply aggregation based on metric type
        if metric.type in [MetricType.ACROSS_SUBJECT, MetricType.ACROSS_VISIT]:
            # Two-level aggregation
            # Prepare first level expressions
            if len(agg_exprs) > 1:
                first_level_exprs = agg_exprs
                if select_expr is None:
                    raise ValueError(f"select_expr required for multiple agg expressions in {metric.type.value}")
            else:
                first_level_exprs = [agg_exprs[0].alias("_value")]
                if select_expr is None:
                    # Default to mean for two-level aggregation
                    select_expr = pl.col("_value").mean()
            
            # First aggregation: by subject/visit
            first_agg = self._aggregate(lf, analysis_cols + base_cols, first_level_exprs)
            
            # Second aggregation: across subjects/visits
            result = self._aggregate(first_agg, analysis_cols, [select_expr.alias("value")])
        else:
            # Single-level aggregation
            group_cols = analysis_cols + base_cols
            result = self._aggregate(lf, group_cols, agg_exprs)
        
        # Add metadata
        return result.with_columns([
            pl.lit(metric.name).alias("metric"),
            pl.lit(metric.label).alias("label"),
            pl.lit(metric.type.value).alias("metric_type")
        ])