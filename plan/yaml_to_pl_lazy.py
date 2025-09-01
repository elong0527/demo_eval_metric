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
    agg_expr: str | list[str] | None = None
    select_expr: str | None = None


class YAMLToPolarsLazyTranslator:
    """Translates YAML metric definitions to Polars lazy expressions"""
    
    def __init__(self, yaml_config: str | dict):
        """
        Initialize translator with YAML configuration
        
        Args:
            yaml_config: Path to YAML file or dict configuration
        """
        if isinstance(yaml_config, str):
            with open(yaml_config, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = yaml_config
            
        # Extract columns configuration
        columns = self.config.get('columns', {})
        self.group_cols = columns.get('group', [])
        self.subgroup_cols = columns.get('subgroup', [])
        
        # Parse metrics
        self.metrics = self._parse_metrics()
        
        # Built-in metric expressions
        self._builtin_metrics = {
            "mae": pl.col("absolute_error").mean(),
            "mse": pl.col("squared_error").mean(),
            "rmse": pl.col("squared_error").mean(),
            "bias": pl.col("error").mean(),
            "mape": pl.col("absolute_percent_error").mean(),
            "n_subject": pl.col("subject_id").n_unique(),
            "n_visit": pl.struct(["subject_id", "visit_id"]).n_unique(),
            "n_sample": pl.len(),
            "total_subject": pl.col("subject_id").n_unique(),
            "total_visit": pl.struct(["subject_id", "visit_id"]).n_unique(),
        }
        
        # Built-in selection functions
        self._builtin_selectors = {
            "mean": lambda col: pl.col(col).mean(),
            "median": lambda col: pl.col(col).median(),
            "std": lambda col: pl.col(col).std(),
            "min": lambda col: pl.col(col).min(),
            "max": lambda col: pl.col(col).max(),
            "sum": lambda col: pl.col(col).sum(),
            "sqrt": lambda col: pl.col(col).sqrt(),
        }
    
    def _parse_metrics(self) -> list[MetricDefinition]:
        """Parse metric definitions from YAML"""
        metrics = []
        for metric_config in self.config.get('metrics', []):
            # Parse type
            metric_type = MetricType(metric_config.get('type', 'across_samples'))
            
            # Parse shared_by
            shared_by_value = metric_config.get('shared_by')
            shared_type = None
            if shared_by_value == 'model':
                shared_type = SharedType.MODEL
            elif shared_by_value == 'all':
                shared_type = SharedType.ALL
            elif shared_by_value == 'group':
                shared_type = SharedType.GROUP
            
            # Parse expressions
            agg_config = metric_config.get('agg', {})
            agg_expr = None
            if isinstance(agg_config, dict):
                agg_expr = agg_config.get('expr')
            
            select_config = metric_config.get('select', {})
            select_expr = None
            if isinstance(select_config, dict):
                select_expr = select_config.get('expr')
            
            metric = MetricDefinition(
                name=metric_config['name'],
                label=metric_config.get('label', metric_config['name']),
                type=metric_type,
                shared_by=shared_type,
                agg_expr=agg_expr,
                select_expr=select_expr
            )
            metrics.append(metric)
        return metrics
    
    def prepare_dataframe(self, df: pl.DataFrame | pl.LazyFrame, 
                         ground_truth: str, estimate: str) -> pl.LazyFrame:
        """
        Add error columns needed for metrics calculation
        
        Args:
            df: Input dataframe
            ground_truth: Name of ground truth column
            estimate: Name of estimate column
            
        Returns:
            LazyFrame with added error columns
        """
        if isinstance(df, pl.DataFrame):
            df = df.lazy()
            
        return df.with_columns([
            (pl.col(estimate) - pl.col(ground_truth)).alias("error"),
            (pl.col(estimate) - pl.col(ground_truth)).abs().alias("absolute_error"),
            ((pl.col(estimate) - pl.col(ground_truth)) ** 2).alias("squared_error"),
            
            pl.when(pl.col(ground_truth) != 0)
            .then((pl.col(estimate) - pl.col(ground_truth)) / pl.col(ground_truth) * 100)
            .otherwise(None)
            .alias("percent_error"),
            
            pl.when(pl.col(ground_truth) != 0)
            .then(((pl.col(estimate) - pl.col(ground_truth)) / pl.col(ground_truth) * 100).abs())
            .otherwise(None)
            .alias("absolute_percent_error"),
        ])
    
    def build_expression(self, metric: MetricDefinition) -> tuple[pl.Expr | list[pl.Expr], pl.Expr | None]:
        """
        Build Polars expressions for a metric
        
        Returns:
            Tuple of (aggregation_expression(s), selection_expression)
        """
        # Handle custom expressions
        if metric.agg_expr or metric.select_expr:
            agg_expr = self._parse_agg_expression(metric.agg_expr) if metric.agg_expr else None
            select_expr = self._evaluate_expression(metric.select_expr) if metric.select_expr else None
            
            # If only select_expr provided, use it as the main aggregation
            if select_expr is not None and agg_expr is None:
                return select_expr, None
                
            return agg_expr, select_expr
        
        # Handle built-in metric names
        if ':' in metric.name:
            agg_name, select_name = metric.name.split(':', 1)
        else:
            agg_name, select_name = metric.name, None
        
        # Get aggregation expression
        agg_expr = self._builtin_metrics.get(agg_name)
        if agg_expr is None:
            raise ValueError(f"Unknown metric: {agg_name}")
        
        # Get selection expression
        select_expr = None
        if select_name:
            selector = self._builtin_selectors.get(select_name)
            if selector:
                select_expr = selector("_value")
        elif agg_name == "rmse":
            # Special case: RMSE needs sqrt
            select_expr = pl.col("_value").sqrt()
            
        return agg_expr, select_expr
    
    def _parse_agg_expression(self, agg_expr: str | list[str]) -> pl.Expr | list[pl.Expr]:
        """Parse aggregation expression(s)"""
        if isinstance(agg_expr, list):
            return [self._evaluate_expression(expr) for expr in agg_expr]
        else:
            return self._evaluate_expression(agg_expr)
    
    def _evaluate_expression(self, expr_str: str) -> pl.Expr:
        """
        Evaluate a Polars expression string
        
        Args:
            expr_str: Expression string to evaluate
            
        Returns:
            Polars expression
        """
        # Create namespace with Polars functions and built-in metrics
        namespace = {
            'pl': pl,
            'col': pl.col,
            'lit': pl.lit,
            'len': pl.len,
            'struct': pl.struct,
            # Add built-in metrics as shortcuts
            **{name: expr for name, expr in self._builtin_metrics.items()}
        }
        
        try:
            return eval(expr_str, namespace, {})
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression '{expr_str}': {e}")
    
    def get_lazy_expression(self, lf: pl.LazyFrame, metric: MetricDefinition,
                           use_group: bool = False, use_subgroup: bool = False) -> pl.LazyFrame:
        """
        Generate the Polars lazy expression for a metric
        
        Args:
            lf: Input LazyFrame (should already have error columns)
            metric: Metric definition
            use_group: Whether to include group columns
            use_subgroup: Whether to include subgroup columns
            
        Returns:
            LazyFrame with the metric calculation
        """
        agg_expr, select_expr = self.build_expression(metric)
        
        # Determine grouping columns
        group_cols = self._get_group_columns(metric.type, use_group, use_subgroup)
        
        # Build the aggregation based on metric type
        if metric.type == MetricType.ACROSS_SAMPLES:
            result = self._single_aggregation(lf, group_cols, agg_expr)
            
        elif metric.type in [MetricType.WITHIN_SUBJECT, MetricType.WITHIN_VISIT]:
            base_cols = self._get_base_columns(metric.type)
            result = self._single_aggregation(lf, group_cols + base_cols, agg_expr)
            
        elif metric.type in [MetricType.ACROSS_SUBJECT, MetricType.ACROSS_VISIT]:
            base_cols = self._get_base_columns(metric.type)
            result = self._two_level_aggregation(lf, group_cols + base_cols, 
                                                group_cols, agg_expr, select_expr)
        else:
            raise ValueError(f"Unknown metric type: {metric.type}")
        
        # Add metadata
        return result.with_columns([
            pl.lit(metric.name).alias("metric"),
            pl.lit(metric.label).alias("label"),
            pl.lit(metric.type.value).alias("metric_type")
        ])
    
    def _get_base_columns(self, metric_type: MetricType) -> list[str]:
        """Get base grouping columns for a metric type"""
        if metric_type in [MetricType.WITHIN_SUBJECT, MetricType.ACROSS_SUBJECT]:
            return ["subject_id"]
        elif metric_type in [MetricType.WITHIN_VISIT, MetricType.ACROSS_VISIT]:
            return ["subject_id", "visit_id"]
        else:
            return []
    
    def _get_group_columns(self, metric_type: MetricType, 
                          use_group: bool, use_subgroup: bool) -> list[str]:
        """Get analysis grouping columns"""
        columns = []
        if use_subgroup and self.subgroup_cols:
            columns.extend(self.subgroup_cols)
        if use_group and self.group_cols:
            columns.extend(self.group_cols)
        return columns
    
    def _single_aggregation(self, lf: pl.LazyFrame, group_cols: list[str], 
                           agg_expr: pl.Expr | list[pl.Expr]) -> pl.LazyFrame:
        """Single-level aggregation"""
        if isinstance(agg_expr, list):
            # Multiple expressions
            if group_cols:
                return lf.group_by(group_cols).agg(agg_expr)
            else:
                return lf.select(agg_expr).head(1)
        else:
            # Single expression
            if group_cols:
                return lf.group_by(group_cols).agg(agg_expr.alias("value"))
            else:
                return lf.select(agg_expr.alias("value")).head(1)
    
    def _two_level_aggregation(self, lf: pl.LazyFrame, first_group: list[str],
                              second_group: list[str], agg_expr: pl.Expr | list[pl.Expr],
                              select_expr: pl.Expr | None) -> pl.LazyFrame:
        """Two-level aggregation (e.g., across_subject, across_visit)"""
        # First level aggregation
        if isinstance(agg_expr, list):
            first_agg = lf.group_by(first_group).agg(agg_expr)
            if select_expr is None:
                raise ValueError("select_expr required for multiple agg expressions")
        else:
            first_agg = lf.group_by(first_group).agg(agg_expr.alias("_value"))
            if select_expr is None:
                select_expr = pl.col("_value").mean()
        
        # Second level aggregation
        if second_group:
            return first_agg.group_by(second_group).agg(select_expr.alias("value"))
        else:
            return first_agg.select(select_expr.alias("value"))