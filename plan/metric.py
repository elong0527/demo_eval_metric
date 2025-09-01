"""
Metric definition with Pydantic validation
"""

from enum import Enum
from pydantic import BaseModel, field_validator, model_validator
from typing import Self, Any, ClassVar
import polars as pl


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


class Metric(BaseModel):
    """Metric definition with validation"""
    name: str
    label: str
    type: MetricType
    shared_by: SharedType | None = None
    agg_expr: list[str] | None = None
    select_expr: str | None = None
    
    # Class-level expression definitions using Polars expressions directly
    BUILTIN_METRICS: ClassVar[dict[str, pl.Expr]] = {
        "mae": pl.col("absolute_error").mean().alias("value"),
        "mse": pl.col("squared_error").mean().alias("value"),
        "rmse": pl.col("squared_error").mean().sqrt().alias("value"),
        "bias": pl.col("error").mean().alias("value"),
        "mape": pl.col("absolute_percent_error").mean().alias("value"),
        "n_subject": pl.col("subject_id").n_unique().alias("value"),
        "n_visit": pl.struct(["subject_id", "visit_id"]).n_unique().alias("value"),
        "n_sample": pl.len().alias("value"),
        "total_subject": pl.col("subject_id").n_unique().alias("value"),
        "total_visit": pl.struct(["subject_id", "visit_id"]).n_unique().alias("value"),
    }
    
    BUILTIN_SELECTORS: ClassVar[dict[str, pl.Expr]] = {
        "mean": pl.col("value").mean(),
        "median": pl.col("value").median(),
        "std": pl.col("value").std(),
        "min": pl.col("value").min(),
        "max": pl.col("value").max(),
        "sum": pl.col("value").sum(),
        "sqrt": pl.col("value").sqrt(),
    }
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate metric name is not empty"""
        if not v or not v.strip():
            raise ValueError("Metric name cannot be empty")
        return v.strip()
    
    @field_validator('label')
    @classmethod
    def validate_label(cls, v: str) -> str:
        """Validate label is not empty"""
        if not v or not v.strip():
            raise ValueError("Metric label cannot be empty")
        return v.strip()
    
    @field_validator('select_expr')
    @classmethod
    def validate_select_expr(cls, v: str | None) -> str | None:
        """Validate select expression has alias for single-expression metrics"""
        if v is None:
            return None
        
        # For select expressions that will be used alone (no agg_expr),
        # they should have an alias
        # This will be checked in the model validator based on context
        return v
    
    @field_validator('agg_expr')
    @classmethod
    def validate_agg_expr(cls, v: list[str] | None) -> list[str] | None:
        """Validate aggregation expressions have aliases"""
        if v is None:
            return None
        if not v:
            raise ValueError("agg_expr cannot be an empty list")
        
        # Check each expression has an alias or is a built-in
        for expr in v:
            if not expr or not expr.strip():
                raise ValueError("agg_expr cannot contain empty expressions")
            
            expr_str = expr.strip()
            # Check if it has an alias or is a reference to a built-in metric
            has_alias = '.alias(' in expr_str or '.alias (' in expr_str
            is_builtin = expr_str in cls.BUILTIN_METRICS
            
            if not has_alias and not is_builtin:
                raise ValueError(
                    f"Expression '{expr_str}' must have .alias() or be a built-in metric name. "
                    f"Example: pl.col('column').mean().alias('value')"
                )
        return v
    
    @model_validator(mode='after')
    def validate_expressions(self) -> Self:
        """Validate expression combinations"""
        # For custom metrics, at least one expression must be provided
        is_custom = self.agg_expr is not None or self.select_expr is not None
        
        if is_custom:
            # Custom metrics must have at least one expression
            if not self.agg_expr and not self.select_expr:
                raise ValueError("Custom metrics must have at least agg_expr or select_expr")
            
            # If only select_expr is provided (no agg_expr), it must have an alias
            if self.select_expr and not self.agg_expr:
                expr_str = self.select_expr.strip()
                has_alias = '.alias(' in expr_str or '.alias (' in expr_str
                is_builtin = expr_str in self.BUILTIN_METRICS
                
                if not has_alias and not is_builtin:
                    raise ValueError(
                        f"select_expr '{expr_str}' used alone must have .alias() or be a built-in metric. "
                        f"Example: (...).alias('value')"
                    )
            
            # For two-level aggregation with multiple agg_expr, select_expr is required
            if self.type in (MetricType.ACROSS_SUBJECT, MetricType.ACROSS_VISIT):
                if self.agg_expr and len(self.agg_expr) > 1 and not self.select_expr:
                    raise ValueError(
                        f"select_expr required for multiple agg expressions in {self.type.value}"
                    )
        else:
            # Built-in metrics should follow naming convention
            if ':' in self.name:
                parts = self.name.split(':', 1)
                if len(parts) != 2 or not parts[0] or not parts[1]:
                    raise ValueError(f"Invalid built-in metric name format: {self.name}")
        
        return self
    
    @model_validator(mode='after')
    def validate_shared_by(self) -> Self:
        """Validate shared_by is only used with across_samples metrics"""
        if self.shared_by is not None:
            if self.type != MetricType.ACROSS_SAMPLES:
                raise ValueError(f"shared_by can only be used with across_samples metrics, not {self.type.value}")
        return self
    
    @classmethod
    def from_yaml(cls, config: dict[str, Any]) -> 'Metric':
        """Create Metric from YAML configuration dict
        
        This method handles the YAML-specific parsing logic and ensures
        the resulting Metric is in a complete, valid state.
        """
        # Extract basic fields
        metric_data = {
            'name': config['name'],
            'label': config.get('label', config['name']),
            'type': config.get('type', 'across_samples')
        }
        
        # Parse shared_by if present
        if 'shared_by' in config:
            metric_data['shared_by'] = config['shared_by']
        
        # Parse agg expressions (normalize to list)
        agg_config = config.get('agg', {})
        if isinstance(agg_config, dict) and 'expr' in agg_config:
            raw_expr = agg_config['expr']
            metric_data['agg_expr'] = [raw_expr] if isinstance(raw_expr, str) else raw_expr
        
        # Parse select expression
        select_config = config.get('select', {})
        if isinstance(select_config, dict) and 'expr' in select_config:
            metric_data['select_expr'] = select_config['expr']
        
        # Create the metric with validation
        metric = cls(**metric_data)
        
        # Post-process to ensure completeness for downstream processing
        metric._ensure_complete_expressions()
        
        return metric
    
    def _ensure_complete_expressions(self) -> None:
        """Ensure expressions are complete for downstream processing
        
        This method modifies the metric to ensure that after initialization,
        the metric has all necessary expressions filled in with appropriate defaults,
        eliminating the need for null checks in downstream code.
        
        Rules:
        - For two-level aggregations: ensure select_expr exists (default to mean)
        """
        # For two-level aggregations, ensure select_expr exists
        if self.type in (MetricType.ACROSS_SUBJECT, MetricType.ACROSS_VISIT):
            if self.agg_expr and not self.select_expr:
                # Default to mean for two-level aggregation
                self.select_expr = "pl.col('value').mean()"
    
    @classmethod
    def _evaluate_expression(cls, expr_str: str | None) -> pl.Expr | None:
        """Convert string expression to Polars expression"""
        if expr_str is None:
            return None
        
        namespace = {
            'pl': pl,
            'col': pl.col,
            'lit': pl.lit,
            'len': pl.len,
            'struct': pl.struct,
            # Built-in metric shortcuts (allows using 'mae' directly in expressions)
            **{name: expr for name, expr in cls.BUILTIN_METRICS.items()}
        }
        
        try:
            return eval(expr_str, namespace, {})
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression '{expr_str}': {e}")
    
    
    def get_polars_expressions(self) -> tuple[list[pl.Expr], pl.Expr | None]:
        """Get Polars expressions for this metric
        
        Returns:
            Tuple of (aggregation_expressions, selection_expression)
            All single aggregation expressions must have 'value' alias
        """
        # Handle custom expressions
        if self.agg_expr or self.select_expr:
            # Evaluate custom expressions
            agg_exprs = [self._evaluate_expression(expr) for expr in self.agg_expr] if self.agg_expr else []
            select_expr = self._evaluate_expression(self.select_expr)
            
            # For metrics with only select_expr (no agg_expr), use it as single aggregation
            if not agg_exprs and select_expr is not None:
                # Custom expressions from YAML should have 'value' alias
                # We trust the validation happens when writing YAML
                return [select_expr], None
            
            # For single aggregation without selector, it should have 'value' alias
            # Built-in metrics already have it, custom ones should too
            
            return agg_exprs, select_expr
        
        # Handle built-in metrics
        parts = (self.name + ':').split(':')[:2]
        agg_name, select_name = parts[0], parts[1] if parts[1] else None
        
        # Get built-in expressions (already Polars expressions)
        agg_expr = self.BUILTIN_METRICS.get(agg_name)
        if agg_expr is None:
            raise ValueError(f"Unknown built-in metric: {agg_name}")
        
        # Get selector expression if specified
        select_expr = self.BUILTIN_SELECTORS.get(select_name)
        
        return [agg_expr], select_expr