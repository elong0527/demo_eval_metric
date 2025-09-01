"""
Metric Compiler for converting metric expressions to Polars expressions

This module handles all Polars-specific logic for metric expressions,
decoupled from the MetricData model.
"""

import polars as pl


class MetricCompiler:
    """Compiles metric expressions to Polars expressions"""
    
    # Built-in metric expressions
    BUILTIN_METRICS = {
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
    
    # Built-in selector expressions
    BUILTIN_SELECTORS = {
        "mean": pl.col("value").mean(),
        "median": pl.col("value").median(),
        "std": pl.col("value").std(),
        "min": pl.col("value").min(),
        "max": pl.col("value").max(),
        "sum": pl.col("value").sum(),
        "sqrt": pl.col("value").sqrt(),
    }
    
    def compile_expressions(self, 
                          name: str,
                          agg_expr: list[str] | None,
                          select_expr: str | None) -> tuple[list[pl.Expr], pl.Expr | None]:
        """
        Compile metric expressions to Polars expressions
        
        Args:
            name: Metric name (may contain built-in references)
            agg_expr: List of aggregation expression strings
            select_expr: Selection expression string
            
        Returns:
            Tuple of (aggregation_expressions, selection_expression)
        """
        # Handle custom expressions
        if agg_expr or select_expr:
            return self._compile_custom_expressions(agg_expr, select_expr)
        
        # Handle built-in metrics
        return self._compile_builtin_expressions(name)
    
    def _compile_custom_expressions(self, 
                                   agg_expr: list[str] | None,
                                   select_expr: str | None) -> tuple[list[pl.Expr], pl.Expr | None]:
        """Compile custom metric expressions"""
        agg_exprs = []
        if agg_expr:
            agg_exprs = [self._evaluate_expression(expr) for expr in agg_expr]
        
        select_pl_expr = None
        if select_expr:
            select_pl_expr = self._evaluate_expression(select_expr)
        
        # If only select_expr provided (no agg_expr), use it as single aggregation
        if not agg_exprs and select_pl_expr is not None:
            return [select_pl_expr], None
        
        return agg_exprs, select_pl_expr
    
    def _compile_builtin_expressions(self, name: str) -> tuple[list[pl.Expr], pl.Expr | None]:
        """Compile built-in metric expressions"""
        parts = (name + ':').split(':')[:2]
        agg_name, select_name = parts[0], parts[1] if parts[1] else None
        
        # Get built-in aggregation expression
        agg_expr = self.BUILTIN_METRICS.get(agg_name)
        if agg_expr is None:
            raise ValueError(f"Unknown built-in metric: {agg_name}")
        
        # Get selector expression if specified
        select_expr = None
        if select_name:
            select_expr = self.BUILTIN_SELECTORS.get(select_name)
            if select_expr is None:
                raise ValueError(f"Unknown built-in selector: {select_name}")
        
        return [agg_expr], select_expr
    
    def _evaluate_expression(self, expr_str: str) -> pl.Expr:
        """Convert string expression to Polars expression"""
        namespace = {
            'pl': pl,
            'col': pl.col,
            'lit': pl.lit,
            'len': pl.len,
            'struct': pl.struct,
            # Allow referencing built-in metrics directly
            **{name: expr for name, expr in self.BUILTIN_METRICS.items()}
        }
        
        try:
            return eval(expr_str, namespace, {})
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression '{expr_str}': {e}")