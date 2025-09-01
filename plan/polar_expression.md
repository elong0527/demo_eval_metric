# Objective

Design an architecture for implementing a general framework to convert YAML config to metric lazy evaluation using Polars.

## Mental Model

The evaluation pipeline consists of three composable stages that form a lazy computation chain, with metadata enrichment:

```python
import polars as pl 

# Stage 1: Data filtering/preparation
def pl_data_expr(df: pl.LazyFrame, filter_expr: pl.Expr | None = None, 
                 ground_truth: str = "actual", estimate: str = "predicted") -> pl.LazyFrame:
    """Prepare data with error columns and filters"""
    # Add error columns for metric computation
    error = pl.col(estimate) - pl.col(ground_truth)
    df = df.with_columns([
        error.alias("error"),
        error.abs().alias("absolute_error"),
        (error ** 2).alias("squared_error"),
    ])
    
    if filter_expr:
        return df.filter(filter_expr)
    return df

# Stage 2: First-level aggregation (optional)
def pl_agg_expr(df: pl.LazyFrame, agg_exprs: list[pl.Expr], group_cols: list[str]) -> pl.LazyFrame:
    """Aggregate data by specified groups"""
    if group_cols:
        return df.group_by(group_cols).agg(agg_exprs)
    return df.select(agg_exprs)  # No grouping = select

# Stage 3: Second-level selection/aggregation (metric computation)
def pl_select_expr(df: pl.LazyFrame, select_expr: pl.Expr, group_cols: list[str]) -> pl.LazyFrame:
    """Final metric computation - always produces 'value' column"""
    if group_cols:
        return df.group_by(group_cols).agg(select_expr.alias("value"))
    else:
        return df.select(select_expr.alias("value"))

# Stage 4: Metadata enrichment
def add_metadata(df: pl.LazyFrame, metric_name: str, estimate_name: str) -> pl.LazyFrame:
    """Add metric and estimate metadata columns"""
    return df.with_columns([
        pl.lit(metric_name).alias("metric"),
        pl.lit(estimate_name).alias("estimate")
    ])
```

### Pipeline Composition

```python
def evaluate_pipeline(
    df: pl.LazyFrame,
    metric_name: str,
    ground_truth: str = "actual",
    estimate: str,
    agg_exprs: list[pl.Expr],
    select_expr: pl.Expr,
    agg_groups: list[str] = None,
    select_groups: list[str] = None,
    filter_expr: pl.Expr = None,
) -> pl.LazyFrame:
    """
    Evaluate a single metric for one model using the pipeline pattern.
    
    Args:
        df: Input LazyFrame
        metric_name: Name of the metric being computed
        estimate: Model column name to evaluate
        agg_exprs: Expressions for first-level aggregation
        select_expr: Expression for final metric computation
        agg_groups: Grouping columns for first aggregation (e.g., ["subject_id", "group"])
        select_groups: Grouping columns for final result (e.g., ["group", "subgroup"])
        filter_expr: Optional filter to apply to data
        ground_truth: Name of ground truth column
    
    Returns:
        LazyFrame with columns: [*select_groups, "metric", "estimate", "value"]
    """
    # Build pipeline
    pipeline = pl_data_expr(df, filter_expr, ground_truth, estimate)
    
    # Apply first-level aggregation if groups specified
    if agg_groups:
        pipeline = pipeline.pipe(pl_agg_expr, agg_exprs, agg_groups)
    
    # Apply final selection/aggregation
    pipeline = pipeline.pipe(pl_select_expr, select_expr, select_groups or [])
    
    # Add metadata
    pipeline = pipeline.pipe(add_metadata, metric_name, estimate)
    
    return pipeline

# Example: Evaluate MAE for multiple models
results = []
for model in ["model1", "model2", "model3"]:
    result = evaluate_pipeline(
        df=data.lazy(),
        metric_name="mae",
        estimate=model,
        agg_exprs=[pl.col("absolute_error").mean().alias("value")],
        select_expr=pl.col("value").mean(),
        agg_groups=["subject_id", "group", "subgroup"],
        select_groups=["group", "subgroup"],
        ground_truth="actual"
    )
    results.append(result)

# Combine and collect all results
final_result = pl.concat(results).collect()
```

**Result Structure:**
```
| group | subgroup | metric | estimate | value |
|-------|----------|--------|----------|-------|
| A     | Male     | mae    | model1   | 0.5   |
| A     | Male     | mae    | model2   | 0.4   |
| B     | Female   | mae    | model1   | 0.6   |
| B     | Female   | mae    | model2   | 0.3   |
```

**Key Design Insight**: The result structure indicates that:
1. Each metric evaluation produces rows grouped by `group` and `subgroup` columns
2. Metadata columns (`metric`, `estimate`) must be added to identify the computation
3. Multiple models/estimates are evaluated separately and combined
