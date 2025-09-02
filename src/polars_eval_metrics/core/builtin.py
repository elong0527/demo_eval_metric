"""
Built-in metric and selector expressions for polars-eval-metrics.

This module contains pre-defined Polars expressions for common metrics
and aggregation selectors.
"""

import polars as pl


# Built-in metric expressions (as Polars expressions)
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

# Built-in selector expressions (as Polars expressions)
BUILTIN_SELECTORS = {
    "mean": pl.col("value").mean(),
    "median": pl.col("value").median(),
    "std": pl.col("value").std(),
    "min": pl.col("value").min(),
    "max": pl.col("value").max(),
    "sum": pl.col("value").sum(),
    "sqrt": pl.col("value").sqrt(),
}
