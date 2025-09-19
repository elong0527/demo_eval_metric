"""Table formatting utilities for polars-eval-metrics."""

import polars as pl
from great_tables import GT, html
from polars import selectors as cs


def pivot_to_gt(df: pl.DataFrame, decimals: int = 1) -> GT:
    """Format a pivot table from MetricEvaluator using great_tables."""
    has_subgroups = "subgroup_name" in df.columns and "subgroup_value" in df.columns

    # Parse JSON column names
    json_columns = [
        col for col in df.columns if col.startswith('{"') and col.endswith('"}')
    ]
    parsed_columns = {}
    metrics = set()

    for col in json_columns:
        inner = col[2:-2]  # Remove {" and "}
        parts = inner.split('","')
        if len(parts) == 2:
            model, metric = parts
            parsed_columns[col] = {"model": model, "metric": metric}
            metrics.add(metric)

    # Create base GT table
    gt_table = GT(df)

    # Handle subgroup columns
    if has_subgroups:
        gt_table = gt_table.tab_stub(
            rowname_col="subgroup_value", groupname_col="subgroup_name"
        )

    # Create column spanners for each metric
    for metric in sorted(metrics):
        metric_columns = [
            col for col, info in parsed_columns.items() if info["metric"] == metric
        ]
        if metric_columns:
            gt_table = gt_table.tab_spanner(label=html(metric), columns=metric_columns)

    # Rename columns
    column_renames = {col: html(info["model"]) for col, info in parsed_columns.items()}
    non_json_columns = [
        col
        for col in df.columns
        if col not in json_columns and col not in ["subgroup_name", "subgroup_value"]
    ]
    column_renames.update({col: html(col) for col in non_json_columns})

    if column_renames:
        gt_table = gt_table.cols_label(**column_renames)

    # Format numeric columns using selectors
    return gt_table.fmt_number(columns=cs.numeric(), decimals=decimals).cols_align(
        align="center", columns=cs.numeric()
    )
