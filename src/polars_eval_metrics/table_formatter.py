"""
Table formatting utilities for polars-eval-metrics.

This module provides functions to format pivot table results using great_tables
for publication-ready output with proper column spanners and formatting.
"""

import polars as pl
from great_tables import GT, loc, style


def pivot_to_gt(
    df: pl.DataFrame,
    decimals: int = 1,
) -> GT:
    """
    Format a pivot table from MetricEvaluator using great_tables.

    Creates professional table formatting with:
    - Metric labels as column spanners
    - Model labels as column names under spanners
    - Proper styling and formatting

    Args:
        df: Pivot table DataFrame from pivot_by_group() or pivot_by_model()
        decimals: Number of decimal places for numeric columns

    Returns:
        GT table object ready for display or export
    """
    # Check if subgroup columns are present
    has_subgroups = "subgroup_name" in df.columns and "subgroup_value" in df.columns

    # Parse JSON column names to identify metrics and models
    json_columns = [col for col in df.columns if col.startswith('{"') and col.endswith('"}')]
    parsed_columns = {}
    metrics = set()
    models = set()

    for col in json_columns:
        try:
            # Parse column name like '{"Model 1","MAE"}'
            # Remove outer braces and quotes, then split by '","'
            inner = col[2:-2]  # Remove {" and "}
            parts = inner.split('","')
            if len(parts) == 2:
                model, metric = parts
                parsed_columns[col] = {"model": model, "metric": metric}
                metrics.add(metric)
                models.add(model)
        except (ValueError, IndexError):
            # Keep non-JSON columns as-is
            continue

    # Create base GT table
    gt_table = GT(df)

    # Handle subgroup columns if present
    if has_subgroups:
        # Use tab_stub to create row groups based on subgroup_name
        # and use subgroup_value as row labels
        gt_table = (gt_table
            .tab_stub(rowname_col="subgroup_value", groupname_col="subgroup_name")
        )


    # Create column spanners for each metric
    metrics_sorted = sorted(metrics)
    for metric in metrics_sorted:
        # Find all columns for this metric
        metric_columns = [
            col for col, info in parsed_columns.items()
            if info["metric"] == metric
        ]

        if metric_columns:
            gt_table = gt_table.tab_spanner(
                label=metric,
                columns=metric_columns
            )

    # Rename JSON columns to show only model names
    column_renames = {}
    for col, info in parsed_columns.items():
        column_renames[col] = info["model"]

    if column_renames:
        gt_table = gt_table.cols_label(**column_renames)

    # Format numeric columns
    numeric_columns = [col for col in json_columns if col in df.columns]
    if numeric_columns:
        gt_table = gt_table.fmt_number(
            columns=numeric_columns,
            decimals=decimals
        )

    # Style the table
    gt_table = (
        gt_table
        .tab_style(
            style=style.text(weight="bold"),
            locations=loc.column_labels()
        )
        .tab_style(
            style=style.borders(sides="top", weight="2px"),
            locations=loc.body(rows=0)
        )
    )

    # Style spanners only if they exist
    if metrics_sorted:
        gt_table = gt_table.tab_style(
            style=style.text(weight="bold"),
            locations=loc.spanner_labels(ids=metrics_sorted)
        )

    # Style row groups if subgroups are present
    if has_subgroups:
        gt_table = gt_table.tab_style(
            style=style.text(weight="bold"),
            locations=loc.row_groups()
        )

    return gt_table