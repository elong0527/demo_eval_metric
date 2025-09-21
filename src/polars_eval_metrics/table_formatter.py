"""Table formatting utilities for polars-eval-metrics."""

import polars as pl
from great_tables import GT, html
from polars import selectors as cs

from .ard import ARD


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


def ard_to_wide(
    ard: ARD,
    index: list[str] | None = None,
    columns: list[str] | None = None,
    values: str = "stat",
    aggregate_fn: str = "first",
) -> pl.DataFrame:
    """Thin wrapper around ``ARD.to_wide`` for backwards compatibility."""

    return ard.to_wide(
        index=index,
        columns=columns,
        values=values,
        aggregate=aggregate_fn,
    )


def ard_to_gt(ard: ARD, decimals: int = 1) -> GT:
    """
    Convert ARD to Great Tables format for HTML display.

    Args:
        ard: ARD data structure
        decimals: Number of decimal places for formatting numbers

    Returns:
        GT: Great Tables object for display
    """
    # Convert to wide format first
    df = ard_to_wide(ard)

    # Identify special columns
    has_subgroups = "subgroups" in df.columns and not df["subgroups"].is_null().all()

    # Parse JSON column names for metric/estimate combinations
    json_columns = [
        col for col in df.columns if col.startswith('{"') and col.endswith('"}')
    ]
    parsed_columns = {}
    metrics = set()

    for col in json_columns:
        inner = col[2:-2]  # Remove {" and "}
        parts = inner.split('","')
        if len(parts) == 2:
            estimate, metric = parts
            parsed_columns[col] = {"estimate": estimate, "metric": metric}
            metrics.add(metric)

    # Create base GT table
    gt_table = GT(df)

    # Handle subgroup columns if present
    if has_subgroups:
        # Get the unnested subgroups for display
        unnested = ard.unnest(["subgroups"])
        subgroup_cols = [
            col for col in unnested.columns if col.startswith("subgroups.")
        ]

        if len(subgroup_cols) == 1:
            # Single subgroup dimension
            subgroup_col = subgroup_cols[0].replace("subgroups.", "")
            if subgroup_col in df.columns:
                gt_table = gt_table.tab_stub(rowname_col=subgroup_col)
        elif len(subgroup_cols) > 1:
            # Multiple subgroup dimensions - use first as groupname, second as rowname
            group_col = subgroup_cols[0].replace("subgroups.", "")
            row_col = subgroup_cols[1].replace("subgroups.", "")
            if group_col in df.columns and row_col in df.columns:
                gt_table = gt_table.tab_stub(
                    rowname_col=row_col, groupname_col=group_col
                )

    # Create column spanners for each metric
    for metric in sorted(metrics):
        metric_columns = [
            col for col, info in parsed_columns.items() if info["metric"] == metric
        ]
        if metric_columns:
            gt_table = gt_table.tab_spanner(label=html(metric), columns=metric_columns)

    # Rename columns
    column_renames = {
        col: html(info["estimate"]) for col, info in parsed_columns.items()
    }

    # Handle non-JSON columns
    non_json_columns = [
        col
        for col in df.columns
        if col not in json_columns and col not in ["subgroups"]
    ]

    # For single metrics without estimates, use the metric name directly
    if not json_columns:
        # Get unique metrics from ARD
        metrics_in_ard = (
            ard.lazy.select("metric").unique().collect()["metric"].to_list()
        )
        for metric in metrics_in_ard:
            if metric in df.columns:
                column_renames[metric] = html(metric)

    column_renames.update(
        {col: html(col.replace(".", " ").title()) for col in non_json_columns}
    )

    if column_renames:
        gt_table = gt_table.cols_label(**column_renames)

    # Format numeric columns and center-align them
    return gt_table.fmt_number(columns=cs.numeric(), decimals=decimals).cols_align(
        align="center", columns=cs.numeric()
    )
