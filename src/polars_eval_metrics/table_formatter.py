"""Table formatting utilities for polars-eval-metrics."""

from typing import Any, Callable

import polars as pl
from great_tables import GT, html
from polars import selectors as cs

from .ard import ARD
from .utils import parse_json_columns

# pyre-strict

ParsedColumns = dict[str, tuple[str, ...]]


def pivot_to_gt(df: pl.DataFrame, decimals: int = 1) -> GT:
    """Format a pivot table from MetricEvaluator using great_tables."""

    stub_builder = _pivot_stub_builder(df)
    return _gt_from_wide(
        df,
        decimals=decimals,
        column_label_index=0,
        spanner_index=1,
        non_json_formatter=_pivot_non_json_label,
        stub_builder=stub_builder,
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

    parsed_columns = parse_json_columns(df.columns)
    stub_builder = _ard_stub_builder(ard, df)

    extra_labels: dict[str, Any] = {}
    if not parsed_columns:
        metrics_in_ard = (
            ard.lazy.select("metric").unique().collect()["metric"].to_list()
        )
        for metric in metrics_in_ard:
            if metric in df.columns:
                extra_labels[metric] = html(metric)

    return _gt_from_wide(
        df,
        decimals=decimals,
        column_label_index=0,
        spanner_index=1,
        non_json_formatter=_ard_non_json_label,
        stub_builder=stub_builder,
        parsed_columns=parsed_columns,
        extra_labels=extra_labels if extra_labels else None,
    )


def _gt_from_wide(
    df: pl.DataFrame,
    *,
    decimals: int,
    column_label_index: int,
    spanner_index: int,
    non_json_formatter: Callable[[str], Any | None],
    stub_builder: Callable[[GT], GT] | None = None,
    parsed_columns: ParsedColumns | None = None,
    extra_labels: dict[str, Any] | None = None,
) -> GT:
    parsed = parsed_columns or parse_json_columns(df.columns)
    gt_table = GT(df)
    if stub_builder is not None:
        gt_table = stub_builder(gt_table)
    return _apply_gt_formatting(
        gt_table,
        df,
        parsed_columns=parsed,
        column_label_index=column_label_index,
        spanner_index=spanner_index,
        non_json_formatter=non_json_formatter,
        decimals=decimals,
        extra_labels=extra_labels,
    )


def _apply_gt_formatting(
    gt_table: GT,
    df: pl.DataFrame,
    *,
    parsed_columns: ParsedColumns,
    column_label_index: int,
    spanner_index: int,
    non_json_formatter: Callable[[str], Any | None],
    decimals: int,
    extra_labels: dict[str, Any] | None,
) -> GT:
    metrics = sorted(
        {
            tokens[spanner_index]
            for tokens in parsed_columns.values()
            if len(tokens) > spanner_index
        }
    )
    for metric in metrics:
        metric_columns = [
            col
            for col, tokens in parsed_columns.items()
            if len(tokens) > spanner_index and tokens[spanner_index] == metric
        ]
        if metric_columns:
            gt_table = gt_table.tab_spanner(label=html(metric), columns=metric_columns)

    column_renames: dict[str, Any] = {}
    for col, tokens in parsed_columns.items():
        if len(tokens) > column_label_index:
            column_renames[col] = html(tokens[column_label_index])

    for col in df.columns:
        if col in parsed_columns:
            continue
        label = non_json_formatter(col)
        if label is not None:
            column_renames[col] = label

    if extra_labels:
        column_renames.update(extra_labels)

    if column_renames:
        gt_table = gt_table.cols_label(**column_renames)

    return gt_table.fmt_number(columns=cs.numeric(), decimals=decimals).cols_align(
        align="center", columns=cs.numeric()
    )


def _pivot_stub_builder(df: pl.DataFrame) -> Callable[[GT], GT] | None:
    if {"subgroup_name", "subgroup_value"}.issubset(df.columns):
        return lambda gt: gt.tab_stub(
            rowname_col="subgroup_value", groupname_col="subgroup_name"
        )
    return None


def _pivot_non_json_label(column: str) -> Any | None:
    if column in {"subgroup_name", "subgroup_value"}:
        return None
    return html(column)


def _ard_stub_builder(ard: ARD, df: pl.DataFrame) -> Callable[[GT], GT] | None:
    if "subgroups" not in df.columns or df["subgroups"].is_null().all():
        return None

    unnested = ard.unnest(["subgroups"])
    subgroup_cols = [col for col in unnested.columns if col.startswith("subgroups.")]

    if len(subgroup_cols) == 1:
        subgroup_col = subgroup_cols[0].replace("subgroups.", "")

        if subgroup_col in df.columns:
            return lambda gt: gt.tab_stub(rowname_col=subgroup_col)

    elif len(subgroup_cols) > 1:
        group_col = subgroup_cols[0].replace("subgroups.", "")
        row_col = subgroup_cols[1].replace("subgroups.", "")

        if group_col in df.columns and row_col in df.columns:
            return lambda gt: gt.tab_stub(rowname_col=row_col, groupname_col=group_col)

    return None


def _ard_non_json_label(column: str) -> Any | None:
    if column == "subgroups":
        return None
    return html(column.replace(".", " ").title())
