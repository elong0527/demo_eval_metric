"""Utilities for ordering pivot columns produced by Polars pivots."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable

import polars as pl


def partition_value_columns(
    columns: Sequence[str],
    index_cols: Iterable[str],
    global_labels: Sequence[str],
    group_labels: Sequence[str],
) -> tuple[list[str], list[str], list[str]]:
    """Split pivot columns into global / group / default buckets."""

    index_set = set(index_cols)
    global_cols: list[str] = []
    group_cols: list[str] = []
    default_cols: list[str] = []

    for col in columns:
        if col in index_set:
            continue
        if any(label in col for label in global_labels):
            global_cols.append(col)
        elif any(label in col for label in group_labels):
            group_cols.append(col)
        else:
            default_cols.append(col)

    return global_cols, group_cols, default_cols


def order_value_columns(
    metric_labels: Sequence[str],
    axis_labels: Sequence[str],
    global_cols: Sequence[str],
    group_cols: Sequence[str],
    default_cols: Sequence[str],
    order_mode: str,
    candidate_provider: Callable[[str, str], list[str]],
) -> list[str]:
    """Return ordered pivot value columns based on metric and axis ordering."""

    ordered: list[str] = []
    all_value_cols = list(global_cols) + list(group_cols) + list(default_cols)

    def append_if_present(candidates: Sequence[str]) -> None:
        for name in candidates:
            if name in all_value_cols and name not in ordered:
                ordered.append(name)
                break

    if order_mode == "metrics":
        for metric_label in metric_labels:
            if metric_label in global_cols and metric_label not in ordered:
                ordered.append(metric_label)
            if metric_label in group_cols and metric_label not in ordered:
                ordered.append(metric_label)

            for axis_label in axis_labels:
                append_if_present(candidate_provider(metric_label, axis_label))

    else:  # axis-first ordering
        for axis_label in axis_labels:
            for metric_label in metric_labels:
                if metric_label in global_cols or metric_label in group_cols:
                    continue
                append_if_present(candidate_provider(metric_label, axis_label))

        for metric_label in metric_labels:
            if metric_label in global_cols and metric_label not in ordered:
                ordered.append(metric_label)
            if metric_label in group_cols and metric_label not in ordered:
                ordered.append(metric_label)

    for col in all_value_cols:
        if col not in ordered:
            ordered.append(col)

    return ordered


def finalize_column_order(
    result: pl.DataFrame, index_cols: Sequence[str], ordered_value_cols: Sequence[str]
) -> pl.DataFrame:
    """Return DataFrame with columns reordered if lengths match."""

    ordered_cols = list(index_cols) + list(ordered_value_cols)
    if len(ordered_cols) != len(result.columns):
        return result
    return result.select(ordered_cols)


def candidate_names_for_estimate(metric_label: str, estimate_label: str) -> list[str]:
    """Column name possibilities for metric/estimate combinations."""

    return [
        f'{{"{estimate_label}","{metric_label}"}}',
        f"{estimate_label}_{metric_label}",
        f"{metric_label}_{estimate_label}",
    ]


def extract_group_combinations(default_cols: Sequence[str]) -> list[str]:
    """Derive unique group combination identifiers from pivot columns."""

    combinations: set[str] = set()
    for col in default_cols:
        if col.startswith("{") and col.endswith("}"):
            inner = col[1:-1]
            parts = [part.strip('"') for part in inner.split('","')]
            if len(parts) > 1:
                group_combo = '","'.join(parts[:-1])
                combinations.add(group_combo)
    return sorted(combinations)


def candidate_names_for_group(metric_label: str, group_combo: str) -> list[str]:
    """Column name possibilities for metric/group combinations."""

    return [f'{{"{group_combo}","{metric_label}"}}']
