"""Shared data preparation utilities for metric evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import polars as pl


@dataclass(slots=True)
class DatasetContext:
    """Describes input dataset schema for metric evaluation."""

    ground_truth: str
    estimates: dict[str, str]
    group_by: dict[str, str]
    subgroup_by: dict[str, str]


class DataPreparation:
    """Prepare lazy frames for downstream metric execution."""

    def __init__(self, context: DatasetContext) -> None:
        self.context = context

    def build_long_frame(
        self,
        df: pl.LazyFrame,
        estimate_columns: Iterable[str],
    ) -> pl.LazyFrame:
        """
        Convert wide dataset into long format with unified estimate column.

        Args:
            df: Source lazy frame
            estimate_columns: Estimate column names to melt

        Returns:
            Long-format lazy frame with standardized ground_truth/estimate columns
        """

        estimate_list = list(estimate_columns)

        # Add deterministic sample index before melting to keep unique sample id
        df_with_index = df.with_row_index("sample_index")

        # Keep all non-estimate columns in the id_vars list
        id_vars = [
            col for col in df_with_index.collect_schema().names() if col not in estimate_list
        ]

        df_long = df_with_index.unpivot(
            index=id_vars,
            on=estimate_list,
            variable_name="estimate_name",
            value_name="estimate_value",
        )

        if self.context.estimates:
            estimate_mapping = pl.col("estimate_name")
            for name, label in self.context.estimates.items():
                estimate_mapping = estimate_mapping.replace(name, label)
            df_long = df_long.with_columns(estimate_mapping.alias("estimate"))
        else:
            df_long = df_long.rename({"estimate_name": "estimate"})

        if self.context.ground_truth in df_long.collect_schema().names():
            df_long = df_long.rename({self.context.ground_truth: "ground_truth"})

        return df_long
