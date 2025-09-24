"""Analysis Results Data (ARD) container."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable, Mapping

import polars as pl


@dataclass
class ARD:
    """Fixed-schema container for metric evaluation output."""

    _lf: pl.LazyFrame
    _group_fields: tuple[str, ...]
    _subgroup_fields: tuple[str, ...]
    _context_fields: tuple[str, ...]
    _id_fields: tuple[str, ...]

    def __init__(self, data: pl.DataFrame | pl.LazyFrame | None = None) -> None:
        if data is None:
            self._lf = self._empty_frame()
        elif isinstance(data, pl.DataFrame):
            self._validate_schema(data)
            self._lf = data.lazy()
        elif isinstance(data, pl.LazyFrame):
            self._lf = data
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        schema = self._lf.collect_schema()
        self._id_fields = self._extract_struct_fields(schema, "id")
        self._group_fields = self._extract_struct_fields(schema, "groups")
        self._subgroup_fields = self._extract_struct_fields(schema, "subgroups")
        self._context_fields = self._extract_struct_fields(schema, "context")

    # ---------------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _empty_frame() -> pl.LazyFrame:
        """Return an empty ARD frame with the canonical schema."""
        stat_dtype = pl.Struct(
            [
                pl.Field("type", pl.Utf8),
                pl.Field("value_float", pl.Float64),
                pl.Field("value_int", pl.Int64),
                pl.Field("value_bool", pl.Boolean),
                pl.Field("value_str", pl.Utf8),
                pl.Field("value_struct", pl.Struct([])),
                pl.Field("format", pl.Utf8),
            ]
        )
        frame = pl.DataFrame(
            {
                "id": pl.Series([], dtype=pl.Null),
                "groups": pl.Series([], dtype=pl.Struct([])),
                "subgroups": pl.Series([], dtype=pl.Struct([])),
                "estimate": pl.Series([], dtype=pl.Utf8),
                "metric": pl.Series([], dtype=pl.Utf8),
                "label": pl.Series([], dtype=pl.Utf8),
                "stat": pl.Series([], dtype=stat_dtype),
                "stat_fmt": pl.Series([], dtype=pl.Utf8),
                "warning": pl.Series([], dtype=pl.List(pl.Utf8)),
                "error": pl.Series([], dtype=pl.List(pl.Utf8)),
                "context": pl.Series([], dtype=pl.Struct([])),
            }
        )
        return frame.lazy()

    @staticmethod
    def _extract_struct_fields(
        schema: Mapping[str, pl.DataType], column: str
    ) -> tuple[str, ...]:
        """Return field names for struct columns, or an empty tuple when not present."""
        dtype = schema.get(column)
        if isinstance(dtype, pl.Struct):
            return tuple(field.name for field in dtype.fields)
        return tuple()

    @staticmethod
    def _validate_schema(df: pl.DataFrame) -> None:
        """Guard against constructing ARD from frames missing required columns."""
        required = {"groups", "subgroups", "estimate", "metric", "stat", "context"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required ARD columns: {missing}")

    # ------------------------------------------------------------------
    # Basic API
    # ------------------------------------------------------------------

    @property
    def lazy(self) -> pl.LazyFrame:
        return self._lf

    def collect(self) -> pl.DataFrame:
        """Collect the lazy evaluation while keeping the canonical columns when available."""
        # Keep core columns for backward compatibility when eagerly collecting
        available = self._lf.collect_schema().names()
        desired = [
            col
            for col in [
                "id",
                "groups",
                "subgroups",
                "subgroup_name",
                "subgroup_value",
                "estimate",
                "metric",
                "label",
                "stat",
                "stat_fmt",
                "warning",
                "error",
                "context",
            ]
            if col in available
        ]
        return self._lf.select(desired).collect()

    def __len__(self) -> int:
        return self.collect().height

    @property
    def shape(self) -> tuple[int, int]:
        collected = self.collect()
        return collected.shape

    @property
    def columns(self) -> list[str]:
        return list(self.schema.keys())

    @property
    def schema(self) -> dict[str, pl.DataType]:
        """Expose the ARD schema for compatibility with tests/utilities."""
        collected = self._lf.collect_schema()
        return dict(zip(collected.names(), collected.dtypes()))

    def __getitem__(self, key: str) -> pl.Series:
        """Allow DataFrame-like column access for compatibility with tests."""
        collected = self.collect()
        if key in collected.columns:
            return collected[key]
        schema_names = self._lf.collect_schema().names()
        if key in schema_names:
            return self._lf.select(pl.col(key)).collect()[key]
        raise KeyError(key)

    def iter_rows(self, *args: Any, **kwargs: Any) -> Iterable[tuple[Any, ...]]:
        """Iterate over rows of the eagerly collected DataFrame."""
        return self.collect().iter_rows(*args, **kwargs)

    def sort(self, *args: Any, **kwargs: Any) -> ARD:
        """Return a sorted ARD (collecting lazily)."""
        return ARD(self._lf.sort(*args, **kwargs))

    # ------------------------------------------------------------------
    # Formatting utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _stat_value(stat: Mapping[str, Any] | None) -> Any:
        """Extract the native value stored in a stat struct regardless of channel used."""
        if stat is None:
            return None

        type_label = (stat.get("type") or "").lower()
        if type_label == "float":
            return stat.get("value_float")
        if type_label == "int":
            return stat.get("value_int")
        if type_label == "bool":
            return stat.get("value_bool")
        if type_label == "string":
            return stat.get("value_str")
        if type_label == "struct":
            return stat.get("value_struct")

        for field in [
            "value_float",
            "value_int",
            "value_bool",
            "value_str",
            "value_struct",
        ]:
            candidate = stat.get(field)
            if candidate is not None:
                if field == "value_struct":
                    return candidate
                return candidate

        return None

    @staticmethod
    def _format_stat(stat: Mapping[str, Any] | None) -> str:
        """Render a stat struct into a string while respecting explicit formatting hints."""
        if stat is None:
            return "NULL"

        value = ARD._stat_value(stat)
        type_label = (stat.get("type") or "").lower()
        fmt = stat.get("format")
        if fmt and value is not None:
            try:
                rendered = fmt.format(value)
            except Exception:
                rendered = str(value)
        elif isinstance(value, float):
            rendered = f"{value:.1f}"
        elif isinstance(value, int):
            rendered = f"{value:,}"
        elif isinstance(value, (dict, list, tuple)):
            rendered = json.dumps(value)
        else:
            rendered = None if value is None else str(value)

        return rendered

    def __repr__(self) -> str:
        summary = self.summary()
        return f"ARD(summary={summary})"

    # ------------------------------------------------------------------
    # Null / empty handling
    # ------------------------------------------------------------------

    def with_empty_as_null(self) -> ARD:
        """Collapse empty structs or blank strings to null for easier downstream filtering."""

        def _collapse(column: str, fields: tuple[str, ...]) -> pl.Expr:
            if not fields:
                return pl.col(column)
            empty = pl.all_horizontal(
                [pl.col(column).struct.field(field).is_null() for field in fields]
            )
            return (
                pl.when(pl.col(column).is_null() | empty)
                .then(None)
                .otherwise(pl.col(column))
                .alias(column)
            )

        lf = self._lf.with_columns(
            [
                _collapse("id", self._id_fields),
                _collapse("groups", self._group_fields),
                _collapse("subgroups", self._subgroup_fields),
                _collapse("context", self._context_fields),
                pl.when(pl.col("estimate") == "")
                .then(None)
                .otherwise(pl.col("estimate"))
                .alias("estimate"),
            ]
        )
        return ARD(lf)

    def with_null_as_empty(self) -> ARD:
        """Fill null structs or estimates with empty shells to simplify presentation."""

        def _expand(column: str, fields: tuple[str, ...]) -> pl.Expr:
            if not fields:
                return pl.col(column)
            placeholders = [pl.lit(None).alias(name) for name in fields]
            return (
                pl.when(pl.col(column).is_null())
                .then(pl.struct(placeholders))
                .otherwise(pl.col(column))
                .alias(column)
            )

        lf = self._lf.with_columns(
            [
                _expand("id", self._id_fields),
                _expand("groups", self._group_fields),
                _expand("subgroups", self._subgroup_fields),
                _expand("context", self._context_fields),
                pl.col("estimate").fill_null(""),
            ]
        )
        return ARD(lf)

    # ------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------

    def unnest(self, columns: list[str] | None = None) -> pl.DataFrame:
        """Expand selected struct columns into top-level fields for inspection or exports."""
        columns = columns or ["groups", "subgroups"]
        lf = self._lf
        schema = lf.collect_schema()
        for column in columns:
            if column not in {"id", "groups", "subgroups", "context", "stat"}:
                continue
            if column not in schema.names():
                continue
            dtype = schema.get(column)
            if not isinstance(dtype, pl.Struct):
                continue
            struct_fields = {field.name for field in dtype.fields}
            existing_fields = set(schema.names())
            if struct_fields & existing_fields:
                continue
            has_values = lf.select(pl.col(column).is_not_null().any()).collect().item()
            if has_values:
                lf = lf.unnest(column)
                schema = lf.collect_schema()
        return lf.collect()

    def to_wide(
        self,
        index: list[str] | None = None,
        columns: list[str] | None = None,
        values: str = "stat",
        aggregate: str = "first",
    ) -> pl.DataFrame:
        """Pivot the ARD into a wide grid, formatting stats unless a value column is provided."""
        df = self.unnest(["groups", "subgroups", "context"])

        if columns is None:
            has_estimates = (
                df.filter(pl.col("estimate").is_not_null())["estimate"].n_unique() > 1
            )
            columns = ["estimate", "metric"] if has_estimates else ["metric"]

        if index is None:
            index = [col for col in df.columns if col not in columns + [values, "stat"]]

        if values == "stat":
            if "stat_fmt" in df.columns:
                formatted_expr = (
                    pl.when(pl.col("stat_fmt").is_null())
                    .then(
                        pl.col("stat").map_elements(
                            ARD._format_stat, return_dtype=pl.Utf8
                        )
                    )
                    .otherwise(pl.col("stat_fmt"))
                    .alias("_value")
                )
            else:
                formatted_expr = (
                    pl.col("stat")
                    .map_elements(ARD._format_stat, return_dtype=pl.Utf8)
                    .alias("_value")
                )

            df = df.with_columns(formatted_expr)
            values = "_value"

        if not index or all(df[col].null_count() == len(df) for col in index):
            df = df.with_row_index("_idx")
            index = ["_idx"]

        pivoted = df.pivot(
            index=index, on=columns, values=values, aggregate_function=aggregate
        )

        if "_idx" in pivoted.columns:
            pivoted = pivoted.drop("_idx")
        if "_value" in pivoted.columns:
            pivoted = pivoted.drop("_value")
        return pivoted

    def to_long(self) -> pl.DataFrame:
        """Convert ARD to long format with flattened columns for direct Polars operations."""
        # Start with a copy of the lazy frame
        lf = self._lf
        schema = lf.collect_schema()

        # Check for potential conflicts with context unnesting
        context_conflicts = False
        if "context" in schema.names():
            context_dtype = schema.get("context")
            if isinstance(context_dtype, pl.Struct):
                context_fields = {field.name for field in context_dtype.fields}
                existing_fields = set(schema.names())
                context_conflicts = bool(context_fields & existing_fields)

        # Unnest struct columns, checking for conflicts
        current_schema = lf.collect_schema()
        for column in ["groups", "subgroups"]:
            if column in current_schema.names():
                has_values = (
                    lf.select(pl.col(column).is_not_null().any()).collect().item()
                )
                if has_values:
                    # Check for column conflicts before unnesting
                    struct_dtype = current_schema.get(column)
                    if isinstance(struct_dtype, pl.Struct):
                        struct_fields = {field.name for field in struct_dtype.fields}
                        existing_fields = set(current_schema.names())
                        conflicts = struct_fields & existing_fields

                        if not conflicts:
                            # Safe to unnest
                            lf = lf.unnest(column)
                        # If there are conflicts, skip unnesting (top-level columns already exist)

        # Only unnest context if no conflicts
        if "context" in schema.names() and not context_conflicts:
            has_values = (
                lf.select(pl.col("context").is_not_null().any()).collect().item()
            )
            if has_values:
                lf = lf.unnest("context")

        # Handle stat column specially to extract value
        schema_names = lf.collect_schema().names()
        if "stat" in schema_names:
            if "stat_fmt" in schema_names:
                value_expr = (
                    pl.when(pl.col("stat_fmt").is_null())
                    .then(
                        pl.col("stat").map_elements(
                            ARD._format_stat, return_dtype=pl.Utf8
                        )
                    )
                    .otherwise(pl.col("stat_fmt"))
                    .alias("value")
                )
            else:
                value_expr = (
                    pl.col("stat")
                    .map_elements(ARD._format_stat, return_dtype=pl.Utf8)
                    .alias("value")
                )

            lf = lf.with_columns(value_expr)

        return lf.collect()

    def pivot(
        self,
        on: str | list[str],
        index: str | list[str] | None = None,
        values: str = "stat",
        aggregate_function: str = "first",
    ) -> pl.DataFrame:
        """Pivot ARD data using flattened column access."""
        # First flatten the ARD to get columns directly accessible
        df = self.to_long()

        # Add value column if using stat
        if values == "stat":
            df = df.with_columns(
                pl.col("stat")
                .map_elements(ARD._stat_value, return_dtype=pl.Float64)
                .alias("value")
            )
            values = "value"

        # Set default index if not provided
        if index is None:
            # Use all remaining columns except the pivot columns and values
            on_list = [on] if isinstance(on, str) else on
            index = [col for col in df.columns if col not in on_list + [values]]

        # Ensure index is a list
        if isinstance(index, str):
            index = [index]

        return df.pivot(
            on=on, index=index, values=values, aggregate_function=aggregate_function
        )

    def get_stats(self, include_metadata: bool = False) -> pl.DataFrame:
        """Return a DataFrame of metric values with optional stat metadata columns."""
        select_cols = ["metric", "stat"]
        schema_names = self._lf.collect_schema().names()
        if "stat_fmt" in schema_names:
            select_cols.append("stat_fmt")
        df = self._lf.select(select_cols).collect()

        values = [ARD._stat_value(stat) for stat in df["stat"]]

        if include_metadata:
            types = [stat.get("type") if stat else None for stat in df["stat"]]
            formats = [stat.get("format") if stat else None for stat in df["stat"]]
            if "stat_fmt" in df.columns:
                formatted = df["stat_fmt"].to_list()
            else:
                formatted = [None] * len(df)
            return pl.DataFrame(
                {
                    "metric": df["metric"],
                    "value": values,
                    "type": types,
                    "format": formats,
                    "formatted": formatted,
                },
                strict=False,
            )

        return pl.DataFrame({"metric": df["metric"], "value": values}, strict=False)

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Summarise key counts and distinct values present in the collected ARD."""
        df = self.collect()
        return {
            "n_rows": len(df),
            "n_metrics": df["metric"].n_unique(),
            "n_estimates": df["estimate"].n_unique(),
            "n_groups": df.filter(pl.col("groups").is_not_null())["groups"].n_unique(),
            "n_subgroups": df.filter(pl.col("subgroups").is_not_null())[
                "subgroups"
            ].n_unique(),
            "metrics": df["metric"].unique().to_list(),
            "estimates": df["estimate"].unique().to_list(),
        }

    def describe(self) -> None:
        """Print a simple console summary and preview of the ARD contents."""
        summary = self.summary()
        print("=" * 50)
        print(f"ARD Summary: {summary['n_rows']} results")
        print("=" * 50)
        print("\nMetrics:")
        for metric in summary["metrics"]:
            print(f"  - {metric}")
        if summary["n_estimates"]:
            print("\nEstimates:")
            for estimate in summary["estimates"]:
                if estimate:
                    print(f"  - {estimate}")
        if summary["n_groups"]:
            print(f"\nGroup combinations: {summary['n_groups']}")
        if summary["n_subgroups"]:
            print(f"Subgroup combinations: {summary['n_subgroups']}")
        print("\nPreview:")
        print(self._lf.limit(5).collect())
