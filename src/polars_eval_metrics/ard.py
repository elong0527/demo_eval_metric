"""Analysis Results Data (ARD) container."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import polars as pl

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr


@dataclass
class ARD:
    """Fixed-schema container for metric evaluation output."""

    _lf: pl.LazyFrame
    _group_fields: tuple[str, ...]
    _subgroup_fields: tuple[str, ...]
    _context_fields: tuple[str, ...]
    _id_fields: tuple[str, ...]

    def __init__(
        self, data: pl.DataFrame | pl.LazyFrame | list[dict[str, Any]] | None = None
    ) -> None:
        if data is None:
            self._lf = self._empty_frame()
        elif isinstance(data, list):
            self._lf = self._from_records(data)
        elif isinstance(data, pl.DataFrame):
            self._validate_schema(data)
            self._lf = data.lazy()
        elif isinstance(data, pl.LazyFrame):
            self._lf = data
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        schema = self._lf.collect_schema()
        self._group_fields = self._extract_struct_fields(schema, "groups")
        self._subgroup_fields = self._extract_struct_fields(schema, "subgroups")
        self._context_fields = self._extract_struct_fields(schema, "context")
        self._id_fields = self._extract_struct_fields(schema, "id")

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
                pl.Field("value_json", pl.Utf8),
                pl.Field("value_struct", pl.Struct([])),
                pl.Field("format", pl.Utf8),
                pl.Field("unit", pl.Utf8),
                pl.Field("extras", pl.Struct([])),
            ]
        )
        frame = pl.DataFrame(
            {
                "groups": pl.Series([], dtype=pl.Struct([])),
                "subgroups": pl.Series([], dtype=pl.Struct([])),
                "estimate": pl.Series([], dtype=pl.Utf8),
                "metric": pl.Series([], dtype=pl.Utf8),
                "label": pl.Series([], dtype=pl.Utf8),
                "stat": pl.Series([], dtype=stat_dtype),
                "context": pl.Series([], dtype=pl.Struct([])),
                "id": pl.Series([], dtype=pl.Null),
            }
        )
        return frame.lazy()

    @staticmethod
    def _from_records(records: list[dict[str, Any]]) -> pl.LazyFrame:
        if not records:
            return ARD._empty_frame()

        group_fields = ARD._collect_struct_keys(records, "groups")
        subgroup_fields = ARD._collect_struct_keys(records, "subgroups")
        context_fields = ARD._collect_struct_keys(records, "context")
        id_fields = ARD._collect_struct_keys(records, "id")

        normalised: list[dict[str, Any]] = []
        for row in records:
            normalised.append(
                {
                    "groups": ARD._normalise_mapping(row.get("groups"), group_fields),
                    "subgroups": ARD._normalise_mapping(
                        row.get("subgroups"), subgroup_fields
                    ),
                    "estimate": row.get("estimate"),
                    "metric": row.get("metric"),
                    "label": row.get("label"),
                    "stat": ARD._normalise_stat(row.get("stat")),
                    "context": ARD._normalise_mapping(
                        row.get("context"), context_fields
                    ),
                    "id": ARD._normalise_mapping(row.get("id"), id_fields),
                }
            )

        return pl.DataFrame(normalised).lazy()

    @staticmethod
    def _collect_struct_keys(
        records: Iterable[Mapping[str, Any]], key: str
    ) -> tuple[str, ...]:
        keys: set[str] = set()
        for record in records:
            mapping = record.get(key)
            if isinstance(mapping, Mapping):
                keys.update(str(name) for name in mapping.keys())
        return tuple(sorted(keys))

    @staticmethod
    def _normalise_mapping(
        mapping: Any, keys: tuple[str, ...]
    ) -> dict[str, Any] | None:
        if not keys:
            return (
                None
                if mapping in (None, {})
                else dict(mapping)
                if isinstance(mapping, Mapping)
                else mapping
            )

        result = {field: None for field in keys}
        if isinstance(mapping, Mapping):
            for field in keys:
                if field in mapping:
                    result[field] = mapping[field]
        elif mapping not in (None, {}):
            raise TypeError(
                "groups, subgroups, and context values must be dictionaries"
            )

        return None if all(value is None for value in result.values()) else result

    @staticmethod
    def _normalise_stat(value: Any) -> dict[str, Any]:
        if isinstance(value, Mapping) and {"type", "value"}.issubset(value.keys()):
            stat = ARD._populate_stat(
                value.get("value"),
                type_hint=value.get("type"),
                fmt=value.get("format"),
                unit=value.get("unit"),
            )
            if "extras" in value:
                stat["extras"] = value.get("extras")
            return stat

        return ARD._populate_stat(value)

    @staticmethod
    def _populate_stat(
        raw_value: Any,
        *,
        type_hint: str | None = None,
        fmt: str | None = None,
        unit: str | None = None,
    ) -> dict[str, Any]:
        stat: dict[str, Any] = {
            "type": None,
            "value_float": None,
            "value_int": None,
            "value_bool": None,
            "value_str": None,
            "value_json": None,
            "value_struct": None,
            "format": fmt,
            "unit": unit,
            "extras": None,
        }

        if raw_value is None:
            stat["type"] = type_hint.lower() if isinstance(type_hint, str) else None
            return stat

        hint = type_hint.lower() if isinstance(type_hint, str) else None

        if isinstance(raw_value, Mapping):
            if hint == "struct" or hint is None:
                stat["value_struct"] = dict(raw_value)
                stat["type"] = "struct"
                return stat

        if hint == "struct" and isinstance(raw_value, Mapping):
            stat["value_struct"] = dict(raw_value)
            stat["type"] = "struct"
            return stat

        def assign(field: str, value: Any, label: str) -> dict[str, Any]:
            stat[field] = value
            stat["type"] = label
            return stat

        if hint in {"float", "double", "numeric"}:
            return assign("value_float", float(raw_value), "float")
        if hint in {"int", "integer", "count"}:
            return assign("value_int", int(raw_value), "int")
        if hint in {"bool", "boolean"}:
            return assign("value_bool", bool(raw_value), "bool")
        if hint in {"string", "str", "text"}:
            return assign("value_str", str(raw_value), "string")
        if hint == "struct":
            return assign("value_struct", raw_value, "struct")
        if hint in {"json", "list"}:
            return assign("value_json", ARD._encode_json(raw_value), "json")

        if isinstance(raw_value, bool):
            return assign("value_bool", raw_value, "bool")
        if isinstance(raw_value, int):
            return assign("value_int", raw_value, "int")
        if isinstance(raw_value, float):
            return assign("value_float", raw_value, "float")
        if isinstance(raw_value, str):
            return assign("value_str", raw_value, "string")
        if isinstance(raw_value, (dict, list, tuple)):
            if isinstance(raw_value, Mapping) and "values" in raw_value:
                stat["extras"] = raw_value
                return assign(
                    "value_json",
                    ARD._encode_json(raw_value),
                    hint or "json",
                )
            return assign("value_json", ARD._encode_json(raw_value), "json")

        if isinstance(raw_value, Mapping):
            return assign("value_struct", raw_value, "struct")

        return assign("value_str", str(raw_value), hint or "string")

    @staticmethod
    def _encode_json(value: Any) -> str:
        try:
            return json.dumps(value)
        except TypeError:
            return json.dumps(value, default=str)

    @staticmethod
    def _extract_struct_fields(
        schema: Mapping[str, pl.DataType], column: str
    ) -> tuple[str, ...]:
        dtype = schema.get(column)
        if isinstance(dtype, pl.Struct):
            return tuple(field.name for field in dtype.fields)
        return tuple()

    @staticmethod
    def _validate_schema(df: pl.DataFrame) -> None:
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
        # Keep core columns for backward compatibility when eagerly collecting
        available = self._lf.collect_schema().names()
        desired = [
            col
            for col in [
                "groups",
                "subgroups",
                "subgroup_name",
                "subgroup_value",
                "estimate",
                "metric",
                "stat",
                "context",
                "id",
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
        if type_label == "json":
            encoded = stat.get("value_json")
            if encoded is None:
                return None
            try:
                return json.loads(encoded)
            except json.JSONDecodeError:
                return encoded
        if type_label == "struct":
            return stat.get("value_struct")

        for field in [
            "value_float",
            "value_int",
            "value_bool",
            "value_str",
            "value_json",
            "value_struct",
        ]:
            candidate = stat.get(field)
            if candidate is not None:
                if field == "value_json":
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        return candidate
                if field == "value_struct":
                    return candidate
                return candidate

        return None

    @staticmethod
    def _format_stat(stat: Mapping[str, Any] | None) -> str:
        if stat is None:
            return "NULL"

        value = ARD._stat_value(stat)
        type_label = (stat.get("type") or "").lower()
        fmt = stat.get("format")
        unit = stat.get("unit")

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
            rendered = "" if value is None else str(value)

        if unit:
            rendered = f"{rendered} {unit}"
        return rendered

    def __repr__(self) -> str:
        preview = self._lf.limit(10).collect()
        total = self._lf.select(pl.len()).collect().item()

        # Determine schema display focusing on struct contents
        schema_columns: list[str] = []
        if "groups" in preview.columns:
            if self._group_fields:
                schema_columns.extend(self._group_fields)
            else:
                schema_columns.append("groups")
        if "subgroups" in preview.columns:
            if self._subgroup_fields:
                schema_columns.extend(self._subgroup_fields)
            else:
                schema_columns.append("subgroups")
        if "estimate" in preview.columns:
            schema_columns.append("estimate")
        if "metric" in preview.columns:
            schema_columns.append("metric")
        schema_columns.append("value")

        lines = [f"ARD: {total} results", f"Schema: {' | '.join(schema_columns)}"]

        if total:
            display_exprs: list[pl.Expr] = []

            if "groups" in preview.columns:
                if self._group_fields:
                    for field in self._group_fields:
                        display_exprs.append(
                            pl.col("groups").struct.field(field).alias(field)
                        )
                else:
                    display_exprs.append(
                        pl.when(pl.col("groups").is_null())
                        .then(pl.lit("--"))
                        .otherwise(
                            pl.col("groups").map_elements(
                                lambda g: ", ".join(
                                    f"{k}={v}" for k, v in (g or {}).items()
                                ),
                                return_dtype=pl.Utf8,
                            )
                        )
                        .alias("groups")
                    )

            if "subgroups" in preview.columns:
                if self._subgroup_fields:
                    for field in self._subgroup_fields:
                        display_exprs.append(
                            pl.col("subgroups").struct.field(field).alias(field)
                        )
                else:
                    display_exprs.append(
                        pl.when(pl.col("subgroups").is_null())
                        .then(pl.lit("--"))
                        .otherwise(
                            pl.col("subgroups").map_elements(
                                lambda g: ", ".join(
                                    f"{k}={v}" for k, v in (g or {}).items()
                                ),
                                return_dtype=pl.Utf8,
                            )
                        )
                        .alias("subgroups")
                    )

            if "estimate" in preview.columns:
                display_exprs.append(
                    pl.col("estimate").fill_null("--").alias("estimate")
                )
            if "metric" in preview.columns:
                display_exprs.append(pl.col("metric"))

            display_exprs.append(
                pl.col("stat")
                .map_elements(ARD._format_stat, return_dtype=pl.Utf8)
                .alias("value")
            )

            display = preview.select(display_exprs)
            lines.append(str(display))
            if total > 10:
                lines.append(f"... and {total - 10} more rows")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Filtering utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_struct_filters(
        lf: pl.LazyFrame,
        column: str,
        filters: Mapping[str, Any] | None,
        valid_fields: tuple[str, ...],
    ) -> pl.LazyFrame:
        if not filters:
            return lf

        for key, value in filters.items():
            key_str = str(key)
            if key_str not in valid_fields:
                lf = lf.filter(pl.lit(False))
                continue
            lf = lf.filter(
                pl.when(pl.col(column).is_null())
                .then(False)
                .otherwise(pl.col(column).struct.field(key_str) == value)
            )
        return lf

    def filter(
        self,
        groups: Mapping[str, Any] | pl.Expr | None = None,
        subgroups: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
        metrics: list[str] | str | None = None,
        estimates: list[str] | str | None = None,
        expr: pl.Expr | None = None,
    ) -> ARD:
        if isinstance(groups, pl.Expr):
            group_expr: pl.Expr = groups
            if expr is None:
                expr = group_expr
            else:
                expr = expr & group_expr
            groups = None

        lf = self._apply_struct_filters(self._lf, "groups", groups, self._group_fields)
        lf = self._apply_struct_filters(
            lf, "subgroups", subgroups, self._subgroup_fields
        )
        lf = self._apply_struct_filters(lf, "context", context, self._context_fields)

        if metrics:
            metrics_list = [metrics] if isinstance(metrics, str) else list(metrics)
            lf = lf.filter(pl.col("metric").is_in(metrics_list))

        if estimates:
            estimate_list = (
                [estimates] if isinstance(estimates, str) else list(estimates)
            )
            lf = lf.filter(pl.col("estimate").is_in(estimate_list))

        if expr is not None:
            lf = lf.filter(expr)

        return ARD(lf)

    # ------------------------------------------------------------------
    # Null / empty handling
    # ------------------------------------------------------------------

    def with_empty_as_null(self) -> ARD:
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
                _collapse("groups", self._group_fields),
                _collapse("subgroups", self._subgroup_fields),
                _collapse("context", self._context_fields),
                _collapse("id", self._id_fields),
                pl.when(pl.col("estimate") == "")
                .then(None)
                .otherwise(pl.col("estimate"))
                .alias("estimate"),
            ]
        )
        return ARD(lf)

    def with_null_as_empty(self) -> ARD:
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
                _expand("groups", self._group_fields),
                _expand("subgroups", self._subgroup_fields),
                _expand("context", self._context_fields),
                _expand("id", self._id_fields),
                pl.col("estimate").fill_null(""),
            ]
        )
        return ARD(lf)

    # ------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------

    def unnest(self, columns: list[str] | None = None) -> pl.DataFrame:
        columns = columns or ["groups", "subgroups"]
        lf = self._lf
        for column in columns:
            if column in {"groups", "subgroups", "context", "stat", "id"}:
                has_values = (
                    lf.select(pl.col(column).is_not_null().any()).collect().item()
                )
                if has_values:
                    lf = lf.unnest(column)
        return lf.collect()

    def to_wide(
        self,
        index: list[str] | None = None,
        columns: list[str] | None = None,
        values: str = "stat",
        aggregate: str = "first",
    ) -> pl.DataFrame:
        df = self.unnest(["groups", "subgroups", "context"])

        if columns is None:
            has_estimates = (
                df.filter(pl.col("estimate").is_not_null())["estimate"].n_unique() > 1
            )
            columns = ["estimate", "metric"] if has_estimates else ["metric"]

        if index is None:
            index = [col for col in df.columns if col not in columns + [values, "stat"]]

        if values == "stat":
            df = df.with_columns(
                pl.col("stat")
                .map_elements(ARD._format_stat, return_dtype=pl.Utf8)
                .alias("_value")
            )
            values = "_value"

        if not index or all(df[col].null_count() == len(df) for col in index):
            df = df.with_row_index("_idx")
            index = ["_idx"]

        try:
            pivoted = df.pivot(
                index=index, on=columns, values=values, aggregate_function=aggregate
            )
        except Exception:
            pivoted = df.pivot(
                index=index, on=columns, values=values, aggregate_function="first"
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
        if "stat" in lf.collect_schema().names():
            lf = lf.with_columns(
                pl.col("stat")
                .map_elements(ARD._format_stat, return_dtype=pl.Utf8)
                .alias("value")
            )

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
        df = self._lf.select(["metric", "stat"]).collect()

        values = [ARD._stat_value(stat) for stat in df["stat"]]

        if include_metadata:
            types = [stat.get("type") if stat else None for stat in df["stat"]]
            formats = [stat.get("format") if stat else None for stat in df["stat"]]
            units = [stat.get("unit") if stat else None for stat in df["stat"]]
            return pl.DataFrame(
                {
                    "metric": df["metric"],
                    "value": values,
                    "type": types,
                    "format": formats,
                    "unit": units,
                },
                strict=False,
            )

        return pl.DataFrame({"metric": df["metric"], "value": values}, strict=False)

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
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
