"""
Analysis Results Data (ARD) - Core data structure for evaluation results.

Fixed schema with struct-based storage for maximum flexibility and type safety.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import polars as pl

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr


@dataclass
class ARD:
    """
    Analysis Results Data with fixed schema.

    Schema (always these 6 columns):
        groups: pl.Struct - Group variables and values (null if no groups)
        subgroups: pl.Struct - Subgroup variables and values (null if no subgroups)
        estimate: pl.Utf8 - Model/estimate identifier (null if not applicable)
        metric: pl.Utf8 - Metric identifier
        stat: pl.Struct - Statistical value with type information
        context: pl.Struct - Metadata and computation details

    The stat struct contains:
        - type: str - Data type hint ("int", "float", "string", "struct")
        - value: Any - The actual value
        - format: str - Format string for display (optional)
        - unit: str - Unit of measurement (optional)
    """

    _lf: pl.LazyFrame
    _group_fields: list[tuple[str, pl.DataType]]
    _subgroup_fields: list[tuple[str, pl.DataType]]
    _context_fields: list[tuple[str, pl.DataType]]
    _group_field_names: list[str]
    _subgroup_field_names: list[str]
    _context_field_names: list[str]

    def __init__(
        self, data: pl.DataFrame | pl.LazyFrame | list[dict[str, Any]] | None = None
    ) -> None:
        """Initialize ARD with validation."""
        if data is None:
            # Create empty ARD with known stat schema
            self._lf = self._create_empty()
        elif isinstance(data, list):
            # Create from list of dicts and normalize schemas
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
        self._group_field_names = [name for name, _ in self._group_fields]
        self._subgroup_field_names = [name for name, _ in self._subgroup_fields]
        self._context_field_names = [name for name, _ in self._context_fields]

    @staticmethod
    def _create_empty() -> pl.LazyFrame:
        """Create empty ARD with correct schema."""
        stat_dtype = pl.Struct(
            [
                ("type", pl.Utf8),
                ("value", pl.Object),
                ("format", pl.Utf8),
                ("unit", pl.Utf8),
            ]
        )

        empty_df = pl.DataFrame(
            {
                "groups": pl.Series([], dtype=pl.Struct([])),
                "subgroups": pl.Series([], dtype=pl.Struct([])),
                "estimate": pl.Series([], dtype=pl.Utf8),
                "metric": pl.Series([], dtype=pl.Utf8),
                "stat": pl.Series([], dtype=stat_dtype),
                "context": pl.Series([], dtype=pl.Struct([])),
            }
        )

        return empty_df.lazy()

    @staticmethod
    def _from_records(records: list[dict[str, Any]]) -> pl.LazyFrame:
        """Create ARD from list of records with proper struct handling."""
        if not records:
            return ARD._create_empty()

        group_keys = ARD._collect_mapping_keys(records, "groups")
        subgroup_keys = ARD._collect_mapping_keys(records, "subgroups")
        context_keys = ARD._collect_mapping_keys(records, "context")

        normalized = []
        for record in records:
            normalized.append(
                {
                    "groups": ARD._normalize_mapping(record.get("groups"), group_keys),
                    "subgroups": ARD._normalize_mapping(
                        record.get("subgroups"), subgroup_keys
                    ),
                    "estimate": record.get("estimate"),
                    "metric": record.get("metric"),
                    "stat": ARD._normalize_stat(record.get("stat")),
                    "context": ARD._normalize_mapping(
                        record.get("context"), context_keys
                    ),
                }
            )

        df = pl.DataFrame(normalized)
        return df.lazy()

    @staticmethod
    def _collect_mapping_keys(records: Iterable[dict[str, Any]], key: str) -> list[str]:
        """Collect sorted list of keys used within mapping values for a given column."""
        all_keys: set[str] = set()
        for record in records:
            mapping = record.get(key)
            if isinstance(mapping, dict):
                all_keys.update(str(k) for k in mapping.keys())
        return sorted(all_keys)

    @staticmethod
    def _normalize_mapping(mapping: Any, keys: list[str]) -> dict[str, Any] | None:
        """Normalize mapping-like values to structs with a consistent field set."""
        if not keys:
            return None if mapping in (None, {}) else mapping

        normalized = {field: None for field in keys}

        if isinstance(mapping, dict):
            for field in keys:
                if field in mapping:
                    normalized[field] = mapping[field]
        elif mapping not in (None, {}):
            raise TypeError(
                "Mappings for groups, subgroups, or context must be dictionaries."
            )

        # Preserve null when no keys have data
        if all(value is None for value in normalized.values()):
            return None

        return normalized

    @staticmethod
    def _normalize_stat(value: Any) -> dict[str, Any]:
        """Normalize stat value to standard struct format."""
        if isinstance(value, dict):
            # Check if already in proper struct format
            if "type" in value and "value" in value:
                return {
                    "type": value.get("type"),
                    "value": value.get("value"),
                    "format": value.get("format"),
                    "unit": value.get("unit"),
                }
            else:
                # Treat as complex value
                return {
                    "type": "struct",
                    "value": value,
                    "format": None,
                    "unit": None,
                }
        elif isinstance(value, bool):
            return {
                "type": "bool",
                "value": value,
                "format": None,
                "unit": None,
            }
        elif isinstance(value, (int, float)):
            return {
                "type": "float" if isinstance(value, float) else "int",
                "value": value,
                "format": None,
                "unit": None,
            }
        elif isinstance(value, str):
            return {
                "type": "string",
                "value": value,
                "format": None,
                "unit": None,
            }
        else:
            return {
                "type": "struct",
                "value": value,
                "format": None,
                "unit": None,
            }

    @staticmethod
    def _extract_struct_fields(
        schema: dict[str, pl.DataType], column: str
    ) -> list[tuple[str, pl.DataType]]:
        """Extract struct field definitions for the given column from the schema."""
        dtype = schema.get(column)
        if isinstance(dtype, pl.Struct):
            # pyre-ignore[7]: field.dtype can be DataType or DataTypeClass
            return [(field.name, field.dtype) for field in dtype.fields]
        return []

    def _validate_schema(self, df: pl.DataFrame) -> None:
        """Validate DataFrame has required ARD schema."""
        required = {"groups", "subgroups", "estimate", "metric", "stat", "context"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required ARD columns: {missing}")

    # Core properties

    @property
    def lazy(self) -> pl.LazyFrame:
        """Access underlying LazyFrame."""
        return self._lf

    def collect(self) -> pl.DataFrame:
        """Collect to DataFrame when needed."""
        return self._lf.collect()

    def __len__(self) -> int:
        """Number of rows in ARD."""
        return self.collect().height

    def __repr__(self) -> str:
        """Rich representation for interactive use."""
        # Collect a small sample for display
        sample = self._lf.limit(10).collect()
        n_total = self._lf.select(pl.len()).collect().item()

        lines = [
            f"ARD: {n_total} results",
            "Schema: groups | subgroups | estimate | metric | stat | context",
            "",
        ]

        if n_total > 0:
            # Format sample for display
            display_df = self._format_for_display(sample)
            lines.append(str(display_df))

            if n_total > 10:
                lines.append(f"... and {n_total - 10} more rows")

        return "\n".join(lines)

    def _format_for_display(self, df: pl.DataFrame) -> pl.DataFrame:
        """Format DataFrame for human-readable display."""
        return df.select(
            [
                # Format groups as "var1=val1, var2=val2"
                pl.when(pl.col("groups").is_null())
                .then(pl.lit("--"))
                .otherwise(
                    pl.col("groups").map_elements(
                        lambda g: ", ".join(f"{k}={v}" for k, v in (g or {}).items()),
                        return_dtype=pl.Utf8,
                    )
                )
                .alias("groups"),
                # Format subgroups similarly
                pl.when(pl.col("subgroups").is_null())
                .then(pl.lit("--"))
                .otherwise(
                    pl.col("subgroups").map_elements(
                        lambda s: ", ".join(f"{k}={v}" for k, v in (s or {}).items()),
                        return_dtype=pl.Utf8,
                    )
                )
                .alias("subgroups"),
                # Keep estimate and metric as-is
                pl.col("estimate").fill_null("--"),
                pl.col("metric"),
                # Format stat based on type and format string
                pl.col("stat")
                .map_elements(self._format_stat_value, return_dtype=pl.Utf8)
                .alias("value"),
            ]
        )

    @staticmethod
    def _format_stat_value(stat: dict[str, Any] | None) -> str:
        """Format stat value for display."""
        if stat is None:
            return "NULL"

        value = stat.get("value")
        fmt = stat.get("format")
        unit = stat.get("unit")

        if fmt and value is not None:
            try:
                formatted = fmt.format(value)
            except Exception:
                formatted = str(value)
        elif isinstance(value, float):
            formatted = f"{value:.4g}"
        elif isinstance(value, int):
            formatted = f"{value:,}"
        else:
            formatted = str(value)

        if unit:
            formatted = f"{formatted} {unit}"

        return formatted

    # Query methods

    def filter(
        self,
        groups: dict[str, Any] | None = None,
        subgroups: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        metrics: list[str] | str | None = None,
        estimates: list[str] | str | None = None,
        expr: IntoExpr | None = None,
    ) -> ARD:
        """
        Filter ARD by various criteria.

        Args:
            groups: Dict of group variable filters
            subgroups: Dict of subgroup variable filters
            context: Dict of context metadata filters
            metrics: Metric name(s) to keep
            estimates: Estimate name(s) to keep
            expr: Additional filter expression

        Returns:
            New filtered ARD
        """
        lf = self._lf

        if groups:
            for key, value in groups.items():
                key_str = str(key)
                if key_str not in self._group_field_names:
                    lf = lf.filter(pl.lit(False))
                    continue

                lf = lf.filter(
                    pl.when(pl.col("groups").is_null())
                    .then(False)
                    .otherwise(pl.col("groups").struct.field(key_str) == value)
                )

        if subgroups:
            for key, value in subgroups.items():
                key_str = str(key)
                if key_str not in self._subgroup_field_names:
                    lf = lf.filter(pl.lit(False))
                    continue

                lf = lf.filter(
                    pl.when(pl.col("subgroups").is_null())
                    .then(False)
                    .otherwise(pl.col("subgroups").struct.field(key_str) == value)
                )

        if context:
            for key, value in context.items():
                key_str = str(key)
                if key_str not in self._context_field_names:
                    lf = lf.filter(pl.lit(False))
                    continue

                lf = lf.filter(
                    pl.when(pl.col("context").is_null())
                    .then(False)
                    .otherwise(pl.col("context").struct.field(key_str) == value)
                )

        if metrics:
            if isinstance(metrics, str):
                metrics = [metrics]
            lf = lf.filter(pl.col("metric").is_in(metrics))

        if estimates:
            if isinstance(estimates, str):
                estimates = [estimates]
            lf = lf.filter(pl.col("estimate").is_in(estimates))

        if expr is not None:
            lf = lf.filter(expr)

        return ARD(lf)

    def with_empty_as_null(self) -> ARD:
        """Convert empty dicts/strings to null."""
        return ARD(
            self._lf.with_columns(
                [
                    self._nullify_empty_struct("groups", self._group_fields),
                    self._nullify_empty_struct("subgroups", self._subgroup_fields),
                    self._nullify_empty_struct("context", self._context_fields),
                    pl.when(pl.col("estimate") == "")
                    .then(None)
                    .otherwise(pl.col("estimate"))
                    .alias("estimate"),
                ]
            )
        )

    def with_null_as_empty(self) -> ARD:
        """Convert null values to empty dicts/strings."""
        return ARD(
            self._lf.with_columns(
                [
                    self._fill_null_struct("groups", self._group_fields),
                    self._fill_null_struct("subgroups", self._subgroup_fields),
                    self._fill_null_struct("context", self._context_fields),
                    pl.col("estimate").fill_null(""),
                ]
            )
        )

    def _nullify_empty_struct(
        self, column: str, fields: Sequence[tuple[str, pl.DataType]]
    ) -> pl.Expr:
        """Set struct column to null when all fields are null or it is already null."""
        if not fields:
            return pl.col(column)

        field_exprs = [pl.col(column).struct.field(name) for name, _ in fields]
        all_fields_null = reduce(
            lambda acc, expr: acc & expr.is_null(), field_exprs, pl.lit(True)
        )
        return (
            pl.when(pl.col(column).is_null() | all_fields_null)
            .then(None)
            .otherwise(pl.col(column))
            .alias(column)
        )

    def _fill_null_struct(
        self, column: str, fields: Sequence[tuple[str, pl.DataType]]
    ) -> pl.Expr:
        """Replace null struct column with an empty struct that preserves schema."""
        if not fields:
            return pl.col(column)

        empty_struct = pl.struct(
            [pl.lit(None).cast(dtype).alias(name) for name, dtype in fields]
        )
        return (
            pl.when(pl.col(column).is_null())
            .then(empty_struct)
            .otherwise(pl.col(column))
            .alias(column)
        )

    # Transformation methods

    def unnest(self, columns: list[str] | None = None) -> pl.DataFrame:
        """
        Unnest struct columns to flat format.

        Args:
            columns: Specific columns to unnest (default: groups, subgroups)

        Returns:
            DataFrame with unnested columns
        """
        if columns is None:
            columns = ["groups", "subgroups"]

        lf = self._lf
        for col in columns:
            if col in ["groups", "subgroups", "context"]:
                # Check if column has any non-null values
                has_values = lf.select(pl.col(col).is_not_null().any()).collect().item()
                if has_values:
                    lf = lf.unnest(col)

        return lf.collect()

    def to_wide(
        self,
        index: list[str] | None = None,
        columns: list[str] | None = None,
        values: str = "stat",
        aggregate_fn: str = "first",
    ) -> pl.DataFrame:
        """
        Pivot to wide format.

        Args:
            index: Columns to use as index (auto-detected if None)
            columns: Columns to pivot (default: ["estimate", "metric"])
            values: Column containing values (default: "stat")
            aggregate_fn: Aggregation function for duplicate values

        Returns:
            Wide-format DataFrame
        """
        # First unnest groups/subgroups for index
        df = self.unnest(["groups", "subgroups"])

        if columns is None:
            # Auto-detect pivot columns
            has_estimates = (
                df.filter(pl.col("estimate").is_not_null())["estimate"].n_unique() > 1
            )
            columns = ["estimate", "metric"] if has_estimates else ["metric"]

        if index is None:
            # Auto-detect index columns - be more careful about null values
            index = [
                c for c in df.columns if c not in columns + [values, "context", "stat"]
            ]

        # Extract value from stat struct for pivoting
        if values == "stat":
            df = df.with_columns(
                pl.col("stat").struct.field("value").alias("_pivot_value")
            )
            values = "_pivot_value"

        # Handle case where no meaningful index exists
        if not index or all(df[col].null_count() == len(df) for col in index):
            # Create a dummy index
            df = df.with_row_index("_idx")
            index = ["_idx"]

        try:
            result = df.pivot(
                index=index,
                on=columns,
                values=values,
                aggregate_function=aggregate_fn,
            )
        except Exception:
            # Fallback: use first() aggregation explicitly
            result = df.pivot(
                index=index,
                on=columns,
                values=values,
                aggregate_function="first",
            )

        # Clean up temporary columns
        if "_idx" in result.columns:
            result = result.drop("_idx")

        return result

    def to_long(self) -> pl.DataFrame:
        """
        Convert to fully normalized long format.

        Returns:
            DataFrame with all structs unnested
        """
        return self.unnest(["groups", "subgroups", "context", "stat"])

    def get_stats(
        self, as_values: bool = True, include_metadata: bool = False
    ) -> pl.DataFrame:
        """
        Extract statistics in convenient format.

        Args:
            as_values: Extract just the values from stat struct
            include_metadata: Include format and unit information

        Returns:
            DataFrame with stat information
        """
        if as_values and not include_metadata:
            # Just extract values
            return self._lf.select(
                [pl.col("metric"), pl.col("stat").struct.field("value").alias("value")]
            ).collect()
        elif include_metadata:
            # Unnest full stat struct
            return self._lf.select(
                [
                    pl.col("metric"),
                    pl.col("stat").struct.field("value").alias("value"),
                    pl.col("stat").struct.field("type").alias("type"),
                    pl.col("stat").struct.field("format").alias("format"),
                    pl.col("stat").struct.field("unit").alias("unit"),
                ]
            ).collect()
        else:
            # Return stat struct as-is
            return self._lf.select(["metric", "stat"]).collect()

    # Summary methods

    def summary(self) -> dict[str, Any]:
        """Generate summary statistics about the ARD."""
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
        """Print detailed description of ARD contents."""
        summary = self.summary()

        print("=" * 50)
        print(f"ARD Summary: {summary['n_rows']} results")
        print("=" * 50)

        print(f"\nMetrics ({summary['n_metrics']}):")
        for metric in summary["metrics"]:
            print(f"  - {metric}")

        if summary["n_estimates"] > 0:
            print(f"\nEstimates ({summary['n_estimates']}):")
            for est in summary["estimates"]:
                if est:  # Skip null/empty
                    print(f"  - {est}")

        if summary["n_groups"] > 0:
            print(f"\nGroup combinations: {summary['n_groups']}")

        if summary["n_subgroups"] > 0:
            print(f"Subgroup combinations: {summary['n_subgroups']}")

        print("\nFirst 5 results:")
        print(self._format_for_display(self._lf.limit(5).collect()))
