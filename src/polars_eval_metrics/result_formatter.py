from __future__ import annotations

from typing import Any, Mapping, Sequence

import polars as pl

from .ard import ARD
from .evaluation_context import EstimateCatalog, FormatterContext
from .utils import parse_json_tokens


def convert_to_ard(result_lf: pl.LazyFrame, context: FormatterContext) -> ARD:
    """Convert the evaluator output into canonical ARD columns lazily."""

    schema = result_lf.collect_schema()
    schema_names = set(schema.names())

    ard_frame = result_lf.with_columns(
        [
            _expr_groups(schema, context.group_by),
            _expr_subgroups(schema, context.subgroup_by),
            _expr_estimate(schema, context.estimate_catalog),
            _expr_metric_enum(context.metric_catalog.names),
            _expr_label_enum(context.metric_catalog.labels),
            _expr_stat_struct(schema),
            _expr_context_struct(schema, context),
        ]
    )

    warning_expr = (
        pl.col("_diagnostic_warning")
        if "_diagnostic_warning" in schema_names
        else pl.lit([], dtype=pl.List(pl.Utf8))
    )
    error_expr = (
        pl.col("_diagnostic_error")
        if "_diagnostic_error" in schema_names
        else pl.lit([], dtype=pl.List(pl.Utf8))
    )

    ard_frame = ard_frame.with_columns(
        [
            pl.col("stat")
            .map_elements(ARD._format_stat, return_dtype=pl.Utf8)
            .alias("stat_fmt"),
            warning_expr.alias("warning"),
            error_expr.alias("error"),
        ]
    )

    cleanup_cols = [
        "_value_kind",
        "_value_format",
        "_value_float",
        "_value_int",
        "_value_bool",
        "_value_str",
        "_value_struct",
        "_diagnostic_warning",
        "_diagnostic_error",
    ]
    drop_cols = [col for col in cleanup_cols if col in schema_names]
    if drop_cols:
        ard_frame = ard_frame.drop(drop_cols)

    return ARD(ard_frame)


def build_group_pivot(
    long_df: pl.DataFrame,
    context: FormatterContext,
    *,
    column_order_by: str,
    row_order_by: str,
) -> pl.DataFrame:
    """Pivot results with groups as rows and model × metric as columns."""

    group_labels = [
        label for label in context.group_by.values() if label in long_df.columns
    ]
    subgroup_present = (
        "subgroup_name" in long_df.columns and "subgroup_value" in long_df.columns
    )

    if row_order_by == "subgroup" and subgroup_present:
        index_cols = ["subgroup_name", "subgroup_value"] + group_labels
    else:
        index_cols = group_labels + (
            ["subgroup_name", "subgroup_value"] if subgroup_present else []
        )

    display_col = (
        "estimate_label" if "estimate_label" in long_df.columns else "estimate"
    )

    result, sections = _build_pivot_table(
        long_df,
        index_cols=index_cols,
        default_on=[display_col, "label"],
        scoped_on=[("global", ["label"]), ("group", ["label"])],
    )

    section_lookup = {name: cols for name, cols in sections}

    if result.is_empty():
        if index_cols:
            return pl.DataFrame({col: [] for col in index_cols})
        return pl.DataFrame()

    value_cols = [col for col in result.columns if col not in index_cols]
    default_cols = section_lookup.get("default", [])
    default_cols = [
        col for col in default_cols if col.startswith('{"') and col.endswith('"}')
    ]

    estimate_order_lookup = context.estimate_catalog.label_order
    metric_label_order_lookup = context.metric_catalog.label_order
    metric_name_order_lookup = context.metric_catalog.name_order

    def metric_order(label: str) -> int:
        if label in metric_label_order_lookup:
            return metric_label_order_lookup[label]
        return metric_name_order_lookup.get(label, len(metric_label_order_lookup))

    def estimate_order(label: str) -> int:
        return estimate_order_lookup.get(label, len(estimate_order_lookup))

    def sort_default(columns: list[str]) -> list[str]:
        def parse(column: str) -> tuple[str, str]:
            inner = column[2:-2]
            parts = inner.split('","')
            return (parts[0], parts[1]) if len(parts) == 2 else (column, "")

        if column_order_by == "metrics":
            return sorted(
                columns,
                key=lambda col: (
                    metric_order(parse(col)[1]),
                    estimate_order(parse(col)[0]),
                ),
            )
        return sorted(
            columns,
            key=lambda col: (
                estimate_order(parse(col)[0]),
                metric_order(parse(col)[1]),
            ),
        )

    ordered = (
        index_cols
        + section_lookup.get("global", [])
        + section_lookup.get("group", [])
        + sort_default(default_cols)
    )

    remaining = [col for col in value_cols if col not in ordered]
    ordered.extend(remaining)
    ordered = [col for col in ordered if col in result.columns]

    if "subgroup_value" in result.columns:
        if context.subgroup_categories and all(
            isinstance(cat, str) for cat in context.subgroup_categories
        ):
            result = result.with_columns(
                pl.col("subgroup_value").cast(
                    pl.Enum(list(context.subgroup_categories))
                )
            )
        else:
            result = result.with_columns(pl.col("subgroup_value").cast(pl.Utf8))

    sort_columns: list[str] = []
    temp_sort_columns: list[str] = []
    subgroup_order_map = {
        label: idx for idx, label in enumerate(context.subgroup_by.values())
    }

    if row_order_by == "group":
        sort_columns.extend([col for col in group_labels if col in result.columns])
        if "subgroup_name" in result.columns and context.subgroup_by:
            result = result.with_columns(
                pl.col("subgroup_name")
                .replace(subgroup_order_map)
                .fill_null(len(subgroup_order_map))
                .cast(pl.Int32)
                .alias("__subgroup_name_order")
            )
            temp_sort_columns.append("__subgroup_name_order")
            sort_columns.append("__subgroup_name_order")
        if "subgroup_value" in result.columns:
            sort_columns.append("subgroup_value")
    else:
        if "subgroup_value" in result.columns:
            sort_columns.append("subgroup_value")
        if "subgroup_name" in result.columns and context.subgroup_by:
            result = result.with_columns(
                pl.col("subgroup_name")
                .replace(subgroup_order_map)
                .fill_null(len(subgroup_order_map))
                .cast(pl.Int32)
                .alias("__subgroup_name_order")
            )
            temp_sort_columns.append("__subgroup_name_order")
            sort_columns.insert(0, "__subgroup_name_order")
        sort_columns.extend([col for col in group_labels if col in result.columns])

    if "estimate" in result.columns:
        sort_columns.append("estimate")

    if sort_columns:
        result = result.sort(sort_columns)
    if temp_sort_columns:
        result = result.drop(temp_sort_columns)

    seen: set[str] = set()
    deduped: list[str] = []
    for col in ordered:
        if col not in seen:
            deduped.append(col)
            seen.add(col)

    return result.select(deduped)


def build_model_pivot(
    long_df: pl.DataFrame,
    context: FormatterContext,
    *,
    column_order_by: str,
    row_order_by: str,
) -> pl.DataFrame:
    """Pivot results with models as rows and group × metric as columns."""

    subgroup_present = (
        "subgroup_name" in long_df.columns and "subgroup_value" in long_df.columns
    )

    if row_order_by == "subgroup" and subgroup_present:
        index_cols = ["estimate", "subgroup_name", "subgroup_value"]
    else:
        index_cols = ["estimate"] + (
            ["subgroup_name", "subgroup_value"] if subgroup_present else []
        )

    if "estimate" in index_cols:
        estimate_series = (
            long_df.get_column("estimate") if "estimate" in long_df.columns else None
        )
        if estimate_series is None or estimate_series.is_null().all():
            index_cols = [col for col in index_cols if col != "estimate"]

    result, sections = _build_pivot_table(
        long_df,
        index_cols=index_cols,
        default_on=[*context.group_by.values(), "label"],
        scoped_on=[
            ("global", ["label"]),
            ("group", [*context.group_by.values(), "label"]),
        ],
    )

    section_lookup = {name: cols for name, cols in sections}

    group_labels = list(context.group_by.values())
    group_label_count = len(group_labels)
    group_value_orders: list[dict[Any, int]] = []

    if group_label_count:
        for label in group_labels:
            if label not in long_df.columns:
                group_value_orders.append({})
                continue

            series = long_df.get_column(label)
            dtype = series.dtype

            if isinstance(dtype, pl.Enum):
                categories = dtype.categories.to_list()
            else:
                categories = sorted(series.drop_nulls().unique().to_list())

            group_value_orders.append(
                {value: idx for idx, value in enumerate(categories)}
            )

    metric_label_order_lookup = context.metric_catalog.label_order
    metric_name_order_lookup = context.metric_catalog.name_order
    estimate_label_map = context.estimate_catalog.key_to_label

    def metric_order(label: str) -> int:
        if label in metric_label_order_lookup:
            return metric_label_order_lookup[label]
        return metric_name_order_lookup.get(label, len(metric_label_order_lookup))

    def group_order(tokens: tuple[str, ...]) -> tuple[int, ...]:
        if not group_label_count:
            return tuple()
        values = tokens[:group_label_count]
        order_positions: list[int] = []
        for idx, value in enumerate(values):
            mapping = group_value_orders[idx] if idx < len(group_value_orders) else {}
            order_positions.append(mapping.get(value, len(mapping)))
        return tuple(order_positions)

    def column_sort_key(column: str) -> tuple[Any, ...]:
        tokens = parse_json_tokens(column)
        if tokens is None:
            return (float("inf"), column)
        metric_label = tokens[-1] if tokens else ""
        metric_idx = metric_order(metric_label)
        group_idx = group_order(tokens)
        if column_order_by == "metrics":
            return (metric_idx, group_idx, tokens)
        return (group_idx, metric_idx, tokens)

    if "group" in section_lookup:
        section_lookup["group"] = sorted(section_lookup["group"], key=column_sort_key)

    if "default" in section_lookup:
        section_lookup["default"] = sorted(
            section_lookup["default"], key=column_sort_key
        )

    if "estimate" in result.columns:
        result = result.with_columns(
            pl.col("estimate")
            .map_elements(
                lambda val, mapping=estimate_label_map: mapping.get(val, val),
                return_dtype=pl.Utf8,
            )
            .alias("estimate_label")
        )

    if "subgroup_value" in result.columns:
        if context.subgroup_categories and all(
            isinstance(cat, str) for cat in context.subgroup_categories
        ):
            result = result.with_columns(
                pl.col("subgroup_value").cast(
                    pl.Enum(list(context.subgroup_categories))
                )
            )
        else:
            result = result.with_columns(pl.col("subgroup_value").cast(pl.Utf8))

    ordered = (
        [col for col in index_cols if col in result.columns]
        + [col for col in section_lookup.get("global", []) if col in result.columns]
        + [col for col in section_lookup.get("group", []) if col in result.columns]
        + [col for col in section_lookup.get("default", []) if col in result.columns]
    )

    remaining = [col for col in result.columns if col not in ordered]
    ordered.extend(remaining)

    result = result.select(ordered)

    sort_columns: list[str] = []
    if "subgroup_value" in result.columns:
        sort_columns.append("subgroup_value")
    if "estimate" in result.columns:
        sort_columns.append("estimate")
    for label in context.group_by.values():
        if label in result.columns:
            sort_columns.append(label)
    if sort_columns:
        result = result.sort(sort_columns)

    return result


def _expr_groups(schema: pl.Schema, group_by: Mapping[str, str]) -> pl.Expr:
    group_cols = [col for col in group_by.keys() if col in schema.names()]
    if not group_cols:
        return pl.lit(None).alias("groups")

    dtype = pl.Struct([pl.Field(col, schema[col]) for col in group_cols])
    return (
        pl.when(pl.all_horizontal([pl.col(col).is_null() for col in group_cols]))
        .then(pl.lit(None, dtype=dtype))
        .otherwise(pl.struct([pl.col(col).alias(col) for col in group_cols]))
        .alias("groups")
    )


def _expr_subgroups(schema: pl.Schema, subgroup_by: Mapping[str, str]) -> pl.Expr:
    if (
        not subgroup_by
        or "subgroup_name" not in schema.names()
        or "subgroup_value" not in schema.names()
    ):
        return pl.lit(None).alias("subgroups")

    labels = list(subgroup_by.values())
    dtype = pl.Struct([pl.Field(label, pl.Utf8) for label in labels])
    fields = [
        pl.when(pl.col("subgroup_name") == pl.lit(label))
        .then(pl.col("subgroup_value").cast(pl.Utf8))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
        .alias(label)
        for label in labels
    ]
    return (
        pl.when(pl.col("subgroup_name").is_null() | pl.col("subgroup_value").is_null())
        .then(pl.lit(None, dtype=dtype))
        .otherwise(pl.struct(fields))
        .alias("subgroups")
    )


def _expr_estimate(schema: pl.Schema, estimate_catalog: EstimateCatalog) -> pl.Expr:
    null_utf8 = pl.lit(None, dtype=pl.Utf8)
    if "estimate" not in schema.names():
        return null_utf8.alias("estimate")

    estimate_names = list(estimate_catalog.keys)
    if estimate_names:
        return (
            pl.col("estimate")
            .cast(pl.Utf8)
            .replace({name: name for name in estimate_names})
            .cast(pl.Enum(estimate_names))
            .alias("estimate")
        )

    return pl.col("estimate").cast(pl.Utf8).alias("estimate")


def _expr_metric_enum(metric_names: Sequence[str]) -> pl.Expr:
    metric_categories = list(dict.fromkeys(metric_names))
    return (
        pl.col("metric")
        .cast(pl.Utf8)
        .replace({name: name for name in metric_categories})
        .cast(pl.Enum(metric_categories))
        .alias("metric")
    )


def _expr_label_enum(metric_labels: Sequence[str]) -> pl.Expr:
    unique_labels = list(dict.fromkeys(metric_labels))
    return pl.col("label").cast(pl.Enum(unique_labels)).alias("label")


def _expr_stat_struct(schema: pl.Schema) -> pl.Expr:
    null_utf8 = pl.lit(None, dtype=pl.Utf8)
    null_float = pl.lit(None, dtype=pl.Float64)
    null_int = pl.lit(None, dtype=pl.Int64)
    null_bool = pl.lit(None, dtype=pl.Boolean)
    null_struct_expr = pl.lit(None, dtype=pl.Struct([]))

    kind_expr = pl.col("_value_kind") if "_value_kind" in schema.names() else None
    format_col = (
        pl.col("_value_format") if "_value_format" in schema.names() else null_utf8
    )

    float_value = (
        pl.col("_value_float") if "_value_float" in schema.names() else null_float
    )
    int_value = pl.col("_value_int") if "_value_int" in schema.names() else null_int
    bool_value = pl.col("_value_bool") if "_value_bool" in schema.names() else null_bool
    string_value = pl.col("_value_str") if "_value_str" in schema.names() else null_utf8
    struct_value = (
        pl.col("_value_struct")
        if "_value_struct" in schema.names()
        else null_struct_expr
    )

    inferred_kind = "float"
    if kind_expr is None:
        value_dtype = schema.get("value") if "value" in schema.names() else None
        inferred_kind = _infer_value_kind_from_dtype(value_dtype)
        kind_expr = pl.lit(inferred_kind, dtype=pl.Utf8)

    type_label = pl.when(kind_expr.is_null()).then(null_utf8).otherwise(kind_expr)

    return pl.struct(
        [
            type_label.alias("type"),
            float_value.alias("value_float"),
            int_value.alias("value_int"),
            bool_value.alias("value_bool"),
            string_value.alias("value_str"),
            struct_value.alias("value_struct"),
            format_col.alias("format"),
        ]
    ).alias("stat")


def _expr_context_struct(schema: pl.Schema, context: FormatterContext) -> pl.Expr:
    null_utf8 = pl.lit(None, dtype=pl.Utf8)
    fields = []
    for field in ("metric_type", "scope", "label"):
        if field in schema.names():
            fields.append(pl.col(field).cast(pl.Utf8).alias(field))
        else:
            fields.append(null_utf8.alias(field))
    if "estimate" in schema.names():
        label_map = context.estimate_catalog.key_to_label
        fields.append(
            pl.col("estimate")
            .cast(pl.Utf8)
            .map_elements(
                lambda val, mapping=label_map: mapping.get(val, val),
                return_dtype=pl.Utf8,
            )
            .alias("estimate_label")
        )
    else:
        fields.append(null_utf8.alias("estimate_label"))
    return pl.struct(fields).alias("context")


def _infer_value_kind_from_dtype(dtype: pl.DataType | None) -> str:
    if dtype is None or dtype == pl.Null:
        return "float"
    if dtype == pl.Struct:
        return "struct"
    if dtype == pl.Boolean:
        return "bool"
    if dtype == pl.Utf8:
        return "string"
    if hasattr(dtype, "is_numeric") and dtype.is_numeric():
        return "int" if dtype.is_integer() else "float"
    return "string"


def _pivot_frame(
    df: pl.DataFrame,
    *,
    index_cols: Sequence[str],
    on_cols: Sequence[str],
) -> pl.DataFrame:
    if df.is_empty():
        if index_cols:
            return pl.DataFrame({col: [] for col in index_cols})
        return pl.DataFrame()

    if index_cols:
        return df.pivot(
            index=list(index_cols),
            on=list(on_cols),
            values="value",
            aggregate_function="first",
        )

    with_idx = df.with_row_index("_idx")
    return with_idx.pivot(
        index=["_idx"],
        on=list(on_cols),
        values="value",
        aggregate_function="first",
    ).drop("_idx")


def _merge_pivot_frames(
    base: pl.DataFrame,
    candidate: pl.DataFrame,
    index_cols: Sequence[str],
) -> pl.DataFrame:
    if base.is_empty():
        return candidate
    if candidate.is_empty():
        return base

    if not index_cols:
        return pl.concat([base, candidate], how="horizontal")

    if candidate.height == 1:
        broadcast_cols = [col for col in candidate.columns if col not in index_cols]
        if not broadcast_cols:
            return base
        row_values = candidate.row(0, named=True)
        return base.with_columns(
            [pl.lit(row_values[col]).alias(col) for col in broadcast_cols]
        )

    join_index_cols = list(index_cols)
    all_null_cols: list[str] = []
    for col in index_cols:
        if col in candidate.columns:
            column = candidate.get_column(col)
            if column.null_count() == candidate.height:
                all_null_cols.append(col)
    if all_null_cols:
        join_index_cols = [col for col in join_index_cols if col not in all_null_cols]
        candidate = candidate.drop(all_null_cols)

    if not join_index_cols:
        value_cols = [col for col in candidate.columns if col not in index_cols]
        if not value_cols:
            return base
        candidate_unique = candidate.select(value_cols).unique()
        if candidate_unique.height == 0:
            return base
        if candidate_unique.height > 1:
            candidate_unique = candidate_unique.head(1)
        row_values = candidate_unique.row(0, named=True)
        return base.with_columns(
            [pl.lit(row_values[col]).alias(col) for col in row_values]
        )

    return base.join(candidate, on=join_index_cols, how="left")


def _build_pivot_table(
    long_df: pl.DataFrame,
    *,
    index_cols: Sequence[str],
    default_on: Sequence[str],
    scoped_on: Sequence[tuple[str, Sequence[str]]],
) -> tuple[pl.DataFrame, list[tuple[str, list[str]]]]:
    default_df = long_df.filter(pl.col("scope").is_null())
    pivot = _pivot_frame(default_df, index_cols=index_cols, on_cols=default_on)
    sections: list[tuple[str, list[str]]] = [
        ("default", [col for col in pivot.columns if col not in index_cols])
    ]

    for scope_name, on_cols in scoped_on:
        scoped_df = long_df.filter(pl.col("scope") == scope_name)
        scoped_pivot = _pivot_frame(scoped_df, index_cols=index_cols, on_cols=on_cols)
        if scoped_pivot.is_empty():
            continue
        sections.append(
            (scope_name, [col for col in scoped_pivot.columns if col not in index_cols])
        )
        pivot = _merge_pivot_frames(pivot, scoped_pivot, index_cols)

    return pivot, sections


def format_verbose_frame(ard: ARD) -> pl.DataFrame:
    """Render an ARD as a fully expanded DataFrame suitable for inspection."""

    long_df = ard.to_long()
    group_sort_cols = list(ard._group_fields)
    subgroup_struct_cols = list(ard._subgroup_fields)
    sort_cols: list[str] = []

    for col in group_sort_cols:
        if col in long_df.columns:
            sort_cols.append(col)

    for col in ("subgroup_name", "subgroup_value"):
        if col in long_df.columns:
            sort_cols.append(col)

    for col in subgroup_struct_cols:
        if col in long_df.columns and col not in sort_cols:
            sort_cols.append(col)

    for col in ("metric", "estimate"):
        if col in long_df.columns:
            sort_cols.append(col)

    if sort_cols:
        long_df = long_df.sort(sort_cols)

    preferred_order = [
        "id",
        "groups",
        "subgroups",
        "subgroup_name",
        "subgroup_value",
        "estimate",
        "metric",
        "label",
        "value",
        "stat",
        "stat_fmt",
        "context",
        "warning",
        "error",
    ]
    ordered_columns = [col for col in preferred_order if col in long_df.columns]
    remaining_columns = [col for col in long_df.columns if col not in ordered_columns]
    return long_df.select(ordered_columns + remaining_columns)


def format_compact_frame(ard: ARD) -> pl.DataFrame:
    """Render an ARD as a compact DataFrame with struct columns flattened."""

    verbose_df = format_verbose_frame(ard)
    flattened = _flatten_struct_columns(
        verbose_df,
        group_fields=ard._group_fields,
        subgroup_fields=ard._subgroup_fields,
    )
    detail_cols = [
        col
        for col in ("stat", "stat_fmt", "context", "warning", "error")
        if col in flattened.columns
    ]
    if detail_cols:
        flattened = flattened.drop(detail_cols)
    return flattened


def _flatten_struct_columns(
    df: pl.DataFrame,
    *,
    group_fields: Sequence[str],
    subgroup_fields: Sequence[str],
) -> pl.DataFrame:
    """Flatten struct columns for a compact DataFrame view."""

    working = df

    if "groups" in working.columns and group_fields:
        group_exprs = [
            pl.col("groups").struct.field(field).alias(field) for field in group_fields
        ]
        working = working.with_columns(group_exprs)

    if "subgroups" in working.columns and subgroup_fields:
        subgroup_exprs = [
            pl.col("subgroups").struct.field(field).alias(field)
            for field in subgroup_fields
        ]
        working = working.with_columns(subgroup_exprs)

    drop_cols = [col for col in ("id", "groups", "subgroups") if col in working.columns]
    if drop_cols:
        working = working.drop(drop_cols)

    return working
