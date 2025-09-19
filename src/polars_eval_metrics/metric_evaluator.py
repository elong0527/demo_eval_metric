"""
Unified Metric Evaluation Pipeline

This module implements a simplified, unified evaluation pipeline for computing metrics
using Polars LazyFrames with comprehensive support for scopes, groups, and subgroups.
"""

from typing import Any

# pyre-strict

import polars as pl

from .metric_define import MetricDefine, MetricScope, MetricType
from .metric_registry import MetricRegistry


class MetricEvaluator:
    """Unified metric evaluation pipeline"""

    # Instance attributes with type annotations
    df_raw: pl.LazyFrame
    metrics: list[MetricDefine]
    ground_truth: str
    estimates: dict[str, str]  # Maps estimate names to display labels
    group_by: dict[str, str]  # Maps group column names to display labels
    subgroup_by: dict[str, str]  # Maps subgroup column names to display labels
    filter_expr: pl.Expr | None
    error_params: dict[str, dict[str, Any]]
    df: pl.LazyFrame
    _evaluation_cache: dict[tuple[tuple[str, ...], tuple[str, ...]], pl.DataFrame]

    def __init__(
        self,
        df: pl.DataFrame | pl.LazyFrame,
        metrics: MetricDefine | list[MetricDefine],
        ground_truth: str = "actual",
        estimates: str | list[str] | dict[str, str] | None = None,
        group_by: list[str] | dict[str, str] | None = None,
        subgroup_by: list[str] | dict[str, str] | None = None,
        filter_expr: pl.Expr | None = None,
        error_params: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Initialize evaluator with complete evaluation context

        Args:
            df: Input data as DataFrame or LazyFrame
            metrics: Metric definitions to evaluate
            ground_truth: Column name containing ground truth values
            estimates: Estimate column names. Can be:
                - str: Single column name
                - list[str]: List of column names
                - dict[str, str]: Mapping from column names to display labels
            group_by: Columns to group by for analysis. Can be:
                - list[str]: List of column names
                - dict[str, str]: Mapping from column names to display labels
            subgroup_by: Columns for subgroup analysis. Can be:
                - list[str]: List of column names
                - dict[str, str]: Mapping from column names to display labels
            filter_expr: Optional filter expression
            error_params: Parameters for error calculations
        """
        # Store data as LazyFrame
        self.df_raw = df.lazy() if isinstance(df, pl.DataFrame) else df

        # Normalize inputs to lists
        self.metrics = [metrics] if isinstance(metrics, MetricDefine) else metrics
        self.ground_truth = ground_truth

        # Process inputs using dedicated methods
        self.estimates = self._process_estimates(estimates)
        self.group_by = self._process_grouping(group_by)
        self.subgroup_by = self._process_grouping(subgroup_by)
        self.filter_expr = filter_expr
        self.error_params = error_params or {}

        # Apply base filter once
        self.df = self._apply_base_filter()

        # Initialize evaluation cache
        self._evaluation_cache = {}

    def _apply_base_filter(self) -> pl.LazyFrame:
        """Apply initial filter if provided"""
        if self.filter_expr is not None:
            return self.df_raw.filter(self.filter_expr)
        return self.df_raw

    def _get_cache_key(
        self,
        metrics: MetricDefine | list[MetricDefine] | None,
        estimates: str | list[str] | None,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Generate cache key for evaluation parameters"""
        target_metrics = self._resolve_metrics(metrics)
        target_estimates = self._resolve_estimates(estimates)

        # Create hashable key from metric names and estimates
        metric_names = tuple(sorted(m.name for m in target_metrics))
        estimate_names = tuple(sorted(target_estimates))

        return (metric_names, estimate_names)

    def _get_cached_evaluation(
        self,
        metrics: MetricDefine | list[MetricDefine] | None = None,
        estimates: str | list[str] | None = None,
    ) -> pl.DataFrame:
        """Get cached evaluation result or compute and cache if not exists"""
        cache_key = self._get_cache_key(metrics, estimates)

        if cache_key not in self._evaluation_cache:
            # Compute and cache the result
            result_data = self.evaluate(
                metrics=metrics, estimates=estimates, collect=True
            )
            # Ensure we have a DataFrame for caching
            result = (
                result_data
                if isinstance(result_data, pl.DataFrame)
                else result_data.collect()
            )
            self._evaluation_cache[cache_key] = result

        return self._evaluation_cache[cache_key]

    def clear_cache(self) -> None:
        """Clear the evaluation cache"""
        self._evaluation_cache.clear()

    def _build_index_cols(
        self, long_df: pl.DataFrame, include_estimate: bool = False
    ) -> list[str]:
        """Build index columns for pivot operations"""
        index_cols = []
        # Put subgroup columns first when they exist
        if "subgroup_name" in long_df.columns:
            index_cols.extend(["subgroup_name", "subgroup_value"])
        if not include_estimate and self.group_by:
            index_cols.extend(list(self.group_by.keys()))
        if include_estimate:
            index_cols.append("estimate")
        return index_cols

    def _add_group_columns(
        self, result: pl.DataFrame, group_data: pl.DataFrame
    ) -> pl.DataFrame:
        """Add group scope columns by broadcasting group x metric combinations"""
        if group_data.is_empty():
            return result

        # Pivot group data to get JSON-like column names
        # Add a dummy index column for pivoting
        group_data_with_index = group_data.with_columns(pl.lit(1).alias("_dummy_index"))

        if self.group_by:
            # Use the group_by columns plus label for pivot
            pivot_cols = list(self.group_by.keys()) + ["label"]
            group_pivoted = group_data_with_index.pivot(
                on=pivot_cols,
                values="value",
                index=["_dummy_index"],
                aggregate_function="first",  # Take first value if duplicates
            )
        else:
            # Pivot by label only when no group_by
            group_pivoted = group_data_with_index.pivot(
                on="label",
                values="value",
                index=["_dummy_index"],
                aggregate_function="first",
            )

        # Remove the dummy index column
        group_pivoted = group_pivoted.drop("_dummy_index")

        # Broadcast the pivoted columns to all rows in result
        for col in group_pivoted.columns:
            result = result.with_columns(pl.lit(group_pivoted[col][0]).alias(col))

        return result

    def _add_global_columns(
        self, result: pl.DataFrame, global_data: pl.DataFrame
    ) -> pl.DataFrame:
        """Add global scope columns by broadcasting values"""
        if global_data.is_empty():
            return result
        for row in global_data.iter_rows(named=True):
            col_name = row["label"]
            value = row["value"]
            result = result.with_columns(pl.lit(value).alias(col_name))
        return result

    def _reorder_columns(
        self,
        result: pl.DataFrame,
        index_cols: list[str],
        global_data: pl.DataFrame,
        group_data: pl.DataFrame,
    ) -> pl.DataFrame:
        """Reorder columns: index -> global -> group -> default"""
        if result.is_empty():
            return result

        all_cols = result.columns
        global_cols = []
        group_cols = []
        default_cols = []

        # Get scope information from the original data
        global_labels = []
        if not global_data.is_empty():
            global_labels = global_data.select("label").unique().to_series().to_list()

        group_labels = []
        if not group_data.is_empty():
            group_labels = group_data.select("label").unique().to_series().to_list()

        for col in all_cols:
            if col in index_cols:
                continue
            elif col in global_labels:
                global_cols.append(col)
            elif any(col.endswith(f"_{label}") for label in group_labels):
                group_cols.append(col)
            else:
                default_cols.append(col)

        # Build ordered column list
        ordered_cols = (
            index_cols + sorted(global_cols) + sorted(group_cols) + sorted(default_cols)
        )

        # Only reorder if we have all columns (safety check)
        if len(ordered_cols) == len(all_cols):
            result = result.select(ordered_cols)

        return result

    def evaluate(
        self,
        metrics: MetricDefine | list[MetricDefine] | None = None,
        estimates: str | list[str] | None = None,
        collect: bool = True,
    ) -> pl.DataFrame | pl.LazyFrame:
        """
        Unified evaluation method for all metrics and combinations

        Args:
            metrics: Subset of metrics to evaluate (None = use all configured)
            estimates: Subset of estimates to evaluate (None = use all configured)
            collect: Whether to collect result to DataFrame

        Returns:
            Evaluation results
        """
        # Determine evaluation targets
        target_metrics = self._resolve_metrics(metrics)
        target_estimates = self._resolve_estimates(estimates)

        if not target_metrics or not target_estimates:
            raise ValueError("No metrics or estimates to evaluate")

        # Vectorized evaluation - single Polars operation
        combined = self._vectorized_evaluate(target_metrics, target_estimates)

        # Format final result
        formatted = self._format_result(combined)

        return formatted.collect() if collect else formatted

    def pivot_by_group(
        self,
        metrics: MetricDefine | list[MetricDefine] | None = None,
        estimates: str | list[str] | None = None,
        column_order_by: str = "metrics",
        row_order_by: str = "group",
    ) -> pl.DataFrame:
        """
        Pivot results with groups as rows and model x metric as columns.

        Args:
            metrics: Subset of metrics to evaluate (None = use all configured)
            estimates: Subset of estimates to evaluate (None = use all configured)
            column_order_by: Column ordering strategy ("metrics" or "estimates"). Default: "metrics"
                     - "metrics": Order columns by metric first, then estimate (metric1_model1, metric1_model2, metric2_model1, ...)
                     - "estimates": Order columns by estimate first, then metric (model1_metric1, model1_metric2, model2_metric1, ...)
            row_order_by: Row ordering strategy ("group" or "subgroup"). Default: "group"
                     - "group": Order rows by group columns first, then subgroup
                     - "subgroup": Order rows by subgroup columns first, then group

        Returns:
            DataFrame with group combinations as rows
        """
        long_df = self._get_cached_evaluation(metrics=metrics, estimates=estimates)

        # Build index columns
        index_cols = self._build_index_cols(long_df, include_estimate=False)

        # Separate data by scope
        default_data = long_df.filter(pl.col("scope").is_null())
        group_data = long_df.filter(pl.col("scope") == "group")
        global_data = long_df.filter(pl.col("scope") == "global")

        # Handle each scope separately
        result_parts = []

        # 1. Default scope: model x metric columns
        if not default_data.is_empty():
            # Use multiple columns directly in pivot
            if index_cols:
                default_result = default_data.pivot(
                    on=["estimate", "label"], values="value", index=index_cols
                )
            else:
                default_result = default_data.pivot(
                    on=["estimate", "label"],
                    values="value",
                    index=pl.lit(1).alias("_row"),
                )
                if "_row" in default_result.columns:
                    default_result = default_result.drop("_row")
            result_parts.append(default_result)

        # 2. Group scope: metric label as column
        if not group_data.is_empty():
            if index_cols:
                group_result = group_data.pivot(
                    on="label", values="value", index=index_cols
                )
            else:
                group_result = group_data.pivot(
                    on="label", values="value", index=pl.lit(1).alias("_row")
                )
                if "_row" in group_result.columns:
                    group_result = group_result.drop("_row")
            result_parts.append(group_result)

        # Combine results
        if result_parts:
            result = result_parts[0]
            for part in result_parts[1:]:
                if index_cols:
                    result = result.join(part, on=index_cols, how="full")
                    # Clean up duplicate columns from join
                    for col in index_cols:
                        if f"{col}_right" in result.columns:
                            result = result.drop(f"{col}_right")
                else:
                    # Add columns that don't already exist
                    for col in part.columns:
                        if col not in result.columns:
                            result = result.with_columns(part.select(col))
        else:
            # Create result with proper index structure
            if index_cols:
                # For global scope only, get unique combinations from original data
                available_cols = [
                    col
                    for col in index_cols
                    if col in self.df_raw.collect_schema().names()
                ]
                if available_cols:
                    # Get unique combinations from the original data
                    result = self.df_raw.select(available_cols).unique().collect()
                else:
                    # If no group columns available, create single row
                    result = pl.DataFrame({col: [None] for col in index_cols})
            else:
                result = pl.DataFrame({})

        # Add global columns and reorder
        result = self._add_global_columns(result, global_data)
        result = self._reorder_columns_pivot_by_group(
            result,
            index_cols,
            global_data,
            group_data,
            metrics,
            estimates,
            column_order_by,
            row_order_by,
        )

        # Replace group_by column names with their display labels in final output
        if self.group_by:
            group_rename_mapping = {}
            for col_name, label in self.group_by.items():
                if col_name != label and col_name in result.columns:
                    group_rename_mapping[col_name] = label
            if group_rename_mapping:
                result = result.rename(group_rename_mapping)

        return result

    def _reorder_columns_pivot_by_group(
        self,
        result: pl.DataFrame,
        index_cols: list[str],
        global_data: pl.DataFrame,
        group_data: pl.DataFrame,
        metrics: MetricDefine | list[MetricDefine] | None,
        estimates: str | list[str] | None,
        column_order_by: str,
        row_order_by: str,
    ) -> pl.DataFrame:
        """Reorder columns for pivot_by_group with proper metric/estimate ordering"""
        if result.is_empty():
            return result

        # Validate parameters
        self._validate_evaluation_params(column_order_by, row_order_by)

        # Get the ordered metrics and estimates based on original configuration
        target_metrics = self._resolve_metrics(metrics)
        target_estimates = self._resolve_estimates(estimates)

        # Extract labels in order
        metric_labels = [m.label or m.name for m in target_metrics]
        # Get estimate labels for column ordering (use labels if available, otherwise names)
        estimate_labels = [self.estimates.get(est, est) for est in target_estimates]

        all_cols = result.columns
        global_cols = []
        group_cols = []
        default_cols = []

        # Get scope information from the original data
        global_labels = []
        if not global_data.is_empty():
            global_labels = global_data.select("label").unique().to_series().to_list()

        group_labels = []
        if not group_data.is_empty():
            group_labels = group_data.select("label").unique().to_series().to_list()

        # Separate columns by scope
        for col in all_cols:
            if col in index_cols:
                continue
            # Check if this column contains a global metric label
            elif any(label in col for label in global_labels):
                global_cols.append(col)
            # Check if this column contains a group metric label
            elif any(label in col for label in group_labels):
                group_cols.append(col)
            else:
                default_cols.append(col)

        # Order ALL value columns (default, global, group) based on metric definition order
        # Note: Polars pivot creates column names in JSON format: '{"estimate","label"}'
        ordered_value_cols = []

        # Combine all value columns (non-index columns)
        all_value_cols = global_cols + group_cols + default_cols

        if column_order_by == "metrics":
            # Order by metric first, then estimate: metric1_model1, metric1_model2, metric2_model1, ...
            for metric_label in metric_labels:
                # First add global scope metrics (they don't vary by estimate)
                if (
                    metric_label in global_cols
                    and metric_label not in ordered_value_cols
                ):
                    ordered_value_cols.append(metric_label)
                    continue

                # Then add group scope metrics (they also don't vary by estimate)
                if (
                    metric_label in group_cols
                    and metric_label not in ordered_value_cols
                ):
                    ordered_value_cols.append(metric_label)
                    continue

                # Then add estimate-specific metrics (model/default scope)
                for estimate_label in estimate_labels:
                    # Check for different column name formats
                    possible_col_names = [
                        f'{{"{estimate_label}","{metric_label}"}}',  # JSON format
                        f"{estimate_label}_{metric_label}",  # Simple format
                        f"{metric_label}_{estimate_label}",  # Alternative format
                    ]

                    for col_name in possible_col_names:
                        if (
                            col_name in all_value_cols
                            and col_name not in ordered_value_cols
                        ):
                            ordered_value_cols.append(col_name)
                            break
        else:  # column_order_by == "estimates":
            # Order by estimate first, then metric: model1_metric1, model1_metric2, model2_metric1, ...
            for estimate_label in estimate_labels:
                for metric_label in metric_labels:
                    # Skip global and group scope metrics in this loop - they'll be added at the end
                    if metric_label in global_cols or metric_label in group_cols:
                        continue

                    # Check for different column name formats
                    possible_col_names = [
                        f'{{"{estimate_label}","{metric_label}"}}',  # JSON format
                        f"{estimate_label}_{metric_label}",  # Simple format
                        f"{metric_label}_{estimate_label}",  # Alternative format
                    ]

                    for col_name in possible_col_names:
                        if (
                            col_name in all_value_cols
                            and col_name not in ordered_value_cols
                        ):
                            ordered_value_cols.append(col_name)
                            break

            # Add global and group scope metrics at the end (they don't vary by estimate)
            for metric_label in metric_labels:
                if (
                    metric_label in global_cols or metric_label in group_cols
                ) and metric_label not in ordered_value_cols:
                    ordered_value_cols.append(metric_label)

        # Add any remaining value columns that weren't matched (fallback)
        for col in all_value_cols:
            if col not in ordered_value_cols:
                ordered_value_cols.append(col)

        # Build final column order: index -> ordered value columns
        ordered_cols = index_cols + ordered_value_cols

        # Only reorder if we have all columns (safety check)
        if len(ordered_cols) == len(all_cols):
            result = result.select(ordered_cols)

        # Apply row sorting based on row_order_by parameter
        if row_order_by == "subgroup":
            # Sort by subgroup columns first, then group columns
            sort_cols = []
            if "subgroup_value" in result.columns:
                sort_cols.append("subgroup_value")
            if "subgroup_name" in result.columns:
                sort_cols.append("subgroup_name")
            # Add group columns
            sort_cols.extend(
                [
                    col
                    for col in index_cols
                    if col not in ["subgroup_name", "subgroup_value"]
                ]
            )
        else:  # row_order_by == "group" (default)
            # Sort by group columns first, then subgroup columns
            sort_cols = []
            # Add group columns first
            sort_cols.extend(
                [
                    col
                    for col in index_cols
                    if col not in ["subgroup_name", "subgroup_value"]
                ]
            )
            # Then add subgroup columns
            if "subgroup_name" in result.columns:
                sort_cols.append("subgroup_name")
            if "subgroup_value" in result.columns:
                sort_cols.append("subgroup_value")

        # Convert group and subgroup columns to enums first so sorting respects enum order
        result = self._convert_pivot_columns_to_enums(result)

        if sort_cols:
            result = result.sort(sort_cols)

        return result

    def pivot_by_model(
        self,
        metrics: MetricDefine | list[MetricDefine] | None = None,
        estimates: str | list[str] | None = None,
        column_order_by: str = "metrics",
        row_order_by: str = "group",
    ) -> pl.DataFrame:
        """
        Pivot results with models as rows and group x metric as columns.

        Args:
            metrics: Subset of metrics to evaluate (None = use all configured)
            estimates: Subset of estimates to evaluate (None = use all configured)
            column_order_by: Column ordering strategy - "metrics" (default) or "groups"
                     "metrics": Order by metric definition order first, then groups
                     "groups": Order by group combinations first, then metrics
            row_order_by: Row ordering strategy - "group" (default) or "subgroup"
                     "group": Order rows by group columns first, then subgroup
                     "subgroup": Order rows by subgroup columns first, then group

        Returns:
            DataFrame with models as rows
        """
        long_df = self._get_cached_evaluation(metrics=metrics, estimates=estimates)

        # Build index columns
        index_cols = self._build_index_cols(long_df, include_estimate=True)

        # Separate data by scope
        model_default_data = long_df.filter(
            (pl.col("scope").is_in(["model", "default"])) | (pl.col("scope").is_null())
        )
        global_data = long_df.filter(pl.col("scope") == "global")
        group_data = long_df.filter(pl.col("scope") == "group")

        # Create pivot result from model/default data
        if not model_default_data.is_empty():
            # Use metric labels instead of metric names for pivot column headers
            # Use multiple columns directly in pivot
            if self.group_by:
                pivot_on_cols = list(self.group_by.keys()) + ["label"]
            else:
                # Add a constant column for pivoting when no group_by
                model_default_data = model_default_data.with_columns(
                    pl.lit("ALL").alias("_group")
                )
                pivot_on_cols = ["_group", "label"]

            result = model_default_data.pivot(
                on=pivot_on_cols, values="value", index=index_cols
            )
        else:
            # Create empty result with index columns
            result = pl.DataFrame().with_columns(
                [pl.lit(None, dtype=pl.Utf8).alias(col) for col in index_cols]
            )

        # Add global and group columns, then reorder
        result = self._add_global_columns(result, global_data)
        result = self._add_group_columns(result, group_data)
        result = self._reorder_columns_pivot_by_model(
            result,
            index_cols,
            global_data,
            group_data,
            metrics,
            column_order_by,
            row_order_by,
        )

        # Replace group_by column names with their display labels in final output
        if self.group_by:
            group_rename_mapping = {}
            for col_name, label in self.group_by.items():
                if col_name != label and col_name in result.columns:
                    group_rename_mapping[col_name] = label
            if group_rename_mapping:
                result = result.rename(group_rename_mapping)

        return result

    def _reorder_columns_pivot_by_model(
        self,
        result: pl.DataFrame,
        index_cols: list[str],
        global_data: pl.DataFrame,
        group_data: pl.DataFrame,
        metrics: MetricDefine | list[MetricDefine] | None,
        column_order_by: str,
        row_order_by: str,
    ) -> pl.DataFrame:
        """Reorder columns for pivot_by_model with proper metric/group ordering"""
        if result.is_empty():
            return result

        # Validate column_order_by parameter
        if column_order_by not in ["metrics", "groups"]:
            raise ValueError(
                f"column_order_by must be 'metrics' or 'groups', got '{column_order_by}'"
            )

        # Validate row_order_by parameter
        if row_order_by not in ["group", "subgroup"]:
            raise ValueError(
                f"row_order_by must be 'group' or 'subgroup', got '{row_order_by}'"
            )

        # Get the ordered metrics based on original configuration
        target_metrics = self._resolve_metrics(metrics)

        # Extract names and labels for ordering
        # Use labels for column matching (since columns now contain metric labels)
        # Use labels for scope detection (since scope data contains metric labels)
        metric_names = [m.name for m in target_metrics]
        metric_labels = [m.label or m.name for m in target_metrics]

        all_cols = result.columns
        global_cols = []
        group_cols = []
        default_cols = []

        # Get scope information from the original data
        global_labels = []
        if not global_data.is_empty():
            global_labels = global_data.select("label").unique().to_series().to_list()

        group_labels = []
        if not group_data.is_empty():
            group_labels = group_data.select("label").unique().to_series().to_list()

        # Separate columns by scope
        for col in all_cols:
            if col in index_cols:
                continue
            # Check if this column contains a global metric label
            elif any(label in col for label in global_labels):
                global_cols.append(col)
            # Check if this column contains a group metric label
            elif any(label in col for label in group_labels):
                group_cols.append(col)
            else:
                default_cols.append(col)

        # Order ALL value columns (default, global, group) based on metric definition order
        # Note: pivot_by_model creates columns like '{"group1","group2","metric"}' format
        ordered_value_cols = []

        # Combine all value columns (non-index columns)
        all_value_cols = global_cols + group_cols + default_cols

        # Get unique group combinations for ordering
        group_combinations = set()
        for col in default_cols:
            # Extract group combinations from JSON format columns
            # Format: '{"group1","group2","metric"}'
            if col.startswith("{") and col.endswith("}"):
                # Parse the JSON-like format to extract group parts
                # Remove outer braces and split by quotes and commas
                inner = col[1:-1]  # Remove { and }
                parts = [part.strip('"') for part in inner.split('","')]
                if len(parts) > 1:
                    # Get all parts except the last (which is the metric label)
                    group_parts = parts[:-1]
                    # Reconstruct the group combination in the same format
                    group_combo = '","'.join(group_parts)
                    group_combinations.add(group_combo)

        group_combinations = sorted(list(group_combinations))

        if column_order_by == "metrics":
            # Order by metric first, then groups: metric1_group1, metric1_group2, metric2_group1, ...
            for i, metric_label in enumerate(metric_labels):
                # First add global scope metrics (they don't vary by group)
                if (
                    metric_label in global_cols
                    and metric_label not in ordered_value_cols
                ):
                    ordered_value_cols.append(metric_label)
                    continue

                # Then add group scope metrics (they also don't vary by group combinations)
                if (
                    metric_label in group_cols
                    and metric_label not in ordered_value_cols
                ):
                    ordered_value_cols.append(metric_label)
                    continue

                # Then add group-specific metrics (model/default scope) for this metric across all groups
                for group_combo in group_combinations:
                    # Check for different column name formats using metric labels
                    possible_col_names = [
                        f'{{"{group_combo}","{metric_label}"}}',  # JSON format with metric label
                        f"{group_combo}_{metric_label}",  # Simple format with metric label
                        f"{metric_label}_{group_combo}",  # Alternative format with metric label
                    ]

                    for col_name in possible_col_names:
                        if (
                            col_name in all_value_cols
                            and col_name not in ordered_value_cols
                        ):
                            ordered_value_cols.append(col_name)
                            break
        else:  # column_order_by == "groups"
            # Order by groups first, then metric: group1_metric1, group1_metric2, group2_metric1, ...
            for group_combo in group_combinations:
                for i, metric_label in enumerate(metric_labels):
                    # Skip global and group scope metrics in this loop - they'll be added at the end
                    if metric_label in global_cols or metric_label in group_cols:
                        continue

                    # Check for different column name formats using metric labels
                    possible_col_names = [
                        f'{{"{group_combo}","{metric_label}"}}',  # JSON format with metric label
                        f"{group_combo}_{metric_label}",  # Simple format with metric label
                        f"{metric_label}_{group_combo}",  # Alternative format with metric label
                    ]

                    for col_name in possible_col_names:
                        if (
                            col_name in all_value_cols
                            and col_name not in ordered_value_cols
                        ):
                            ordered_value_cols.append(col_name)
                            break

            # Add global and group scope metrics at the end (they don't vary by group combinations)
            for metric_label in metric_labels:
                if (
                    metric_label in global_cols or metric_label in group_cols
                ) and metric_label not in ordered_value_cols:
                    ordered_value_cols.append(metric_label)

        # Add any remaining value columns that weren't matched (fallback)
        for col in all_value_cols:
            if col not in ordered_value_cols:
                ordered_value_cols.append(col)

        # Build final column order: index -> ordered value columns
        ordered_cols = index_cols + ordered_value_cols

        # Only reorder if we have all columns (safety check)
        if len(ordered_cols) == len(all_cols):
            result = result.select(ordered_cols)

        # Apply row sorting based on row_order_by parameter
        if row_order_by == "subgroup":
            # Sort by subgroup columns first, then group columns
            sort_cols = []
            if "subgroup_value" in result.columns:
                sort_cols.append("subgroup_value")
            if "subgroup_name" in result.columns:
                sort_cols.append("subgroup_name")
            # Add group columns (excluding estimate and subgroup columns)
            sort_cols.extend(
                [
                    col
                    for col in index_cols
                    if col not in ["estimate", "subgroup_name", "subgroup_value"]
                ]
            )
            # Finally add estimate
            if "estimate" in result.columns:
                sort_cols.append("estimate")
        else:  # row_order_by == "group" (default)
            # Sort by group columns first, then subgroup columns, then estimate
            sort_cols = []
            # Add group columns first (excluding estimate and subgroup columns)
            sort_cols.extend(
                [
                    col
                    for col in index_cols
                    if col not in ["estimate", "subgroup_name", "subgroup_value"]
                ]
            )
            # Then add subgroup columns
            if "subgroup_name" in result.columns:
                sort_cols.append("subgroup_name")
            if "subgroup_value" in result.columns:
                sort_cols.append("subgroup_value")
            # Finally add estimate
            if "estimate" in result.columns:
                sort_cols.append("estimate")

        # Convert group and subgroup columns to enums first so sorting respects enum order
        result = self._convert_pivot_columns_to_enums(result)

        if sort_cols:
            result = result.sort(sort_cols)

        return result

    def _convert_pivot_columns_to_enums(self, result: pl.DataFrame) -> pl.DataFrame:
        """Convert group and subgroup columns to enums, preserving original enum ordering when available"""

        # Convert group columns to enums based on their labels
        for col_name, label in self.group_by.items():
            target_col = None
            if col_name in result.columns:
                target_col = col_name
            elif label in result.columns and label != col_name:
                target_col = label

            if target_col:
                # Check if original data had enum type for this column and preserve its ordering
                original_enum_categories = self._get_original_enum_categories(col_name)
                if original_enum_categories is not None:
                    # Use original enum categories, filtered to only include values present in result
                    result_values = set(
                        result.get_column(target_col).unique().to_list()
                    )
                    filtered_categories = [
                        cat for cat in original_enum_categories if cat in result_values
                    ]
                    if len(filtered_categories) > 1:
                        enum_type = pl.Enum(filtered_categories)
                        result = result.with_columns(pl.col(target_col).cast(enum_type))
                else:
                    # Fallback to display order if no original enum found
                    unique_values = (
                        result.get_column(target_col)
                        .unique(maintain_order=True)
                        .to_list()
                    )
                    if len(unique_values) > 1:
                        enum_type = pl.Enum(unique_values)
                        result = result.with_columns(pl.col(target_col).cast(enum_type))

        # Convert subgroup columns to enums
        if "subgroup_name" in result.columns:
            unique_names = (
                result.get_column("subgroup_name").unique(maintain_order=True).to_list()
            )
            if len(unique_names) > 1:
                enum_type = pl.Enum(unique_names)
                result = result.with_columns(pl.col("subgroup_name").cast(enum_type))

        # Convert subgroup_value to enum (values from all subgroups combined)
        if "subgroup_value" in result.columns:
            # Get all unique values actually present in the result
            result_values = (
                result.get_column("subgroup_value")
                .unique(maintain_order=True)
                .to_list()
            )

            # Try to build an ordered list from original enum categories for each subgroup
            ordered_categories = []
            remaining_values = set(result_values)

            # First, add values from enum subgroups in their original order
            for col_name, label in self.subgroup_by.items():
                original_enum_categories = self._get_original_enum_categories(col_name)
                if original_enum_categories is not None:
                    # Add enum categories that are present in result, maintaining original order
                    for cat in original_enum_categories:
                        if cat in remaining_values:
                            ordered_categories.append(cat)
                            remaining_values.remove(cat)

            # Then add any remaining values (from non-enum subgroups) in their display order
            remaining_ordered = [
                val for val in result_values if val in remaining_values
            ]
            ordered_categories.extend(remaining_ordered)

            # Create enum with the combined ordered categories (convert to strings)
            if len(ordered_categories) > 1:
                string_categories = [str(cat) for cat in ordered_categories]
                enum_type = pl.Enum(string_categories)
                result = result.with_columns(
                    pl.col("subgroup_value").cast(pl.Utf8).cast(enum_type)
                )

        return result

    def _get_original_enum_categories(self, col_name: str) -> list[str] | None:
        """Get original enum categories from the source data if the column was an enum"""
        try:
            # Check if the column exists in the original dataframe
            if col_name in self.df_raw.collect_schema().names():
                col_dtype = self.df_raw.collect_schema()[col_name]
                if isinstance(col_dtype, pl.Enum):
                    return col_dtype.categories.to_list()
        except Exception:
            pass
        return None

    def _resolve_metrics(
        self, metrics: MetricDefine | list[MetricDefine] | None
    ) -> list[MetricDefine]:
        """Resolve which metrics to evaluate"""
        if metrics is None:
            return self.metrics

        metrics_list = [metrics] if isinstance(metrics, MetricDefine) else metrics
        configured_names = {m.name for m in self.metrics}

        for m in metrics_list:
            if m.name not in configured_names:
                raise ValueError(f"Metric '{m.name}' not in configured metrics")

        return metrics_list

    def _resolve_estimates(self, estimates: str | list[str] | None) -> list[str]:
        """Resolve which estimates to evaluate"""
        if estimates is None:
            return list(self.estimates.keys())

        estimates_list = [estimates] if isinstance(estimates, str) else estimates

        for e in estimates_list:
            if e not in self.estimates:
                raise ValueError(
                    f"Estimate '{e}' not in configured estimates: {list(self.estimates.keys())}"
                )

        return estimates_list

    def _vectorized_evaluate(
        self, metrics: list[MetricDefine], estimates: list[str]
    ) -> pl.LazyFrame:
        """Vectorized evaluation using single Polars group_by operations"""

        # Step 1: Prepare data in long format with all estimates
        df_long = self._prepare_long_format_data(estimates)

        # Step 2: Generate all error columns for the melted data
        df_with_errors = self._add_error_columns_vectorized(df_long)

        # Step 3: Handle marginal subgroup analysis if needed
        if self.subgroup_by:
            return self._evaluate_with_marginal_subgroups(
                df_with_errors, metrics, estimates
            )
        else:
            return self._evaluate_without_subgroups(df_with_errors, metrics, estimates)

    def _evaluate_without_subgroups(
        self,
        df_with_errors: pl.LazyFrame,
        metrics: list[MetricDefine],
        estimates: list[str],
    ) -> pl.LazyFrame:
        """Evaluate metrics without subgroup analysis"""
        results = []
        for metric in metrics:
            metric_result = self._evaluate_metric_vectorized(
                df_with_errors, metric, estimates
            )
            results.append(metric_result)

        # Harmonize schemas before combining
        if results:
            harmonized_results = self._harmonize_result_schemas(results)
            return pl.concat(harmonized_results, how="diagonal")
        else:
            return pl.DataFrame().lazy()

    def _evaluate_with_marginal_subgroups(
        self,
        df_with_errors: pl.LazyFrame,
        metrics: list[MetricDefine],
        estimates: list[str],
    ) -> pl.LazyFrame:
        """Evaluate metrics with marginal subgroup analysis using vectorized operations"""
        original_subgroup_by = self.subgroup_by

        # Create all subgroup combinations using vectorized unpivot
        subgroup_data = self._prepare_subgroup_data_vectorized(
            df_with_errors, original_subgroup_by
        )

        # Evaluate all metrics across all subgroups in a vectorized manner
        results = []
        for metric in metrics:
            # Temporarily clear subgroup_by to use the vectorized subgroup columns instead
            self.subgroup_by = {}
            try:
                metric_result = self._evaluate_metric_vectorized(
                    subgroup_data, metric, estimates
                )
                results.append(metric_result)
            finally:
                self.subgroup_by = original_subgroup_by

        # Combine results
        harmonized_results = self._harmonize_result_schemas(results)
        return pl.concat(harmonized_results, how="diagonal")

    def _prepare_subgroup_data_vectorized(
        self, df_with_errors: pl.LazyFrame, subgroup_by: dict[str, str]
    ) -> pl.LazyFrame:
        """Prepare subgroup data using vectorized unpivot operations"""
        schema_names = df_with_errors.collect_schema().names()
        subgroup_cols = list(subgroup_by.keys())
        id_vars = [col for col in schema_names if col not in subgroup_cols]

        # Use unpivot to create marginal subgroup analysis
        return df_with_errors.unpivot(
            index=id_vars,
            on=subgroup_cols,
            variable_name="subgroup_name",
            value_name="subgroup_value",
        ).with_columns(
            [
                # Replace subgroup column names with their display labels
                pl.col("subgroup_name").replace(subgroup_by)
            ]
        )

    def _prepare_long_format_data(self, estimates: list[str]) -> pl.LazyFrame:
        """Reshape data from wide to long format for vectorized processing"""

        # Add a row index to the original data to uniquely identify each sample
        # This must be done BEFORE unpivoting to avoid double counting
        df_with_index = self.df.with_row_index("sample_index")

        # Get all columns except estimates to preserve in melt
        schema_names = df_with_index.collect_schema().names()
        id_vars = [col for col in schema_names if col not in estimates]

        # Unpivot estimates into long format
        df_long = df_with_index.unpivot(
            index=id_vars,
            on=estimates,
            variable_name="estimate_name",
            value_name="estimate_value",
        )

        # Replace estimate names with their display labels and rename columns
        df_long = df_long.with_columns(
            [pl.col("estimate_name").replace(self.estimates).alias("estimate")]
        ).rename({self.ground_truth: "ground_truth"})

        return df_long

    def _add_error_columns_vectorized(self, df_long: pl.LazyFrame) -> pl.LazyFrame:
        """Add error columns for the long-format data"""

        # Generate error expressions for the vectorized format
        # Use 'estimate_value' as the estimate column and 'ground_truth' as the renamed ground truth column
        error_expressions = MetricRegistry.generate_error_columns(
            estimate="estimate_value",
            ground_truth="ground_truth",
            error_types=None,
            error_params=self.error_params,
        )

        return df_long.with_columns(error_expressions)

    def _evaluate_metric_vectorized(
        self, df_with_errors: pl.LazyFrame, metric: MetricDefine, estimates: list[str]
    ) -> pl.LazyFrame:
        """Evaluate a single metric using vectorized operations"""

        # Determine grouping columns based on metric scope
        group_cols = self._get_vectorized_grouping_columns(metric, df_with_errors)

        # Compile metric expressions
        within_exprs, across_expr = metric.compile_expressions()

        # Apply metric-specific filtering if needed
        df_filtered = self._apply_metric_scope_filter(df_with_errors, metric, estimates)

        # Perform the evaluation with appropriate grouping
        if metric.type == MetricType.ACROSS_SAMPLE:
            if across_expr is None:
                raise ValueError(
                    f"ACROSS_SAMPLE metric {metric.name} requires across_expr"
                )

            if group_cols:
                result = df_filtered.group_by(group_cols).agg(
                    across_expr.alias("value").cast(pl.Float64)
                )
            else:
                result = df_filtered.select(across_expr.alias("value").cast(pl.Float64))

        elif metric.type in [MetricType.WITHIN_SUBJECT, MetricType.WITHIN_VISIT]:
            # Within-entity aggregation
            entity_groups = self._get_entity_grouping_columns(metric.type) + group_cols

            # Use the appropriate expression for within-entity aggregation
            if within_exprs:
                agg_expr = within_exprs[0].alias("value").cast(pl.Float64)
            elif across_expr is not None:
                agg_expr = across_expr.alias("value").cast(pl.Float64)
            else:
                raise ValueError(f"No valid expression for metric {metric.name}")

            result = df_filtered.group_by(entity_groups).agg(agg_expr)

        elif metric.type in [MetricType.ACROSS_SUBJECT, MetricType.ACROSS_VISIT]:
            # Two-level aggregation: within entities, then across
            entity_groups = self._get_entity_grouping_columns(metric.type) + group_cols

            # First level: within entities
            if within_exprs:
                first_level_expr = within_exprs[0].alias("value")
            elif across_expr is not None:
                first_level_expr = across_expr.alias("value")
            else:
                raise ValueError(
                    f"No valid expression for first level of metric {metric.name}"
                )

            intermediate = df_filtered.group_by(entity_groups).agg(first_level_expr)

            # Second level: across entities
            if across_expr is not None and within_exprs:
                # True two-level case - use the across_expr directly since we named the column 'value'
                second_level_expr = across_expr.alias("value").cast(pl.Float64)
            else:
                # Default aggregation across entities when no across_expr specified
                second_level_expr = (
                    pl.col("value").mean().alias("value").cast(pl.Float64)
                )

            if group_cols:
                result = intermediate.group_by(group_cols).agg(second_level_expr)
            else:
                result = intermediate.select(second_level_expr)

        else:
            raise ValueError(f"Unknown metric type: {metric.type}")

        # Add metadata columns
        return self._add_metadata_vectorized(result, metric)

    def _get_vectorized_grouping_columns(
        self, metric: MetricDefine, df: pl.LazyFrame | None = None
    ) -> list[str]:
        """Get grouping columns for vectorized evaluation based on metric scope"""
        group_cols = []

        # Check if we're in vectorized subgroup mode (data has subgroup_name/subgroup_value columns)
        if df is not None:
            schema_names = df.collect_schema().names()
            using_vectorized_subgroups = (
                "subgroup_name" in schema_names and "subgroup_value" in schema_names
            )
        else:
            using_vectorized_subgroups = False

        # Handle scope-based grouping
        if metric.scope == MetricScope.GLOBAL:
            if using_vectorized_subgroups:
                group_cols.extend(["subgroup_name", "subgroup_value"])
            else:
                group_cols.extend(list(self.subgroup_by.keys()))

        elif metric.scope == MetricScope.MODEL:
            group_cols.append("estimate")
            if using_vectorized_subgroups:
                group_cols.extend(["subgroup_name", "subgroup_value"])
            else:
                group_cols.extend(list(self.subgroup_by.keys()))

        elif metric.scope == MetricScope.GROUP:
            group_cols.extend(list(self.group_by.keys()))
            if using_vectorized_subgroups:
                group_cols.extend(["subgroup_name", "subgroup_value"])
            else:
                group_cols.extend(list(self.subgroup_by.keys()))

        else:
            # Default: estimate + groups + subgroups
            group_cols.append("estimate")
            group_cols.extend(list(self.group_by.keys()))
            if using_vectorized_subgroups:
                group_cols.extend(["subgroup_name", "subgroup_value"])
            else:
                group_cols.extend(list(self.subgroup_by.keys()))

        return group_cols

    def _apply_metric_scope_filter(
        self, df: pl.LazyFrame, metric: MetricDefine, estimates: list[str]
    ) -> pl.LazyFrame:
        """Apply any scope-specific filtering"""
        # For now, no additional filtering needed beyond grouping
        # Future: could add estimate filtering for specific scopes
        _ = metric, estimates  # Suppress unused parameter warnings
        return df

    def _get_entity_grouping_columns(self, metric_type: MetricType) -> list[str]:
        """Get entity-level grouping columns (subject_id, visit_id)"""
        if metric_type in [MetricType.WITHIN_SUBJECT, MetricType.ACROSS_SUBJECT]:
            return ["subject_id"]
        elif metric_type in [MetricType.WITHIN_VISIT, MetricType.ACROSS_VISIT]:
            return ["subject_id", "visit_id"]
        else:
            return []

    def _add_metadata_vectorized(
        self, result: pl.LazyFrame, metric: MetricDefine
    ) -> pl.LazyFrame:
        """Add metadata columns to vectorized result"""

        metadata = [
            pl.lit(metric.name).alias("metric"),
            pl.lit(metric.label).alias("label"),
            pl.lit(metric.type.value).alias("metric_type"),
            pl.lit(metric.scope.value if metric.scope else None).alias("scope"),
        ]

        # Add metadata columns
        result_with_metadata = result.with_columns(metadata)

        return result_with_metadata

    def _format_result(self, combined: pl.LazyFrame) -> pl.LazyFrame:
        """Format final result with proper column ordering and sorting"""

        # Determine available columns
        try:
            available_columns = combined.collect_schema().names()
        except Exception:
            available_columns = combined.limit(1).collect().columns

        # Extract ordered values for enum columns from definitions
        ordered_metrics = []
        ordered_labels = []
        for metric in self.metrics:
            if metric.name not in ordered_metrics:
                ordered_metrics.append(metric.name)
            label = metric.label
            if label not in ordered_labels:
                ordered_labels.append(label)

        # Convert metric column to enum with definition-based ordering
        if "metric" in available_columns and ordered_metrics:
            metric_enum = pl.Enum(ordered_metrics)
            combined = combined.with_columns(pl.col("metric").cast(metric_enum))

        # Convert label column to enum with definition-based ordering
        if "label" in available_columns and ordered_labels:
            label_enum = pl.Enum(ordered_labels)
            combined = combined.with_columns(pl.col("label").cast(label_enum))

        # Convert estimate column to enum with input-based ordering
        if "estimate" in available_columns and self.estimates:
            # Use estimate labels for enum ordering, maintaining original estimate order
            estimate_labels_ordered = [
                self.estimates.get(est, est) for est in self.estimates.keys()
            ]
            estimate_enum = pl.Enum(estimate_labels_ordered)
            combined = combined.with_columns(pl.col("estimate").cast(estimate_enum))

        # Define column order
        column_order = []

        # ID columns
        for col in ["subject_id", "visit_id"]:
            if col in available_columns:
                column_order.append(col)

        # Group columns
        for col in self.group_by.keys():
            if col in available_columns and col not in column_order:
                column_order.append(col)

        # Subgroup columns
        if "subgroup_name" in available_columns:
            column_order.extend(["subgroup_name", "subgroup_value"])

        # Core result columns - only include estimate if it exists
        core_columns = []
        if "estimate" in available_columns:
            core_columns.append("estimate")
        core_columns.extend(["metric", "label", "value", "metric_type", "scope"])
        column_order.extend(core_columns)

        # Sort columns - prioritize subgroup_value first, then other columns
        sort_cols = []

        # Build sort order: subgroup_value first (when present), then others
        if "subgroup_value" in available_columns:
            sort_cols.append("subgroup_value")
        if "subgroup_name" in available_columns:
            sort_cols.append("subgroup_name")

        # Add group columns
        for col in self.group_by.keys():
            if col in available_columns and col not in sort_cols:
                sort_cols.append(col)

        # Add other columns
        other_cols = ["label"]
        if "estimate" in available_columns:
            other_cols.append("estimate")

        for col in other_cols:
            if col in available_columns and col not in sort_cols:
                sort_cols.append(col)

        # Apply sorting - subgroup_value first priority
        result = combined
        if sort_cols:
            result = result.sort(sort_cols)

        # Select columns in order (only those that exist)
        existing_order = [col for col in column_order if col in available_columns]
        result = result.select(existing_order)

        return result

    def _harmonize_result_schemas(
        self, results: list[pl.LazyFrame]
    ) -> list[pl.LazyFrame]:
        """
        Harmonize schemas of result DataFrames to ensure they can be concatenated.

        This addresses the issue where different scopes produce different column structures
        (e.g., some have 'estimate' column, others don't; some have string 'scope', others null).
        """
        if not results:
            return results

        # Collect all unique columns across all results
        all_columns = set()
        schemas = []

        for result in results:
            try:
                schema = result.collect_schema()
                schemas.append(schema)
                all_columns.update(schema.names())
            except Exception:
                # Fallback: collect a small sample to get schema
                sample = result.limit(1).collect()
                schema = sample.schema
                schemas.append(schema)
                all_columns.update(schema.names())

        # Define the target column order and types
        target_columns = []

        # Standard columns that should always be present
        standard_cols = [
            "subject_id",
            "visit_id",  # Entity columns
        ]
        # Add dynamic group_by columns
        standard_cols.extend(list(self.group_by.keys()))
        # Add remaining standard columns
        standard_cols.extend(
            [
                "estimate",  # Model column (may be missing for some scopes)
                "metric",
                "label",
                "value",
                "metric_type",
                "scope",  # Core metric columns
                "subgroup_name",
                "subgroup_value",  # Subgroup columns (may be missing)
            ]
        )

        # Add columns in preferred order
        for col in standard_cols:
            if col in all_columns:
                target_columns.append(col)

        # Add any remaining columns (group_by columns that aren't standard)
        for col in sorted(all_columns):
            if col not in target_columns:
                target_columns.append(col)

        # Harmonize each result
        harmonized = []
        for i, result in enumerate(results):
            schema = schemas[i]

            # Add missing columns with appropriate null values
            exprs = []
            for col in target_columns:
                if col in schema.names():
                    # Handle scope column type conversion
                    if col == "scope" and schema[col] == pl.Null:
                        exprs.append(pl.col(col).cast(pl.Utf8))
                    else:
                        exprs.append(pl.col(col))
                else:
                    # Add missing column with null value of appropriate type
                    if col == "estimate":
                        exprs.append(pl.lit(None, dtype=pl.Utf8).alias(col))
                    elif col == "scope":
                        exprs.append(pl.lit(None, dtype=pl.Utf8).alias(col))
                    elif col in ["subject_id", "visit_id"]:
                        exprs.append(pl.lit(None, dtype=pl.Int64).alias(col))
                    elif col == "value":
                        exprs.append(pl.lit(None, dtype=pl.Float64).alias(col))
                    else:
                        exprs.append(pl.lit(None, dtype=pl.Utf8).alias(col))

            harmonized_result = result.select(exprs)
            harmonized.append(harmonized_result)

        return harmonized

    # ========================================
    # INPUT PROCESSING METHODS - Pure Logic
    # ========================================

    @staticmethod
    def _process_estimates(
        estimates: str | list[str] | dict[str, str] | None,
    ) -> dict[str, str]:
        """Pure transformation: normalize estimates to dict format"""
        if isinstance(estimates, str):
            return {estimates: estimates}
        elif isinstance(estimates, dict):
            return estimates
        elif isinstance(estimates, list):
            return {est: est for est in (estimates or [])}
        else:
            return {}

    @staticmethod
    def _process_grouping(
        grouping: list[str] | dict[str, str] | None,
    ) -> dict[str, str]:
        """Pure transformation: normalize grouping to dict format"""
        if isinstance(grouping, dict):
            return grouping
        elif isinstance(grouping, list):
            return {col: col for col in (grouping or [])}
        else:
            return {}

    # ========================================
    # VALIDATION METHODS - Centralized Logic
    # ========================================

    def _validate_inputs(self) -> None:
        """Validate all inputs after processing"""
        if not self.estimates:
            raise ValueError("No estimates provided")

        if not self.metrics:
            raise ValueError("No metrics provided")

        # Validate that required columns exist
        schema_names = self.df_raw.collect_schema().names()

        if self.ground_truth not in schema_names:
            raise ValueError(
                f"Ground truth column '{self.ground_truth}' not found in data"
            )

        missing_estimates = [
            est for est in self.estimates.keys() if est not in schema_names
        ]
        if missing_estimates:
            raise ValueError(f"Estimate columns not found in data: {missing_estimates}")

        missing_groups = [
            col for col in self.group_by.keys() if col not in schema_names
        ]
        if missing_groups:
            raise ValueError(f"Group columns not found in data: {missing_groups}")

        missing_subgroups = [
            col for col in self.subgroup_by.keys() if col not in schema_names
        ]
        if missing_subgroups:
            raise ValueError(f"Subgroup columns not found in data: {missing_subgroups}")

    @staticmethod
    def _validate_evaluation_params(column_order_by: str, row_order_by: str) -> None:
        """Validate evaluation parameters"""
        if column_order_by not in ["metrics", "estimates"]:
            raise ValueError(
                f"column_order_by must be 'metrics' or 'estimates', got '{column_order_by}'"
            )

        if row_order_by not in ["group", "subgroup"]:
            raise ValueError(
                f"row_order_by must be 'group' or 'subgroup', got '{row_order_by}'"
            )
