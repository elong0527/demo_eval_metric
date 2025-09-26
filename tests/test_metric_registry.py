import polars as pl
import pytest

from polars_eval_metrics import (
    MetricDefine,
    MetricEvaluator,
    MetricRegistry,
    MetricScope,
)
from polars_eval_metrics.ard import ARD
from polars_eval_metrics.metric_registry import MetricInfo


def test_metric_registry_registers_custom_entries() -> None:
    """Custom registry entries should be discoverable through helper APIs."""
    error_name = "test_registry_offset_error"
    metric_name = "test_registry_offset_metric"
    summary_name = "test_registry_offset_summary"

    MetricRegistry.register_error(
        error_name,
        lambda estimate, ground_truth: (pl.col(estimate) - pl.col(ground_truth)).abs(),
    )
    MetricRegistry.register_metric(
        metric_name,
        lambda: pl.col(error_name).max().alias("value"),
    )
    MetricRegistry.register_summary(
        summary_name,
        pl.col("value").median(),
    )

    assert MetricRegistry.has_error(error_name)
    assert metric_name in MetricRegistry.list_metrics()
    assert summary_name in MetricRegistry.list_summaries()

    metric_info = MetricRegistry.get_metric(metric_name)
    summary_expr = MetricRegistry.get_summary(summary_name)

    assert isinstance(metric_info, MetricInfo)
    assert isinstance(metric_info.expr, pl.Expr)
    assert isinstance(summary_expr, pl.Expr)


def test_metric_registry_evaluator_integration(metric_sample_df: pl.DataFrame) -> None:
    """Registered metrics should work end-to-end within MetricEvaluator."""
    error_name = "test_registry_bias_error"
    metric_name = "test_registry_bias_metric"

    MetricRegistry.register_error(
        error_name,
        lambda estimate, ground_truth: pl.col(estimate) - pl.col(ground_truth),
    )
    MetricRegistry.register_metric(
        metric_name,
        lambda: pl.col(error_name).mean().alias("value"),
    )

    metrics = [
        MetricDefine(
            name=metric_name,
            scope=MetricScope.MODEL,
            label="Mean Bias",
        )
    ]

    evaluator = MetricEvaluator(
        df=metric_sample_df,
        metrics=metrics,
        ground_truth="actual",
        estimates={"model_a": "Model A", "model_b": "Model B"},
    )

    compact = evaluator.evaluate()
    stats = ARD(evaluator.evaluate(collect=False)).get_stats()
    detailed = evaluator.evaluate(verbose=True).with_columns(
        pl.col("stat").struct.field("value_float").alias("_value_float")
    )

    expected_bias = (
        metric_sample_df.lazy()
        .select(
            [
                (pl.col("model_a") - pl.col("actual")).alias("model_a_bias"),
                (pl.col("model_b") - pl.col("actual")).alias("model_b_bias"),
            ]
        )
        .collect()
    )
    expected_a = expected_bias["model_a_bias"].drop_nulls().mean()
    expected_b = expected_bias["model_b_bias"].drop_nulls().mean()

    actual_a = detailed.filter(pl.col("estimate") == "model_a").filter(
        pl.col("metric") == metric_name
    )["_value_float"][0]
    actual_b = detailed.filter(pl.col("estimate") == "model_b").filter(
        pl.col("metric") == metric_name
    )["_value_float"][0]

    assert actual_a == pytest.approx(expected_a)
    assert actual_b == pytest.approx(expected_b)

    context_scope = detailed.select(
        pl.col("context").struct.field("scope").alias("scope")
    )
    assert set(context_scope["scope"].drop_nulls().to_list()) == {"model"}


def test_struct_metric_output(metric_sample_df: pl.DataFrame) -> None:
    """Custom metric can surface structured payloads directly in the stat struct."""

    metric_name = "mae_with_bounds"

    MetricRegistry.register_metric(
        metric_name,
        MetricInfo(
            expr=pl.struct(
                [
                    pl.col("absolute_error").mean().alias("mean"),
                    (pl.col("absolute_error").mean() - 0.5).alias("lower"),
                    (pl.col("absolute_error").mean() + 0.5).alias("upper"),
                ]
            ),
            value_kind="struct",
        ),
    )

    evaluator = MetricEvaluator(
        df=metric_sample_df,
        metrics=[MetricDefine(name=metric_name)],
        ground_truth="actual",
        estimates={"model_a": "Model A"},
    )

    verbose_result = evaluator.evaluate(verbose=True)
    stat = verbose_result["stat"][0]

    assert stat["type"] == "struct"
    payload = stat["value_struct"]
    assert isinstance(payload, dict)
    assert set(payload.keys()) == {"mean", "lower", "upper"}
    assert payload["lower"] < payload["upper"]
