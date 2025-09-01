"""Basic tests to verify the package works."""

import sys
from pathlib import Path

# Add examples to path for data generator
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

import polars as pl
import pytest

from polars_eval_metrics import (
    EvaluationConfig,
    MetricCompiler,
    MetricData,
    MetricEvaluator,
    MetricFactory,
    MetricType,
    SharedType,
)
from data_generator import generate_sample_data


def test_metric_data_creation():
    """Test creating a metric data object."""
    metric = MetricData(
        name="mae",
        label="Mean Absolute Error",
        type=MetricType.ACROSS_SAMPLES,
    )
    assert metric.name == "mae"
    assert metric.label == "Mean Absolute Error"
    assert metric.type == MetricType.ACROSS_SAMPLES


def test_metric_factory():
    """Test creating metrics from factory."""
    config = {
        "name": "rmse",
        "label": "Root Mean Squared Error",
        "type": "across_samples",
    }
    metric = MetricFactory.from_yaml(config)
    assert metric.name == "rmse"
    assert metric.type == MetricType.ACROSS_SAMPLES


def test_metric_compiler():
    """Test compiling metric expressions."""
    compiler = MetricCompiler()
    metric = MetricData(name="mae", label="MAE", type=MetricType.ACROSS_SAMPLES)
    
    agg_exprs, select_expr = compiler.compile_expressions(
        metric.name, metric.agg_expr, metric.select_expr
    )
    assert len(agg_exprs) == 1
    assert select_expr is None


def test_metric_evaluator():
    """Test basic metric evaluation."""
    # Generate sample data
    df = generate_sample_data(n_subjects=3, n_visits=2)
    
    # Create metrics
    metrics = [
        MetricData(name="mae", label="MAE", type=MetricType.ACROSS_SAMPLES),
        MetricData(name="rmse", label="RMSE", type=MetricType.ACROSS_SAMPLES),
    ]
    
    # Create evaluator
    evaluator = MetricEvaluator(
        df=df,
        metrics=metrics,
        ground_truth="actual",
        estimates=["model1", "model2"],
        group_by=["treatment"],
    )
    
    # Evaluate all metrics
    results = evaluator.evaluate_all()
    
    # Check results
    assert isinstance(results, pl.DataFrame)
    assert results.shape[0] > 0
    assert "metric" in results.columns
    assert "estimate" in results.columns
    assert "value" in results.columns


def test_evaluation_config():
    """Test loading configuration."""
    config_dict = {
        "ground_truth": "actual",
        "estimates": ["model1", "model2"],
        "group_by": ["treatment"],
        "metrics": [
            {"name": "mae", "label": "MAE", "type": "across_samples"},
            {"name": "bias", "label": "Bias", "type": "across_samples"},
        ],
    }
    
    config = EvaluationConfig.from_yaml(config_dict)
    assert config.ground_truth == "actual"
    assert len(config.estimates) == 2
    assert len(config.metrics) == 2


def test_end_to_end_workflow():
    """Test complete workflow from config to results."""
    # Configuration
    config_dict = {
        "ground_truth": "actual",
        "estimates": ["model1", "model2"],
        "group_by": ["treatment"],
        "metrics": [
            {"name": "mae", "type": "across_samples"},
            {"name": "rmse", "type": "across_samples"},
            {"name": "bias", "type": "across_samples"},
        ],
    }
    
    # Load configuration
    config = EvaluationConfig.from_yaml(config_dict)
    
    # Generate data
    df = generate_sample_data(n_subjects=4, n_visits=3)
    
    # Create evaluator from config
    evaluator = MetricEvaluator(df=df, **config.to_evaluator_kwargs())
    
    # Evaluate
    results = evaluator.evaluate_all()
    
    # Verify results
    assert isinstance(results, pl.DataFrame)
    assert results.shape[0] == len(config.metrics) * len(config.estimates) * 2  # 2 groups
    assert results["metric"].n_unique() == len(config.metrics)
    assert results["estimate"].n_unique() == len(config.estimates)


def test_custom_metric():
    """Test custom metric with expression."""
    metric = MetricData(
        name="custom",
        label="Custom Metric",
        type=MetricType.ACROSS_SAMPLES,
        agg_expr=["pl.col('absolute_error').median().alias('value')"],
    )
    
    df = generate_sample_data()
    evaluator = MetricEvaluator(
        df=df,
        metrics=[metric],
        ground_truth="actual",
        estimates=["model1"],
    )
    
    result = evaluator.evaluate_single(metric, "model1").collect()
    assert isinstance(result, pl.DataFrame)
    assert "value" in result.columns


def test_two_level_aggregation():
    """Test metric with two-level aggregation."""
    metric = MetricData(
        name="mae_per_subject",
        label="MAE per Subject",
        type=MetricType.ACROSS_SUBJECT,
        agg_expr=["pl.col('absolute_error').mean().alias('value')"],
        select_expr="pl.col('value').mean()",
    )
    
    df = generate_sample_data(n_subjects=3, n_visits=3)
    evaluator = MetricEvaluator(
        df=df,
        metrics=[metric],
        ground_truth="actual",
        estimates=["model1"],
        group_by=["treatment"],
    )
    
    result = evaluator.evaluate_single(metric, "model1").collect()
    assert isinstance(result, pl.DataFrame)
    # Should have one row per treatment group
    assert result.shape[0] == df["treatment"].n_unique()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])