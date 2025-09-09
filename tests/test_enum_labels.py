"""Test enum label ordering in MetricEvaluator"""

import polars as pl
import pytest

from polars_eval_metrics import MetricDefine, MetricEvaluator


def test_enum_label_ordering():
    """Test that metric, label, and estimate columns use enum with definition-based ordering"""
    
    # Create sample data
    df = pl.DataFrame({
        "subject_id": [1, 1, 2, 2, 3, 3] * 2,
        "visit_id": [1, 2, 1, 2, 1, 2] * 2,
        "treatment": ["A", "A", "A", "B", "B", "B"] * 2,
        "actual": [10, 12, 15, 18, 20, 22] * 2,
        "model1": [11, 13, 14, 19, 21, 20] * 2,
        "model2": [9, 11, 16, 17, 19, 23] * 2,
    })
    
    # Define metrics in specific order (not alphabetical)
    metrics = [
        MetricDefine(name="rmse", label="RMSE"),
        MetricDefine(name="mae", label="MAE"),
        MetricDefine(name="me", label="Mean Error"),
    ]
    
    # Define estimates in specific order (model2 before model1)
    estimates = ["model2", "model1"]
    
    # Create evaluator
    evaluator = MetricEvaluator(
        df=df,
        metrics=metrics,
        ground_truth="actual",
        estimates=estimates,
        group_by=["treatment"]
    )
    
    # Evaluate
    result = evaluator.evaluate()
    
    # Check that metric column is an enum
    metric_dtype = result.schema["metric"]
    assert isinstance(metric_dtype, pl.Enum), f"Expected Enum type for metric, got {metric_dtype}"
    
    # Check that the metric enum categories match the definition order
    expected_metric_order = ["rmse", "mae", "me"]
    metric_categories = metric_dtype.categories.to_list()
    assert metric_categories == expected_metric_order, f"Expected {expected_metric_order}, got {metric_categories}"
    
    # Check that label column is an enum
    label_dtype = result.schema["label"]
    assert isinstance(label_dtype, pl.Enum), f"Expected Enum type for label, got {label_dtype}"
    
    # Check that the label enum categories match the definition order
    expected_label_order = ["RMSE", "MAE", "Mean Error"]
    label_categories = label_dtype.categories.to_list()
    assert label_categories == expected_label_order, f"Expected {expected_label_order}, got {label_categories}"
    
    # Check that estimate column is an enum
    estimate_dtype = result.schema["estimate"]
    assert isinstance(estimate_dtype, pl.Enum), f"Expected Enum type for estimate, got {estimate_dtype}"
    
    # Check that the estimate enum categories match the input order
    expected_estimate_order = ["model2", "model1"]
    estimate_categories = estimate_dtype.categories.to_list()
    assert estimate_categories == expected_estimate_order, f"Expected {expected_estimate_order}, got {estimate_categories}"
    
    # Check that results are sorted by the enum order (not alphabetically)
    # The labels should appear in definition order within each group
    first_treatment_a_model2 = result.filter(
        (pl.col("treatment") == "A") & (pl.col("estimate") == "model2")
    )
    
    # Check metric order
    metrics_in_group = first_treatment_a_model2["metric"].to_list()
    assert metrics_in_group == expected_metric_order, (
        f"Metrics not in definition order. Expected {expected_metric_order}, got {metrics_in_group}"
    )
    
    # Check label order
    labels_in_group = first_treatment_a_model2["label"].to_list()
    assert labels_in_group == expected_label_order, (
        f"Labels not in definition order. Expected {expected_label_order}, got {labels_in_group}"
    )


def test_enum_label_with_mixed_metrics():
    """Test enum ordering with mixed custom and default labels"""
    
    # Create sample data
    df = pl.DataFrame({
        "actual": [1, 2, 3, 4, 5],
        "predicted": [1.1, 2.2, 2.9, 4.1, 5.3],
    })
    
    # Define metrics with mixed labels
    metrics = [
        MetricDefine(name="rmse"),  # Will use default label "rmse"
        MetricDefine(name="mae", label="Mean Absolute Error"),
        MetricDefine(name="me"),  # Will use default label "me"
    ]
    
    # Create evaluator
    evaluator = MetricEvaluator(
        df=df,
        metrics=metrics,
        ground_truth="actual",
        estimates=["predicted"]
    )
    
    # Evaluate
    result = evaluator.evaluate()
    
    # Check enum categories
    label_dtype = result.schema["label"]
    expected_order = ["rmse", "Mean Absolute Error", "me"]
    enum_categories = label_dtype.categories.to_list()
    assert enum_categories == expected_order, f"Expected {expected_order}, got {enum_categories}"
    
    # Check that labels appear in definition order
    labels_in_result = result["label"].to_list()
    assert labels_in_result == expected_order, (
        f"Labels not in definition order. Expected {expected_order}, got {labels_in_result}"
    )


def test_enum_label_sorting_performance():
    """Test that enum labels improve sorting performance"""
    
    # Create larger dataset for performance testing
    n_subjects = 100
    n_visits = 10
    
    df = pl.DataFrame({
        "subject_id": [i for i in range(n_subjects) for _ in range(n_visits)],
        "visit_id": [j for _ in range(n_subjects) for j in range(n_visits)],
        "actual": [10 + i + j for i in range(n_subjects) for j in range(n_visits)],
        "model1": [10 + i + j + 0.5 for i in range(n_subjects) for j in range(n_visits)],
    })
    
    # Define multiple metrics
    metrics = [
        MetricDefine(name="rmse", label="Root Mean Squared Error"),
        MetricDefine(name="mae", label="Mean Absolute Error"),
        MetricDefine(name="mse", label="Mean Squared Error"),
        MetricDefine(name="me", label="Mean Error"),
        MetricDefine(name="mape", label="Mean Absolute Percentage Error"),
    ]
    
    # Create evaluator
    evaluator = MetricEvaluator(
        df=df,
        metrics=metrics,
        ground_truth="actual",
        estimates=["model1"]
    )
    
    # Evaluate
    result = evaluator.evaluate()
    
    # Verify it's an enum
    assert isinstance(result.schema["label"], pl.Enum)
    
    # Check that sorting by label uses the definition order
    sorted_result = result.sort("label")
    expected_label_order = [
        "Root Mean Squared Error",
        "Mean Absolute Error", 
        "Mean Squared Error",
        "Mean Error",
        "Mean Absolute Percentage Error"
    ]
    
    actual_labels = sorted_result["label"].unique(maintain_order=True).to_list()
    assert actual_labels == expected_label_order, (
        f"Sorting doesn't follow definition order. Expected {expected_label_order}, got {actual_labels}"
    )


if __name__ == "__main__":
    test_enum_label_ordering()
    test_enum_label_with_mixed_metrics()
    test_enum_label_sorting_performance()
    print("All tests passed!")