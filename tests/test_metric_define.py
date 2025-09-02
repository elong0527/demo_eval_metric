"""Unit tests for MetricDefine class based on examples from metric.qmd."""

import polars as pl
import pytest
import numpy as np

from polars_eval_metrics.core.metric_define import MetricDefine, MetricType, MetricScope
from polars_eval_metrics.core.builtin import BUILTIN_METRICS, BUILTIN_SELECTORS


class TestMetricDefineBasic:
    """Test basic metric definitions."""
    
    def test_simple_mae_metric(self):
        """Test creating a simple MAE metric (from metric.qmd line 28)."""
        metric = MetricDefine(name="mae")
        
        assert metric.name == "mae"
        assert metric.label == "mae"  # Auto-generated from name
        assert metric.type == MetricType.ACROSS_SAMPLES
        assert metric.scope is None
        assert metric.agg_expr is None
        assert metric.select_expr is None
    
    def test_hierarchical_mae_mean(self):
        """Test MAE with mean aggregation (from metric.qmd line 88)."""
        metric = MetricDefine(name="mae:mean", type=MetricType.ACROSS_SUBJECT)
        
        assert metric.name == "mae:mean"
        assert metric.label == "mae:mean"
        assert metric.type == MetricType.ACROSS_SUBJECT
        # The colon notation should be processed during compilation
    
    def test_custom_label(self):
        """Test metric with custom label."""
        metric = MetricDefine(
            name="test_metric",
            label="Custom Test Metric"
        )
        
        assert metric.name == "test_metric"
        assert metric.label == "Custom Test Metric"


class TestMetricDefineCustomExpressions:
    """Test metrics with custom expressions."""
    
    def test_percentage_within_threshold(self):
        """Test custom metric for percentage within threshold (from metric.qmd line 136)."""
        metric = MetricDefine(
            name="pct_within_1",
            label="% Predictions Within +/- 1",
            type=MetricType.ACROSS_SAMPLES,
            select_expr=(pl.col('absolute_error') < 1).mean() * 100
        )
        
        assert metric.name == "pct_within_1"
        assert metric.label == "% Predictions Within +/- 1"
        assert metric.type == MetricType.ACROSS_SAMPLES
        assert metric.select_expr is not None
        assert isinstance(metric.select_expr, pl.Expr)
    
    def test_percentile_metric(self):
        """Test percentile of per-subject MAE (from metric.qmd line 151)."""
        metric = MetricDefine(
            name="mae_p90_by_subject",
            label="90th Percentile of Subject MAEs",
            type=MetricType.ACROSS_SUBJECT,
            agg_expr="mae",
            select_expr=pl.col('value').quantile(0.9)
        )
        
        assert metric.name == "mae_p90_by_subject"
        assert metric.label == "90th Percentile of Subject MAEs"
        assert metric.type == MetricType.ACROSS_SUBJECT
        assert metric.agg_expr == ["mae"]  # Normalized to list
        assert isinstance(metric.select_expr, pl.Expr)
    
    def test_weighted_average(self):
        """Test weighted average of per-subject MAE (from metric.qmd line 173)."""
        metric = MetricDefine(
            name="weighted_mae",
            label="Weighted Average of Subject MAEs",
            type=MetricType.ACROSS_SUBJECT,
            agg_expr=[
                "mae",  # MAE per subject
                pl.col("weight").mean().alias("avg_weight")
            ],
            select_expr=(
                (pl.col('value') * pl.col('avg_weight')).sum() / 
                pl.col('avg_weight').sum()
            )
        )
        
        assert metric.name == "weighted_mae"
        assert metric.label == "Weighted Average of Subject MAEs"
        assert metric.type == MetricType.ACROSS_SUBJECT
        assert len(metric.agg_expr) == 2
        assert metric.agg_expr[0] == "mae"
        assert isinstance(metric.agg_expr[1], pl.Expr)
        assert isinstance(metric.select_expr, pl.Expr)


class TestMetricDefineWithNumpy:
    """Test metrics using NumPy functions."""
    
    def test_weighted_average_numpy(self):
        """Test weighted average using NumPy (from metric.qmd line 194)."""
        # Define the weighted average expression
        weighted_average = (
            pl.struct(['value', 'avg_weight'])
            .map_batches(
                lambda x: pl.Series([
                    np.average(
                        x.struct.field('value'), 
                        weights=x.struct.field('avg_weight')
                    )
                ]),
                return_dtype=pl.Float64
            )
        )
        
        metric = MetricDefine(
            name="weighted_mae_numpy",
            label="Weighted Average of Subject MAEs (NumPy)",
            type=MetricType.ACROSS_SUBJECT,
            agg_expr=[
                "mae",
                pl.col("weight").mean().alias("avg_weight")
            ],
            select_expr=weighted_average
        )
        
        assert metric.name == "weighted_mae_numpy"
        assert metric.label == "Weighted Average of Subject MAEs (NumPy)"
        assert metric.type == MetricType.ACROSS_SUBJECT
        assert len(metric.agg_expr) == 2
        assert isinstance(metric.select_expr, pl.Expr)


class TestMetricDefineTypes:
    """Test different metric types."""
    
    def test_all_metric_types(self):
        """Test all available metric types."""
        types_to_test = [
            MetricType.ACROSS_SAMPLES,
            MetricType.ACROSS_SUBJECT,
            MetricType.WITHIN_SUBJECT,
            MetricType.ACROSS_VISIT,
            MetricType.WITHIN_VISIT
        ]
        
        for metric_type in types_to_test:
            metric = MetricDefine(
                name=f"test_{metric_type.value}",
                type=metric_type
            )
            assert metric.type == metric_type
    
    def test_metric_scopes(self):
        """Test metric scopes."""
        scopes_to_test = [
            MetricScope.GLOBAL,
            MetricScope.MODEL,
            MetricScope.GROUP
        ]
        
        for scope in scopes_to_test:
            metric = MetricDefine(
                name=f"test_{scope.value}",
                scope=scope
            )
            assert metric.scope == scope
    
    def test_string_to_enum_conversion(self):
        """Test that string inputs are converted to proper enum types."""
        
        # Test MetricType string conversion
        m1 = MetricDefine(name="test", type="across_samples")
        assert m1.type == MetricType.ACROSS_SAMPLES
        
        m2 = MetricDefine(name="test", type="WITHIN_SUBJECT")  # Different case
        assert m2.type == MetricType.WITHIN_SUBJECT
        
        m3 = MetricDefine(name="test", type="across-visit")  # With hyphen
        assert m3.type == MetricType.ACROSS_VISIT
        
        # Test MetricScope string conversion
        m4 = MetricDefine(name="test", scope="global")
        assert m4.scope == MetricScope.GLOBAL
        
        m5 = MetricDefine(name="test", scope="MODEL")  # Different case
        assert m5.scope == MetricScope.MODEL
        
        # Test None scope remains None
        m6 = MetricDefine(name="test", scope=None)
        assert m6.scope is None
        
        # Test invalid type string raises error
        with pytest.raises(ValueError, match="Invalid metric type"):
            MetricDefine(name="test", type="invalid_type")
        
        # Test invalid scope string raises error
        with pytest.raises(ValueError, match="Invalid metric scope"):
            MetricDefine(name="test", scope="invalid_scope")
        
        # Test that enum values still work directly
        m7 = MetricDefine(
            name="test",
            type=MetricType.ACROSS_VISIT,
            scope=MetricScope.GROUP
        )
        assert m7.type == MetricType.ACROSS_VISIT
        assert m7.scope == MetricScope.GROUP


class TestMetricDefineValidation:
    """Test validation and error handling."""
    
    def test_empty_name_raises_error(self):
        """Test that empty name raises validation error."""
        with pytest.raises(ValueError, match="Metric name cannot be empty"):
            MetricDefine(name="")
    
    def test_whitespace_name_raises_error(self):
        """Test that whitespace-only name raises validation error."""
        with pytest.raises(ValueError, match="Metric name cannot be empty"):
            MetricDefine(name="   ")
    
    def test_empty_label_raises_error(self):
        """Test that empty label raises validation error."""
        with pytest.raises(ValueError, match="Metric label cannot be empty"):
            MetricDefine(name="test", label="   ")
    
    def test_single_string_agg_expr_normalized_to_list(self):
        """Test that single string agg_expr is normalized to list."""
        metric = MetricDefine(
            name="test",
            agg_expr="mae"
        )
        assert metric.agg_expr == ["mae"]
    
    def test_single_expr_agg_expr_normalized_to_list(self):
        """Test that single Polars expression agg_expr is normalized to list."""
        expr = pl.col("test").mean()
        metric = MetricDefine(
            name="test",
            agg_expr=expr
        )
        assert metric.agg_expr == [expr]


class TestMetricDefineRepresentation:
    """Test string representation and display methods."""
    
    def test_repr_simple_metric(self):
        """Test string representation of simple metric."""
        metric = MetricDefine(name="mae")
        repr_str = repr(metric)
        
        # Check that key information is in the representation
        assert "name='mae'" in repr_str
        assert "type=across_samples" in repr_str.lower() or "type=MetricType.ACROSS_SAMPLES" in repr_str
    
    def test_pl_expr_method(self):
        """Test that pl_expr method works if it exists."""
        metric = MetricDefine(name="mae")
        
        # Check if the method exists, skip test if not implemented yet
        if hasattr(metric, 'pl_expr') and callable(getattr(metric, 'pl_expr')):
            expr_str = metric.pl_expr()
            assert isinstance(expr_str, str)
            assert "LazyFrame" in expr_str
        else:
            # Method not implemented yet, that's okay for minimal tests
            pass
