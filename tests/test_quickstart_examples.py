"""Tests based on quickstart.qmd examples for minimal sufficient coverage."""

import polars as pl
import pytest

from polars_eval_metrics import MetricDefine, MetricEvaluator
from test_utils import generate_sample_data as generate_test_data


class TestQuickstartSingleMetric:
    """Test single metric evaluation from quickstart.qmd."""
    
    def test_mae_single_model(self):
        """Test single MAE metric for one model."""
        df = generate_test_data()
        
        evaluator = MetricEvaluator(
            df=df,
            metrics=MetricDefine(name="mae"),
            ground_truth="actual",
            estimates="model1",
        )
        
        result = evaluator.evaluate()
        
        # Check basic structure
        assert result.shape[0] == 1  # One result row
        assert "metric" in result.columns
        assert "estimate" in result.columns
        assert "value" in result.columns
        assert "label" in result.columns
        
        # Check values
        assert result["metric"][0] == "mae"
        assert result["estimate"][0] == "model1"
        assert result["label"][0] == "mae"
        assert isinstance(result["value"][0], float)
        assert result["value"][0] >= 0  # MAE is non-negative


class TestQuickstartGroupedEvaluation:
    """Test grouped evaluation from quickstart.qmd."""
    
    def test_mae_rmse_by_treatment(self):
        """Test MAE and RMSE metrics grouped by treatment."""
        df = generate_test_data()
        
        evaluator = MetricEvaluator(
            df=df,
            metrics=[
                MetricDefine(name="mae"),
                MetricDefine(name="rmse"),
            ],
            ground_truth="actual",
            estimates=["model1", "model2"],
            group_by=["treatment"]
        )
        
        result = evaluator.evaluate()
        
        # Check structure: 2 treatments × 2 models × 2 metrics = 8 rows
        assert result.shape[0] == 8
        
        # Check columns
        expected_cols = ["treatment", "estimate", "metric", "label", "value", "metric_type"]
        for col in expected_cols:
            assert col in result.columns
        
        # Check unique values
        assert set(result["treatment"]) == {"A", "B"}
        assert set(result["estimate"]) == {"model1", "model2"}
        assert set(result["metric"]) == {"mae", "rmse"}
        
        # Check that all values are valid
        assert all(result["value"] >= 0)  # Both MAE and RMSE are non-negative


class TestQuickstartSubgroupEvaluation:
    """Test subgroup evaluation from quickstart.qmd."""
    
    def test_subgroup_evaluation(self):
        """Test evaluation with subgroups (gender and race)."""
        df = generate_test_data()
        
        evaluator = MetricEvaluator(
            df=df,
            metrics=[
                MetricDefine(name="mae"),
                MetricDefine(name="rmse"),
            ],
            ground_truth="actual",
            estimates=["model1", "model2"],
            group_by=["treatment"],
            subgroup_by=["gender", "race"]
        )
        
        result = evaluator.evaluate()
        
        # Check that subgroup columns exist
        assert "subgroup_name" in result.columns
        assert "subgroup_value" in result.columns
        
        # Check subgroup names are correct
        assert set(result["subgroup_name"]) == {"gender", "race"}
        
        # Check that we have results for each subgroup
        gender_results = result.filter(pl.col("subgroup_name") == "gender")
        race_results = result.filter(pl.col("subgroup_name") == "race")
        
        assert len(gender_results) > 0
        assert len(race_results) > 0
        
        # Check subgroup values
        assert "F" in gender_results["subgroup_value"].to_list()
        assert "M" in gender_results["subgroup_value"].to_list()
        assert "White" in race_results["subgroup_value"].to_list()
        assert "Black" in race_results["subgroup_value"].to_list()
        assert "Asian" in race_results["subgroup_value"].to_list()


class TestQuickstartDataIntegrity:
    """Test data integrity and validation."""
    
    def test_column_order_and_sorting(self):
        """Test that results are properly sorted and columns are in correct order."""
        df = generate_test_data()
        
        evaluator = MetricEvaluator(
            df=df,
            metrics=[MetricDefine(name="mae"), MetricDefine(name="rmse")],
            ground_truth="actual",
            estimates=["model1", "model2"],
            group_by=["treatment"],
            subgroup_by=["gender"]
        )
        
        result = evaluator.evaluate()
        
        # Check that results are sorted properly
        # Should be sorted by treatment, subgroup_name, subgroup_value, metric, estimate
        prev_treatment = None
        prev_subgroup = None
        prev_metric = None
        
        for row in result.iter_rows(named=True):
            # Check treatment ordering
            if prev_treatment is not None:
                assert row["treatment"] >= prev_treatment
            
            # Within same treatment, check subgroup and metric ordering  
            if prev_treatment == row["treatment"]:
                if prev_subgroup is not None:
                    assert row["subgroup_value"] >= prev_subgroup or row["subgroup_name"] != "gender"
                
                if prev_subgroup == row["subgroup_value"] and prev_metric is not None:
                    assert row["metric"] >= prev_metric or row["estimate"] != "model1"
            
            prev_treatment = row["treatment"]
            if row["subgroup_name"] == "gender":
                prev_subgroup = row["subgroup_value"]
            prev_metric = row["metric"]
    
    def test_lazy_evaluation_option(self):
        """Test that collect=False returns LazyFrame."""
        df = generate_test_data()
        
        evaluator = MetricEvaluator(
            df=df,
            metrics=MetricDefine(name="mae"),
            ground_truth="actual",
            estimates="model1",
        )
        
        # Test LazyFrame return
        lazy_result = evaluator.evaluate(collect=False)
        assert isinstance(lazy_result, pl.LazyFrame)
        
        # Test that it can be collected
        collected_result = lazy_result.collect()
        assert isinstance(collected_result, pl.DataFrame)
        
        # Test that default behavior returns DataFrame
        default_result = evaluator.evaluate()
        assert isinstance(default_result, pl.DataFrame)


class TestQuickstartErrorHandling:
    """Test error handling and edge cases."""
    
    def test_missing_columns_error(self):
        """Test error when required columns are missing."""
        df = generate_test_data().drop("actual")  # Remove ground truth column
        
        evaluator = MetricEvaluator(
            df=df,
            metrics=MetricDefine(name="mae"),
            ground_truth="actual",
            estimates="model1",
        )
        
        with pytest.raises(Exception):  # Should raise error for missing column
            evaluator.evaluate()
    
    def test_empty_estimates_error(self):
        """Test error when no estimates are provided."""
        df = generate_test_data()
        
        with pytest.raises(ValueError, match="No metrics or estimates to evaluate"):
            evaluator = MetricEvaluator(
                df=df,
                metrics=MetricDefine(name="mae"),
                ground_truth="actual",
                estimates=[],  # Empty estimates
            )
            evaluator.evaluate()
    
    def test_empty_metrics_error(self):
        """Test error when no metrics are provided."""
        df = generate_test_data()
        
        with pytest.raises(ValueError, match="No metrics or estimates to evaluate"):
            evaluator = MetricEvaluator(
                df=df,
                metrics=[],  # Empty metrics
                ground_truth="actual",
                estimates="model1",
            )
            evaluator.evaluate()


class TestQuickstartEquivalentCalculations:
    """Test that results match equivalent direct Polars calculations."""
    
    def test_mae_equivalent_calculation(self):
        """Test that MAE matches direct Polars calculation."""
        df = generate_test_data()
        
        # Framework calculation
        evaluator = MetricEvaluator(
            df=df,
            metrics=MetricDefine(name="mae"),
            ground_truth="actual",
            estimates="model1",
        )
        framework_result = evaluator.evaluate()
        framework_mae = framework_result["value"][0]
        
        # Direct Polars calculation (from quickstart.qmd)
        direct_result = df.select(
            (pl.col("model1") - pl.col("actual")).abs().mean().alias("mae")
        )
        direct_mae = direct_result["mae"][0]
        
        # Should be approximately equal (accounting for floating point precision)
        assert abs(framework_mae - direct_mae) < 1e-10
    
    def test_grouped_mae_equivalent_calculation(self):
        """Test that grouped MAE matches direct Polars calculation."""
        df = generate_test_data()
        
        # Framework calculation
        evaluator = MetricEvaluator(
            df=df,
            metrics=MetricDefine(name="mae"),
            ground_truth="actual",
            estimates="model1",
            group_by=["treatment"]
        )
        framework_result = evaluator.evaluate().sort("treatment")
        
        # Direct Polars calculation
        direct_result = df.group_by("treatment").agg([
            (pl.col("model1") - pl.col("actual")).abs().mean().alias("mae_model1")
        ]).sort("treatment")
        
        # Compare values for each treatment group
        for i in range(len(framework_result)):
            framework_mae = framework_result["value"][i]
            direct_mae = direct_result["mae_model1"][i]
            assert abs(framework_mae - direct_mae) < 1e-10
