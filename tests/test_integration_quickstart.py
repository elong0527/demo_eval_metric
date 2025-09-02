"""Integration tests that run exact examples from quickstart.qmd."""

import polars as pl

from polars_eval_metrics import MetricDefine, MetricEvaluator
from test_utils import generate_sample_data


class TestQuickstartIntegration:
    """Integration tests running exact examples from quickstart.qmd."""
    
    def test_example_data_generation(self):
        """Test the data generation example from quickstart.qmd."""
        # From quickstart.qmd line 36
        df = generate_sample_data(n_subjects=3, n_visits=2, n_groups=2)
        
        # Verify data structure
        assert df.shape == (6, 9)  # 3 subjects × 2 visits = 6 rows, 9 columns
        expected_columns = [
            "subject_id", "visit_id", "treatment", "gender", "race", 
            "actual", "model1", "model2", "weight"
        ]
        assert df.columns == expected_columns
    
    def test_single_metric_mae_example(self):
        """Test single MAE metric example from quickstart.qmd."""
        df = generate_sample_data(n_subjects=3, n_visits=2, n_groups=2)
        
        # From quickstart.qmd lines 62-67
        evaluator = MetricEvaluator(
            df=df,
            metrics=MetricDefine(name="mae"),
            ground_truth="actual",
            estimates="model1",
        )
        
        # From quickstart.qmd line 74
        result = evaluator.evaluate()
        
        # Verify result structure
        assert result.shape[0] == 1  # Single result
        assert "metric" in result.columns
        assert "estimate" in result.columns
        assert "value" in result.columns
        assert result["metric"][0] == "mae"
        assert result["estimate"][0] == "model1"
    
    def test_equivalent_polars_single_metric(self):
        """Test equivalent Polars code example from quickstart.qmd."""
        df = generate_sample_data(n_subjects=3, n_visits=2, n_groups=2)
        
        # From quickstart.qmd lines 83-85
        direct_result = df.select(
            (pl.col("model1") - pl.col("actual")).abs().mean().alias("mae")
        )
        
        # Framework result
        evaluator = MetricEvaluator(
            df=df,
            metrics=MetricDefine(name="mae"),
            ground_truth="actual",
            estimates="model1",
        )
        framework_result = evaluator.evaluate()
        
        # Should match
        assert abs(framework_result["value"][0] - direct_result["mae"][0]) < 1e-10
    
    def test_grouped_evaluation_example(self):
        """Test grouped evaluation example from quickstart.qmd."""
        df = generate_sample_data(n_subjects=3, n_visits=2, n_groups=2)
        
        # From quickstart.qmd lines 100-109
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
        
        # From quickstart.qmd line 115
        result = evaluator.evaluate()
        
        # Verify structure: 2 treatments × 2 models × 2 metrics = 8 rows
        assert result.shape[0] == 8
        assert set(result["treatment"]) == {"A", "B"}
        assert set(result["estimate"]) == {"model1", "model2"}
        assert set(result["metric"]) == {"mae", "rmse"}
    
    def test_equivalent_polars_grouped_metric(self):
        """Test equivalent Polars code for grouped metrics."""
        df = generate_sample_data(n_subjects=3, n_visits=2, n_groups=2)
        
        # From quickstart.qmd lines 122-129 (simplified for MAE only)
        direct_result = df.group_by("treatment").agg([
            (pl.col("model1") - pl.col("actual")).abs().mean().alias("mae_model1"),
            (pl.col("model2") - pl.col("actual")).abs().mean().alias("mae_model2"),
        ]).sort("treatment")
        
        # Framework result (MAE only for comparison)
        evaluator = MetricEvaluator(
            df=df,
            metrics=MetricDefine(name="mae"),
            ground_truth="actual",
            estimates=["model1", "model2"],
            group_by=["treatment"]
        )
        framework_result = evaluator.evaluate().sort(["treatment", "estimate"])
        
        # Compare MAE values for each treatment/model combination
        framework_a_m1 = framework_result.filter(
            (pl.col("treatment") == "A") & (pl.col("estimate") == "model1")
        )["value"][0]
        direct_a_m1 = direct_result.filter(pl.col("treatment") == "A")["mae_model1"][0]
        assert abs(framework_a_m1 - direct_a_m1) < 1e-10
        
        framework_a_m2 = framework_result.filter(
            (pl.col("treatment") == "A") & (pl.col("estimate") == "model2")
        )["value"][0]
        direct_a_m2 = direct_result.filter(pl.col("treatment") == "A")["mae_model2"][0]
        assert abs(framework_a_m2 - direct_a_m2) < 1e-10
    
    def test_subgroup_evaluation_example(self):
        """Test subgroup evaluation example from quickstart.qmd."""
        df = generate_sample_data(n_subjects=3, n_visits=2, n_groups=2)
        
        # From quickstart.qmd lines 136-147
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
        
        # From quickstart.qmd line 153
        result = evaluator.evaluate()
        
        # Verify subgroup structure
        assert "subgroup_name" in result.columns
        assert "subgroup_value" in result.columns
        assert set(result["subgroup_name"]) == {"gender", "race"}
        
        # Check that we have results for different subgroup values
        gender_values = result.filter(pl.col("subgroup_name") == "gender")["subgroup_value"].unique().to_list()
        race_values = result.filter(pl.col("subgroup_name") == "race")["subgroup_value"].unique().to_list()
        
        assert "F" in gender_values or "M" in gender_values  # At least one gender
        assert len([v for v in race_values if v in ["White", "Black", "Asian", "Hispanic"]]) > 0  # At least one race
    
    def test_lazy_evaluation_explain_example(self):
        """Test LazyFrame explain example from quickstart.qmd."""
        df = generate_sample_data(n_subjects=3, n_visits=2, n_groups=2)
        
        evaluator = MetricEvaluator(
            df=df,
            metrics=MetricDefine(name="mae"),
            ground_truth="actual",
            estimates="model1",
        )
        
        # From quickstart.qmd line 92 - returns LazyFrame
        lazy_result = evaluator.evaluate(collect=False)
        
        # From quickstart.qmd line 93 - should be able to explain
        explain_output = lazy_result.explain()
        
        assert isinstance(lazy_result, pl.LazyFrame)
        assert isinstance(explain_output, str)
        assert "SELECTION" in explain_output or "SELECT" in explain_output
    
    def test_data_shape_and_columns_example(self):
        """Test data inspection example from quickstart.qmd."""
        # From quickstart.qmd line 36
        df = generate_sample_data(n_subjects=3, n_visits=2, n_groups=2)
        
        # From quickstart.qmd lines 37-38
        shape = df.shape
        columns = df.columns
        
        assert shape == (6, 9)  # Expected shape from example (3 subjects × 2 visits = 6 rows)
        assert "subject_id" in columns
        assert "visit_id" in columns
        assert "treatment" in columns
        assert "gender" in columns
        assert "race" in columns
        assert "actual" in columns
        assert "model1" in columns
        assert "model2" in columns
        assert "weight" in columns
