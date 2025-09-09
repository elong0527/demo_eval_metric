"""Test pivot_by_group() and pivot_by_model() methods matching metric_pivot.qmd examples"""

import polars as pl
import pytest

from polars_eval_metrics import MetricDefine, MetricEvaluator


class TestPivotByMethods:
    """Test pivot_by_group() and pivot_by_model() methods with scenarios from documentation"""

    @pytest.fixture
    def sample_data(self):
        """Create sample dataset matching metric_pivot.qmd"""
        return pl.DataFrame(
            {
                "subject_id": list(range(1, 21)) * 2,
                "treatment": (["A"] * 10 + ["B"] * 10) * 2,
                "region": (
                    ["North"] * 5 + ["South"] * 5 + ["North"] * 5 + ["South"] * 5
                )
                * 2,
                "age_group": (["Young", "Old"] * 10) * 2,
                "sex": (["M", "F"] * 20),
                "actual": [10, 15, 20, 25, 30, 35, 40, 45, 50, 55] * 4,
                # Model A: Generally accurate with small errors
                "model_a": [10.5, 14.8, 19.2, 25.3, 29.7, 35.1, 39.9, 44.6, 50.4, 54.2]
                * 4,
                # Model B: Less accurate with some larger errors
                "model_b": [9.2, 16.1, 22.5, 23.8, 32.4, 37.8, 38.1, 48.2, 47.5, 58.9]
                * 4,
            }
        )

    @pytest.fixture
    def group_metrics(self):
        """Metrics used in documentation examples"""
        return [
            MetricDefine(
                name="n_subject", label="Total Enrolled Subjects", scope="global"
            ),
            MetricDefine(name="n_subject", label="Number of Subjects", scope="group"),
            MetricDefine(name="mae", label="MAE"),
            MetricDefine(name="rmse", label="RMSE"),
        ]

    def test_case1_pivot_by_group_without_subgroups(self, sample_data, group_metrics):
        """Test Case 1: Pivot by Group (without subgroups)"""

        evaluator = MetricEvaluator(
            df=sample_data,
            metrics=group_metrics,
            ground_truth="actual",
            estimates=["model_a", "model_b"],
            group_by=["treatment", "region"],
        )

        result = evaluator.pivot_by_group()

        # Verify structure
        assert isinstance(result, pl.DataFrame)
        assert not result.is_empty()

        # Should have group combinations as rows
        assert "treatment" in result.columns
        assert "region" in result.columns

        # Should have 4 group combinations (2 treatments x 2 regions)
        assert result.height == 4

        # Should have global scope column (broadcast to all rows)
        global_cols = [
            col for col in result.columns if "Total Enrolled Subjects" in col
        ]
        assert len(global_cols) == 1

        # Should have group scope columns (one per group)
        group_cols = [col for col in result.columns if "Number of Subjects" in col]
        assert len(group_cols) == 1

        # Should have default scope columns (model x metric)
        default_cols = [
            col for col in result.columns if col.startswith(("model_a_", "model_b_"))
        ]
        assert len(default_cols) == 4  # 2 models x 2 metrics (MAE, RMSE)

        print("✓ Case 1: Pivot by Group (without subgroups) passed")

    def test_case2_pivot_by_group_with_subgroups(self, sample_data, group_metrics):
        """Test Case 2: Pivot by Group (with subgroups)"""

        evaluator = MetricEvaluator(
            df=sample_data,
            metrics=group_metrics,
            ground_truth="actual",
            estimates=["model_a", "model_b"],
            group_by=["treatment", "region"],
            subgroup_by=["age_group"],
        )

        result = evaluator.pivot_by_group()

        # Verify structure
        assert isinstance(result, pl.DataFrame)
        assert not result.is_empty()

        # Should have subgroup columns
        assert "subgroup_name" in result.columns
        assert "subgroup_value" in result.columns

        # Should have group combinations as rows, stratified by subgroups
        assert "treatment" in result.columns
        assert "region" in result.columns

        # Should have rows for each group x subgroup combination
        # 4 groups x 2 age_groups = 8 rows (but some might be missing if no data)
        assert result.height >= 4  # At least one row per group

        # Verify subgroup stratification
        subgroup_values = result["subgroup_value"].unique().sort().to_list()
        assert "Young" in subgroup_values
        assert "Old" in subgroup_values

        print("✓ Case 2: Pivot by Group (with subgroups) passed")

    def test_case3_pivot_by_model_without_subgroups(self, sample_data, group_metrics):
        """Test Case 3: Pivot by Model (without subgroups)"""

        evaluator = MetricEvaluator(
            df=sample_data,
            metrics=group_metrics,
            ground_truth="actual",
            estimates=["model_a", "model_b"],
            group_by=["treatment", "region"],
        )

        result = evaluator.pivot_by_model()

        # Verify structure
        assert isinstance(result, pl.DataFrame)
        assert not result.is_empty()

        # Should have models as rows
        assert "estimate" in result.columns
        model_names = result["estimate"].unique().sort().to_list()
        assert "model_a" in model_names
        assert "model_b" in model_names
        assert result.height == 2  # One row per model

        # Should have global scope columns
        global_cols = [
            col for col in result.columns if "Total Enrolled Subjects" in col
        ]
        assert len(global_cols) == 1

        # Should have group scope columns (group x metric)
        group_cols = [col for col in result.columns if "Number of Subjects" in col]
        assert len(group_cols) >= 1  # Group combinations

        # Should have default scope columns (group x metric)
        default_cols = [
            col
            for col in result.columns
            if any(grp in col for grp in ["A_", "B_"])
            and any(met in col for met in ["mae", "rmse"])
        ]
        assert len(default_cols) == 8  # 4 groups x 2 metrics

        print("✓ Case 3: Pivot by Model (without subgroups) passed")

    def test_case4_pivot_by_model_with_subgroups(self, sample_data, group_metrics):
        """Test Case 4: Pivot by Model (with subgroups)"""

        evaluator = MetricEvaluator(
            df=sample_data,
            metrics=group_metrics,
            ground_truth="actual",
            estimates=["model_a", "model_b"],
            group_by=["treatment", "region"],
            subgroup_by=["age_group"],
        )

        result = evaluator.pivot_by_model()

        # Verify structure
        assert isinstance(result, pl.DataFrame)
        assert not result.is_empty()

        # Should have subgroup columns
        assert "subgroup_name" in result.columns
        assert "subgroup_value" in result.columns
        assert "estimate" in result.columns

        # Should have rows for each model x subgroup combination
        # 2 models x 2 age_groups = 4 rows
        assert result.height >= 2  # At least one row per model

        # Verify model stratification within subgroups
        estimates = result["estimate"].unique().sort().to_list()
        assert "model_a" in estimates
        assert "model_b" in estimates

        # Verify subgroup stratification
        subgroup_values = result["subgroup_value"].unique().sort().to_list()
        assert "Young" in subgroup_values
        assert "Old" in subgroup_values

        print("✓ Case 4: Pivot by Model (with subgroups) passed")

    def test_column_ordering(self, sample_data, group_metrics):
        """Test that column ordering follows index -> global -> group -> default pattern"""

        evaluator = MetricEvaluator(
            df=sample_data,
            metrics=group_metrics,
            ground_truth="actual",
            estimates=["model_a", "model_b"],
            group_by=["treatment", "region"],
        )

        # Test pivot_by_group ordering
        result_group = evaluator.pivot_by_group()
        cols_group = result_group.columns

        # Index columns should come first
        index_start = 0
        assert cols_group[index_start] == "treatment"
        assert cols_group[index_start + 1] == "region"

        # Global columns should come after index
        global_col_idx = next(
            i for i, col in enumerate(cols_group) if "Total Enrolled Subjects" in col
        )
        assert global_col_idx > 1  # After index columns

        # Test pivot_by_model ordering
        result_model = evaluator.pivot_by_model()
        cols_model = result_model.columns

        # Estimate should be in index
        assert "estimate" in cols_model

        # Global columns should come before default columns
        if any("Total Enrolled Subjects" in col for col in cols_model):
            global_idx = next(
                i
                for i, col in enumerate(cols_model)
                if "Total Enrolled Subjects" in col
            )
            default_indices = [
                i
                for i, col in enumerate(cols_model)
                if any(grp in col for grp in ["A_", "B_"])
                and any(met in col for met in ["mae", "rmse"])
            ]
            if default_indices:
                assert global_idx < min(default_indices)

        print("✓ Column ordering test passed")

    def test_mixed_scopes_compatibility(self, sample_data):
        """Test that different scope combinations work correctly"""

        # Test with only global scope
        evaluator_global = MetricEvaluator(
            df=sample_data,
            metrics=[
                MetricDefine(name="n_subject", label="Total Subjects", scope="global")
            ],
            ground_truth="actual",
            estimates=["model_a"],
            group_by=["treatment"],
        )

        result_global_group = evaluator_global.pivot_by_group()
        result_global_model = evaluator_global.pivot_by_model()

        assert not result_global_group.is_empty()
        assert not result_global_model.is_empty()

        # Test with only group scope
        evaluator_group = MetricEvaluator(
            df=sample_data,
            metrics=[
                MetricDefine(name="n_subject", label="Group Subjects", scope="group")
            ],
            ground_truth="actual",
            estimates=["model_a"],
            group_by=["treatment"],
        )

        result_group_group = evaluator_group.pivot_by_group()
        result_group_model = evaluator_group.pivot_by_model()

        assert not result_group_group.is_empty()
        assert not result_group_model.is_empty()

        # Test with only default scope
        evaluator_default = MetricEvaluator(
            df=sample_data,
            metrics=[MetricDefine(name="mae", label="MAE")],
            ground_truth="actual",
            estimates=["model_a"],
            group_by=["treatment"],
        )

        result_default_group = evaluator_default.pivot_by_group()
        result_default_model = evaluator_default.pivot_by_model()

        assert not result_default_group.is_empty()
        assert not result_default_model.is_empty()

        print("✓ Mixed scopes compatibility test passed")

    def test_caching_efficiency(self, sample_data, group_metrics):
        """Test that both pivot methods use cached evaluation results"""

        evaluator = MetricEvaluator(
            df=sample_data,
            metrics=group_metrics,
            ground_truth="actual",
            estimates=["model_a", "model_b"],
            group_by=["treatment", "region"],
        )

        # Clear cache to start fresh
        evaluator.clear_cache()
        assert len(evaluator._evaluation_cache) == 0

        # First call should populate cache
        result1 = evaluator.pivot_by_group()
        assert len(evaluator._evaluation_cache) == 1

        # Second call should use cache (same parameters)
        result2 = evaluator.pivot_by_model()
        assert len(evaluator._evaluation_cache) == 1  # Still just one cached result

        # Results should not be empty
        assert not result1.is_empty()
        assert not result2.is_empty()

        print("✓ Caching efficiency test passed")


if __name__ == "__main__":
    # Create test instance and run tests
    test_instance = TestPivotByMethods()

    # Create data manually for standalone execution
    sample_data = pl.DataFrame(
        {
            "subject_id": list(range(1, 21)) * 2,
            "treatment": (["A"] * 10 + ["B"] * 10) * 2,
            "region": (["North"] * 5 + ["South"] * 5 + ["North"] * 5 + ["South"] * 5)
            * 2,
            "age_group": (["Young", "Old"] * 10) * 2,
            "sex": (["M", "F"] * 20),
            "actual": [10, 15, 20, 25, 30, 35, 40, 45, 50, 55] * 4,
            "model_a": [10.5, 14.8, 19.2, 25.3, 29.7, 35.1, 39.9, 44.6, 50.4, 54.2] * 4,
            "model_b": [9.2, 16.1, 22.5, 23.8, 32.4, 37.8, 38.1, 48.2, 47.5, 58.9] * 4,
        }
    )

    group_metrics = [
        MetricDefine(name="n_subject", label="Total Enrolled Subjects", scope="global"),
        MetricDefine(name="n_subject", label="Number of Subjects", scope="group"),
        MetricDefine(name="mae", label="MAE"),
        MetricDefine(name="rmse", label="RMSE"),
    ]

    # Run all tests
    test_instance.test_case1_pivot_by_group_without_subgroups(
        sample_data, group_metrics
    )
    test_instance.test_case2_pivot_by_group_with_subgroups(sample_data, group_metrics)
    test_instance.test_case3_pivot_by_model_without_subgroups(
        sample_data, group_metrics
    )
    test_instance.test_case4_pivot_by_model_with_subgroups(sample_data, group_metrics)
    test_instance.test_column_ordering(sample_data, group_metrics)
    test_instance.test_mixed_scopes_compatibility(sample_data)
    test_instance.test_caching_efficiency(sample_data, group_metrics)

    print("\n✅ All pivot_by_* method tests passed!")
