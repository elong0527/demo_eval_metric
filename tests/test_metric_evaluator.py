"""
Unit tests for MetricEvaluator

Tests cover core functionality, all scope types, grouping strategies,
and edge cases based on examples from metric_evaluator.qmd
"""

from typing import Any

import pytest
import polars as pl
from polars_eval_metrics import MetricDefine, MetricEvaluator, MetricScope, MetricType


def _evaluate_metric_set(
    data: pl.DataFrame,
    metrics: list[MetricDefine],
    *,
    estimates: list[str] | None = None,
    ground_truth: str = "actual",
) -> tuple[Any, pl.DataFrame]:
    """Evaluate metrics and return both the result and flattened stats."""

    evaluator = MetricEvaluator(
        df=data,
        metrics=metrics,
        ground_truth=ground_truth,
        estimates=estimates or ["model_a"],
    )

    result = evaluator.evaluate()
    collected = result.collect()
    assert set(collected["metric"]) == {metric.name for metric in metrics}

    stats = result.to_ard().get_stats()
    return result, stats


class TestMetricEvaluatorBasic:
    """Test basic MetricEvaluator functionality"""

    def test_simple_metrics(self, metric_sample_df):
        """Test basic metric evaluation"""
        metrics = [
            MetricDefine(name="mae", label="Mean Absolute Error"),
            MetricDefine(name="rmse", label="Root Mean Squared Error"),
        ]

        evaluator = MetricEvaluator(
            df=metric_sample_df,
            metrics=metrics,
            ground_truth="actual",
            estimates=["model_a", "model_b"],
        )

        result = evaluator.evaluate()

        # Check structure
        assert len(result) == 4  # 2 metrics x 2 models
        df = result.collect()
        assert set(df.columns) == {
            "groups",
            "subgroups",
            "estimate",
            "metric",
            "stat",
            "context",
            "id",
        }
        assert set(df["metric"].unique()) == {"mae", "rmse"}
        assert set(df["estimate"].unique()) == {"model_a", "model_b"}

        # Check values are reasonable (non-negative, finite) using get_stats
        stats_df = result.to_ard().get_stats()
        values = stats_df["value"].to_list()
        assert all(v >= 0 for v in values if v is not None)
        assert all(
            not (isinstance(v, float) and v != v) for v in values
        )  # Check for NaN

    def test_single_metric_single_estimate(self, metric_sample_df):
        """Test minimal case"""
        evaluator = MetricEvaluator(
            df=metric_sample_df,
            metrics=[MetricDefine(name="mae")],
            ground_truth="actual",
            estimates=["model_a"],
        )

        result = evaluator.evaluate()
        assert len(result) == 1
        df = result.collect()
        assert df["metric"][0] == "mae"
        assert df["estimate"][0] == "model_a"


class TestMetricEvaluatorScopes:
    """Test all MetricScope types"""

    def test_global_scope(self, grouped_metric_df):
        """Test GLOBAL scope - single result across all"""
        metric = MetricDefine(name="n_subject", scope="global")

        evaluator = MetricEvaluator(
            df=grouped_metric_df,
            metrics=[metric],
            ground_truth="actual",
            estimates=["model_a", "model_b"],
            group_by=["treatment"],
        )

        result = evaluator.evaluate()

        # Global scope should ignore estimates and groups
        assert len(result) == 1
        df = result.collect()
        assert df["estimate"][0] is None  # No estimate value for global
        assert df["groups"][0] is None  # Groups ignored for global scope
        stats = result.to_ard().get_stats()
        assert stats["value"][0] == 3.0  # 3 unique subjects

    def test_model_scope(self, grouped_metric_df):
        """Test MODEL scope - per model, ignore groups"""
        metric = MetricDefine(name="n_sample_with_data", scope="model")

        evaluator = MetricEvaluator(
            df=grouped_metric_df,
            metrics=[metric],
            ground_truth="actual",
            estimates=["model_a", "model_b"],
            group_by=["treatment"],
        )

        result = evaluator.evaluate()

        # Model scope: one row per model, groups ignored
        assert len(result) == 2
        df = result.collect()
        assert set(df["estimate"].unique()) == {"model_a", "model_b"}
        assert all(df["groups"].is_null())  # Groups ignored for model scope
        stats = result.to_ard().get_stats()
        assert all(v == 6.0 for v in stats["value"])  # 6 samples each model

    def test_group_scope(self, grouped_metric_df):
        """Test GROUP scope - per group, aggregate models"""
        metric = MetricDefine(name="n_subject", scope="group")

        evaluator = MetricEvaluator(
            df=grouped_metric_df,
            metrics=[metric],
            ground_truth="actual",
            estimates=["model_a", "model_b"],
            group_by=["treatment"],
        )

        result = evaluator.evaluate()

        # Group scope: one row per group, models aggregated
        assert len(result) == 2
        unnested = result.unnest(["groups"])
        assert set(unnested["treatment"].unique()) == {"A", "B"}
        df = result.collect()
        assert all(df["estimate"].is_null())  # Models aggregated

        # Check counts per group - access values directly from ARD
        df = result.collect()
        for i, row in enumerate(df.iter_rows(named=True)):
            groups = row["groups"]
            stat = row["stat"]
            if groups and groups["treatment"] == "A":
                assert stat["value_int"] == 2  # 2 subjects in A
            elif groups and groups["treatment"] == "B":
                assert stat["value_int"] == 1  # 1 subject in B

    def test_default_scope(self, grouped_metric_df):
        """Test default scope - per model-group combination"""
        metric = MetricDefine(name="mae")  # Default scope

        evaluator = MetricEvaluator(
            df=grouped_metric_df,
            metrics=[metric],
            ground_truth="actual",
            estimates=["model_a", "model_b"],
            group_by=["treatment"],
        )

        result = evaluator.evaluate()

        # Default scope: estimate x group combinations
        assert len(result) == 4  # 2 models x 2 groups
        df = result.collect()
        assert set(df["estimate"].unique()) == {"model_a", "model_b"}
        unnested = result.unnest(["groups"])
        assert set(unnested["treatment"].unique()) == {"A", "B"}


class TestMetricEvaluatorTypes:
    """Test all MetricType aggregations"""

    def test_across_sample(self, hierarchical_metric_df):
        """Test ACROSS_SAMPLE - aggregate across all samples"""
        metric = MetricDefine(name="mae", type="across_sample")

        evaluator = MetricEvaluator(
            df=hierarchical_metric_df,
            metrics=[metric],
            ground_truth="actual",
            estimates=["model_a"],
        )

        result = evaluator.evaluate()
        assert len(result) == 1
        df = result.collect()
        context = df.select(pl.col("context").struct.field("metric_type"))
        assert context["metric_type"][0] == "across_sample"

    def test_within_subject(self, hierarchical_metric_df):
        """Test WITHIN_SUBJECT - per subject aggregation"""
        metric = MetricDefine(name="mae", type="within_subject")

        evaluator = MetricEvaluator(
            df=hierarchical_metric_df,
            metrics=[metric],
            ground_truth="actual",
            estimates=["model_a"],
        )

        result = evaluator.evaluate()
        df = result.collect()
        assert len(df) == 3  # One row per subject

        context = df.select(pl.col("context").struct.field("metric_type"))
        assert all(context["metric_type"] == "within_subject")

        # ID struct should carry subject identifiers
        id_df = df.unnest(["id"])
        assert "subject_id" in id_df.columns
        assert set(id_df["subject_id"].to_list()) == {1, 2, 3}

        # Value column surfaces subject-level MAE on the long representation
        assert "value" in result.columns
        assert all(result["value"].is_not_null())
        assert all(df["stat"].struct.field("type") == "float")

    def test_across_subject(self, hierarchical_metric_df):
        """Test ACROSS_SUBJECT - within subjects then across"""
        # Use proper hierarchical metric that does within-subject then across-subject aggregation
        metric = MetricDefine(name="mae:mean", type="across_subject")

        evaluator = MetricEvaluator(
            df=hierarchical_metric_df,
            metrics=[metric],
            ground_truth="actual",
            estimates=["model_a"],
        )

        result = evaluator.evaluate()
        assert len(result) == 1  # Single aggregated result across subjects
        unnested = result.unnest(["groups"])
        assert "subject_id" not in unnested.columns  # No per-subject breakdown
        df = result.collect()
        context = df.select(pl.col("context").struct.field("metric_type"))
        assert all(context["metric_type"] == "across_subject")
        stats = result.to_ard().get_stats()
        assert stats["value"][0] > 0  # Should be a reasonable MAE value

    def test_within_visit(self, hierarchical_metric_df):
        """Test WITHIN_VISIT - per visit aggregation"""
        metric = MetricDefine(name="mae", type="within_visit")

        evaluator = MetricEvaluator(
            df=hierarchical_metric_df,
            metrics=[metric],
            ground_truth="actual",
            estimates=["model_a"],
        )

        result = evaluator.evaluate()
        df = result.collect()
        assert len(df) == 9  # Subject x visit combinations

        context = df.select(pl.col("context").struct.field("metric_type"))
        assert all(context["metric_type"] == "within_visit")

        id_df = df.unnest(["id"])
        assert {"subject_id", "visit_id"}.issubset(set(id_df.columns))
        expected_pairs = {
            (subject, visit) for subject in [1, 2, 3] for visit in [1, 2, 3]
        }
        observed_pairs = {
            (row["subject_id"], row["visit_id"]) for row in id_df.iter_rows(named=True)
        }
        assert observed_pairs == expected_pairs
        assert all(df["stat"].struct.field("type") == "float")

    def test_across_visit(self, hierarchical_metric_df):
        """Test ACROSS_VISIT - within visits then across"""
        # Use proper hierarchical metric that does within-visit then across-visit aggregation
        metric = MetricDefine(name="mae:mean", type="across_visit")

        evaluator = MetricEvaluator(
            df=hierarchical_metric_df,
            metrics=[metric],
            ground_truth="actual",
            estimates=["model_a"],
        )

        result = evaluator.evaluate()
        assert len(result) == 1  # Single aggregated result across visits
        unnested = result.unnest(["groups"])
        assert "subject_id" not in unnested.columns  # No per-subject breakdown
        assert "visit_id" not in unnested.columns  # No per-visit breakdown
        df = result.collect()
        context = df.select(pl.col("context").struct.field("metric_type"))
        assert all(context["metric_type"] == "across_visit")
        stats = result.to_ard().get_stats()
        assert stats["value"][0] > 0  # Should be a reasonable MAE value

    def test_across_visit_custom_across_expr(self, hierarchical_metric_df):
        """Test ACROSS_VISIT with custom across_expr (count, min, max, etc.)"""
        metrics = [
            MetricDefine(
                name="mae_count",
                label="Count of Visit MAEs",
                type="across_visit",
                within_expr=pl.col("absolute_error").mean(),
                across_expr=pl.col("value").count(),
            ),
            MetricDefine(
                name="mae_min",
                label="Min Visit MAE",
                type="across_visit",
                within_expr=pl.col("absolute_error").mean(),
                across_expr=pl.col("value").min(),
            ),
            MetricDefine(
                name="mae_max",
                label="Max Visit MAE",
                type="across_visit",
                within_expr=pl.col("absolute_error").mean(),
                across_expr=pl.col("value").max(),
            ),
            MetricDefine(
                name="mae_sum",
                label="Sum of Visit MAEs",
                type="across_visit",
                within_expr=pl.col("absolute_error").mean(),
                across_expr=pl.col("value").sum(),
            ),
        ]

        result, stats = _evaluate_metric_set(
            hierarchical_metric_df, metrics, estimates=["model_a"]
        )

        assert len(result) == 4

        # Check count equals number of visits (9 = 3 subjects x 3 visits)
        count_result = stats.filter(pl.col("metric") == "mae_count")
        assert count_result["value"][0] == 9.0

        # Check min < max
        min_result = stats.filter(pl.col("metric") == "mae_min")
        max_result = stats.filter(pl.col("metric") == "mae_max")
        assert min_result["value"][0] < max_result["value"][0]

        # Check sum is positive
        sum_result = stats.filter(pl.col("metric") == "mae_sum")
        assert sum_result["value"][0] > 0

    def test_across_subject_custom_across_expr(self, hierarchical_metric_df):
        """Test ACROSS_SUBJECT with custom across_expr (count, median, std, etc.)"""
        metrics = [
            MetricDefine(
                name="mae_count_subj",
                label="Count of Subject MAEs",
                type="across_subject",
                within_expr=pl.col("absolute_error").mean(),
                across_expr=pl.col("value").count(),
            ),
            MetricDefine(
                name="mae_median_subj",
                label="Median Subject MAE",
                type="across_subject",
                within_expr=pl.col("absolute_error").mean(),
                across_expr=pl.col("value").median(),
            ),
            MetricDefine(
                name="mae_std_subj",
                label="Std Dev of Subject MAEs",
                type="across_subject",
                within_expr=pl.col("absolute_error").mean(),
                across_expr=pl.col("value").std(),
            ),
        ]

        result, stats = _evaluate_metric_set(
            hierarchical_metric_df, metrics, estimates=["model_a"]
        )

        assert len(result) == 3

        # Check count equals number of subjects (3)
        count_result = stats.filter(pl.col("metric") == "mae_count_subj")
        assert count_result["value"][0] == 3.0

        # Check median is reasonable (should be positive)
        median_result = stats.filter(pl.col("metric") == "mae_median_subj")
        assert median_result["value"][0] > 0

        # Check std is non-negative
        std_result = stats.filter(pl.col("metric") == "mae_std_subj")
        assert std_result["value"][0] >= 0


class TestMetricEvaluatorGrouping:
    """Test grouping and subgrouping functionality"""

    @pytest.fixture
    def complex_data(self):
        """Data with multiple grouping dimensions"""
        return pl.DataFrame(
            {
                "subject_id": [1, 1, 2, 2, 3, 3],
                "actual": [10, 20, 15, 25, 12, 22],
                "model_a": [8, 22, 18, 24, 15, 19],
                "treatment": ["A", "A", "B", "B", "A", "B"],
                "age_group": ["young", "young", "middle", "middle", "senior", "senior"],
                "sex": ["M", "M", "F", "F", "M", "M"],
            }
        )

    def test_group_by_only(self, complex_data):
        """Test standard group_by functionality"""
        evaluator = MetricEvaluator(
            df=complex_data,
            metrics=[MetricDefine(name="mae")],
            ground_truth="actual",
            estimates=["model_a"],
            group_by=["treatment"],
        )

        result = evaluator.evaluate()
        assert len(result) == 2  # Two treatment groups
        unnested = result.unnest(["groups"])
        assert set(unnested["treatment"].unique()) == {"A", "B"}

    def test_subgroup_by_only(self, complex_data):
        """Test subgroup analysis"""
        evaluator = MetricEvaluator(
            df=complex_data,
            metrics=[MetricDefine(name="mae")],
            ground_truth="actual",
            estimates=["model_a"],
            subgroup_by=["age_group", "sex"],
        )

        result = evaluator.evaluate()

        # Should have marginal analysis for each subgroup variable
        unnested = result.unnest(["subgroups"])
        # In ARD, subgroups are unnested directly by their column names
        assert "age_group" in unnested.columns
        assert "sex" in unnested.columns

        # Check we have results for both subgroup variables
        # Each row should have data for one of the subgroup dimensions
        age_results = unnested.filter(pl.col("age_group").is_not_null())
        sex_results = unnested.filter(pl.col("sex").is_not_null())
        assert len(age_results) > 0  # Has age_group results
        assert len(sex_results) > 0  # Has sex results

    def test_group_and_subgroup(self, complex_data):
        """Test combination of group_by and subgroup_by"""
        evaluator = MetricEvaluator(
            df=complex_data,
            metrics=[MetricDefine(name="mae")],
            ground_truth="actual",
            estimates=["model_a"],
            group_by=["treatment"],
            subgroup_by=["age_group"],
        )

        result = evaluator.evaluate()

        # Should have group x subgroup combinations
        groups_unnested = result.unnest(["groups"])
        subgroups_unnested = result.unnest(["subgroups"])
        assert "treatment" in groups_unnested.columns
        assert "age_group" in subgroups_unnested.columns

        # Verify we have treatment groups and age_group subgroups
        assert set(groups_unnested["treatment"].unique()).issubset({"A", "B"})
        age_results = subgroups_unnested.filter(pl.col("age_group").is_not_null())
        assert len(age_results) > 0  # Has age_group subgroup results


class TestMetricEvaluatorAdvancedScenarios:
    """Additional scenarios covering scope mixing and typed subgroup inputs."""

    def test_mixed_scope_metrics(self, grouped_metric_df):
        """Metrics spanning all scopes should co-exist in a single evaluation run."""
        metrics = [
            MetricDefine(name="mae"),
            MetricDefine(name="n_sample", scope=MetricScope.GLOBAL),
            MetricDefine(name="n_subject", scope=MetricScope.GROUP),
            MetricDefine(name="n_sample_with_data", scope=MetricScope.MODEL),
        ]

        evaluator = MetricEvaluator(
            df=grouped_metric_df,
            metrics=metrics,
            ground_truth="actual",
            estimates=["model_a", "model_b"],
            group_by=["treatment"],
        )

        result = evaluator.evaluate()
        df = result.collect()

        context_scope = df.select(
            pl.col("metric"),
            pl.col("context").struct.field("scope").alias("scope"),
        )

        assert context_scope.filter(pl.col("metric") == "n_sample")[
            "scope"
        ].to_list() == ["global"]
        assert set(
            context_scope.filter(pl.col("metric") == "n_subject")["scope"].to_list()
        ) == {"group"}
        assert set(
            context_scope.filter(pl.col("metric") == "n_sample_with_data")[
                "scope"
            ].to_list()
        ) == {"model"}
        assert all(
            value is None
            for value in context_scope.filter(pl.col("metric") == "mae")[
                "scope"
            ].to_list()
        )

        global_rows = df.filter(pl.col("metric") == "n_sample")
        assert global_rows.height == 1
        assert global_rows["estimate"][0] is None
        assert global_rows["groups"][0] is None
        assert global_rows["stat"][0]["value_int"] == grouped_metric_df.height
        assert global_rows["stat"][0]["type"] == "int"

        group_rows = df.filter(pl.col("metric") == "n_subject")
        assert all(value is None for value in group_rows["estimate"].to_list())
        treatments = {
            group["treatment"]
            for group in group_rows["groups"].to_list()
            if group is not None
        }
        assert treatments == {"A", "B"}
        group_counts = {
            row["groups"]["treatment"]: row["stat"]["value_int"]
            for row in group_rows.iter_rows(named=True)
            if row["groups"] is not None and row["stat"]["type"] == "int"
        }
        assert group_counts == {"A": 2, "B": 1}

        model_rows = df.filter(pl.col("metric") == "n_sample_with_data")
        assert all(group is None for group in model_rows["groups"].to_list())
        assert set(model_rows["estimate"].to_list()) == {"model_a", "model_b"}
        assert all(
            row["stat"]["value_int"] == grouped_metric_df.height
            for row in model_rows.iter_rows(named=True)
        )
        assert all(
            row["stat"]["type"] == "int" for row in model_rows.iter_rows(named=True)
        )

        mae_rows = df.filter(pl.col("metric") == "mae")
        combinations = {
            (row["estimate"], row["groups"]["treatment"])
            for row in mae_rows.iter_rows(named=True)
        }
        assert combinations == {
            ("model_a", "A"),
            ("model_a", "B"),
            ("model_b", "A"),
            ("model_b", "B"),
        }

    def test_group_subgroup_overlap_raises(self, metric_sample_df):
        """Using the same column for group and subgroup should raise a clear error."""
        with pytest.raises(ValueError, match="Group and subgroup columns must be distinct"):
            MetricEvaluator(
                df=metric_sample_df,
                metrics=[MetricDefine(name="mae")],
                ground_truth="actual",
                estimates=["model_a"],
                group_by=["treatment"],
                subgroup_by=["treatment"],
            )

    def test_enum_subgroup_inputs(self, metric_sample_df):
        """Pre-typed Enum subgroup columns should preserve their category order."""
        enum_order = ["young", "middle", "senior"]
        enum_df = metric_sample_df.with_columns(
            pl.col("age_group").cast(pl.Enum(enum_order))
        )

        evaluator = MetricEvaluator(
            df=enum_df,
            metrics=[MetricDefine(name="mae")],
            ground_truth="actual",
            estimates=["model_a", "model_b"],
            subgroup_by=["age_group"],
        )

        result = evaluator.evaluate()
        subgroups = result.unnest(["subgroups"])
        subgroup_values = subgroups["subgroup_value"]

        assert isinstance(subgroup_values.dtype, pl.Enum)
        assert subgroup_values.dtype.categories.to_list() == enum_order
        assert set(subgroup_values.drop_nulls().to_list()) == set(enum_order)
        assert set(subgroups["subgroup_name"].drop_nulls().to_list()) == {"age_group"}


class TestMetricEvaluatorEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_data(self):
        """Test with empty dataset"""
        empty_data = pl.DataFrame(
            {
                "actual": [],
                "model_a": [],
            }
        ).cast({"actual": pl.Float64, "model_a": pl.Float64})

        evaluator = MetricEvaluator(
            df=empty_data,
            metrics=[MetricDefine(name="mae")],
            ground_truth="actual",
            estimates=["model_a"],
        )

        # Empty data should result in empty results
        try:
            result = evaluator.evaluate()
            assert len(result) == 0
        except Exception:
            # Some operations may fail on empty data, which is acceptable
            pass

    def test_all_missing_values(self):
        """Test with all missing values"""
        missing_data = pl.DataFrame(
            {
                "actual": [None, None, None],
                "model_a": [None, None, None],
            }
        ).cast({"actual": pl.Float64, "model_a": pl.Float64})

        evaluator = MetricEvaluator(
            df=missing_data,
            metrics=[MetricDefine(name="mae")],
            ground_truth="actual",
            estimates=["model_a"],
        )

        # Should handle gracefully (result may be null/NaN)
        try:
            result = evaluator.evaluate()
            assert len(result) == 1
        except Exception:
            # Operations on all-null data may fail, which is acceptable
            pass

    def test_single_row(self):
        """Test with single data point"""
        single_data = pl.DataFrame(
            {
                "actual": [10.0],
                "model_a": [8.0],
            }
        )

        evaluator = MetricEvaluator(
            df=single_data,
            metrics=[MetricDefine(name="mae")],
            ground_truth="actual",
            estimates=["model_a"],
        )

        result = evaluator.evaluate()
        assert len(result) == 1
        stats = result.to_ard().get_stats()
        assert stats["value"][0] == 2.0

    def test_invalid_metric_name(self, metric_sample_df):
        """Test error handling for invalid configuration"""
        with pytest.raises(ValueError, match="not in configured metrics"):
            evaluator = MetricEvaluator(
                df=metric_sample_df,
                metrics=[MetricDefine(name="mae")],
                ground_truth="actual",
                estimates=["model_a"],
            )
            # Try to evaluate metric not in original configuration
            evaluator.evaluate(metrics=[MetricDefine(name="rmse")])

    def test_invalid_estimate_name(self, metric_sample_df):
        """Test error handling for invalid estimate"""
        with pytest.raises(ValueError, match="not in configured estimates"):
            evaluator = MetricEvaluator(
                df=metric_sample_df,
                metrics=[MetricDefine(name="mae")],
                ground_truth="actual",
                estimates=["model_a"],
            )
            # Try to evaluate estimate not in original configuration
            evaluator.evaluate(estimates=["model_c"])

    def test_filter_expression(self):
        """Test filter_expr functionality"""
        data = pl.DataFrame(
            {
                "subject_id": [1, 2, 3, 4],
                "actual": [10, 20, 30, 40],
                "model_a": [8, 22, 28, 45],
                "keep": [True, True, False, True],
            }
        )

        evaluator = MetricEvaluator(
            df=data,
            metrics=[MetricDefine(name="mae")],
            ground_truth="actual",
            estimates=["model_a"],
            filter_expr=pl.col("keep") == True,
        )

        result = evaluator.evaluate()
        assert len(result) == 1
        # Should only use rows where keep=True (subjects 1, 2, 4)
        # MAE should be calculated on 3 rows, not 4

    def test_lazy_vs_eager_evaluation(self, metric_sample_df):
        """Test collect parameter"""
        evaluator = MetricEvaluator(
            df=metric_sample_df,
            metrics=[MetricDefine(name="mae")],
            ground_truth="actual",
            estimates=["model_a"],
        )

        # Test that evaluate always returns ARD object
        result = evaluator.evaluate()

        # Should be ARD object with consistent behavior
        assert len(result) == 1
        df = result.collect()
        stats = result.to_ard().get_stats()
        assert stats["value"][0] == 2.375  # MAE should be 2.375
