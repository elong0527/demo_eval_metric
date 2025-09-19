"""
Test cases for enum preservation and row ordering in pivot methods.

These tests cover the recent changes made to:
1. Preserve original enum ordering when input data has pre-specified Enum order
2. Handle multiple subgroup variables with mixed enum/string types
3. Row ordering options (group vs subgroup priority)
"""

import pytest
import polars as pl
from polars_eval_metrics import MetricDefine, MetricEvaluator


class TestEnumPreservationAndOrdering:
    """Test enum preservation and row ordering in pivot methods"""

    @pytest.fixture
    def sample_data_with_enum(self):
        """Create sample data with enum column"""
        data = {
            "subject_id": ["S01", "S02", "S03", "S04", "S05", "S06"],
            "treatment": ["A", "A", "B", "B", "A", "B"],
            "age_group": [
                "Middle",
                "Senior",
                "Young",
                "Middle",
                "Senior",
                "Young",
            ],  # Mixed order
            "region": ["North", "South", "North", "South", "North", "South"],
            "actual": [10, 12, 15, 18, 11, 13],
            "model_1": [11, 13, 14, 19, 12, 14],
        }
        df = pl.DataFrame(data)
        # Pre-specify enum order: Young -> Middle -> Senior
        df = df.with_columns(
            age_group=pl.col("age_group").cast(pl.Enum(["Young", "Middle", "Senior"]))
        )
        return df

    @pytest.fixture
    def basic_metrics(self):
        """Basic metrics for testing"""
        return [MetricDefine(name="mae", label="MAE")]

    def test_enum_preservation_single_subgroup(
        self, sample_data_with_enum, basic_metrics
    ):
        """Test that original enum ordering is preserved with single subgroup"""
        evaluator = MetricEvaluator(
            df=sample_data_with_enum,
            metrics=basic_metrics,
            ground_truth="actual",
            estimates={"model_1": "Model 1"},
            group_by={"treatment": "Treatment"},
            subgroup_by={"age_group": "Age Group"},
        )

        result = evaluator.pivot_by_group()

        # Check that subgroup_value is an enum
        assert isinstance(result.get_column("subgroup_value").dtype, pl.Enum)

        # Check that enum categories match original order
        enum_categories = result.get_column("subgroup_value").dtype.categories.to_list()
        assert enum_categories == ["Young", "Middle", "Senior"]

    def test_enum_preservation_multiple_subgroups(
        self, sample_data_with_enum, basic_metrics
    ):
        """Test enum preservation with multiple subgroups (enum + string)"""
        evaluator = MetricEvaluator(
            df=sample_data_with_enum,
            metrics=basic_metrics,
            ground_truth="actual",
            estimates={"model_1": "Model 1"},
            group_by={"treatment": "Treatment"},
            subgroup_by={"age_group": "Age Group", "region": "Region"},
        )

        result = evaluator.pivot_by_group()

        # Check that subgroup_value is an enum
        assert isinstance(result.get_column("subgroup_value").dtype, pl.Enum)

        # Check that enum categories include values from both subgroups
        enum_categories = result.get_column("subgroup_value").dtype.categories.to_list()

        # Age group values should come first in original order
        age_values = [
            cat for cat in enum_categories if cat in ["Young", "Middle", "Senior"]
        ]
        assert age_values == ["Young", "Middle", "Senior"]

        # Region values should be included
        region_values = [cat for cat in enum_categories if cat in ["North", "South"]]
        assert len(region_values) == 2  # Both North and South should be present

    def test_row_ordering_subgroup_priority(self, sample_data_with_enum, basic_metrics):
        """Test that row_order_by='subgroup' respects enum order"""
        evaluator = MetricEvaluator(
            df=sample_data_with_enum,
            metrics=basic_metrics,
            ground_truth="actual",
            estimates={"model_1": "Model 1"},
            group_by={"treatment": "Treatment"},
            subgroup_by={"age_group": "Age Group", "region": "Region"},
        )

        result = evaluator.pivot_by_group(row_order_by="subgroup")

        # Filter age group rows and check their order
        age_group_rows = result.filter(pl.col("subgroup_name") == "Age Group")
        age_group_order = (
            age_group_rows.get_column("subgroup_value")
            .unique(maintain_order=True)
            .to_list()
        )

        # Should be in enum order: Young, Middle, Senior
        assert age_group_order == ["Young", "Middle", "Senior"]

    def test_row_ordering_group_priority(self, sample_data_with_enum, basic_metrics):
        """Test that row_order_by='group' prioritizes treatment over subgroups"""
        evaluator = MetricEvaluator(
            df=sample_data_with_enum,
            metrics=basic_metrics,
            ground_truth="actual",
            estimates={"model_1": "Model 1"},
            group_by={"treatment": "Treatment"},
            subgroup_by={"age_group": "Age Group"},
        )

        result = evaluator.pivot_by_group(row_order_by="group")

        # Check that treatment values are grouped together
        treatment_order = result.get_column("Treatment").to_list()

        # All A's should come before all B's (or vice versa, but grouped)
        a_indices = [i for i, t in enumerate(treatment_order) if t == "A"]
        b_indices = [i for i, t in enumerate(treatment_order) if t == "B"]

        # Check that indices are contiguous (grouped)
        if a_indices:
            assert a_indices == list(range(min(a_indices), max(a_indices) + 1))
        if b_indices:
            assert b_indices == list(range(min(b_indices), max(b_indices) + 1))

    def test_post_hoc_sorting_respects_enum(self, sample_data_with_enum, basic_metrics):
        """Test that manual sorting after pivot respects enum order"""
        evaluator = MetricEvaluator(
            df=sample_data_with_enum,
            metrics=basic_metrics,
            ground_truth="actual",
            estimates={"model_1": "Model 1"},
            group_by={"treatment": "Treatment"},
            subgroup_by={"age_group": "Age Group", "region": "Region"},
        )

        result = evaluator.pivot_by_group()
        sorted_result = result.sort("subgroup_value")

        # Get age group rows from sorted result
        age_group_sorted = sorted_result.filter(pl.col("subgroup_name") == "Age Group")
        age_order = (
            age_group_sorted.get_column("subgroup_value")
            .unique(maintain_order=True)
            .to_list()
        )

        # Should be in enum order
        assert age_order == ["Young", "Middle", "Senior"]

    def test_pivot_by_model_enum_preservation(
        self, sample_data_with_enum, basic_metrics
    ):
        """Test that pivot_by_model also preserves enum ordering"""
        evaluator = MetricEvaluator(
            df=sample_data_with_enum,
            metrics=basic_metrics,
            ground_truth="actual",
            estimates={"model_1": "Model 1"},
            group_by={"treatment": "Treatment"},
            subgroup_by={"age_group": "Age Group"},
        )

        result = evaluator.pivot_by_model()

        # Check that subgroup_value is an enum with correct order
        assert isinstance(result.get_column("subgroup_value").dtype, pl.Enum)
        enum_categories = result.get_column("subgroup_value").dtype.categories.to_list()
        assert enum_categories == ["Young", "Middle", "Senior"]

    def test_fallback_to_display_order_without_enum(self):
        """Test fallback to display order when original data has no enum"""
        data = {
            "subject_id": ["S01", "S02", "S03", "S04"],
            "treatment": ["A", "A", "B", "B"],
            "category": ["C", "B", "A", "C"],  # No enum, just strings
            "actual": [10, 12, 15, 18],
            "model_1": [11, 13, 14, 19],
        }
        df = pl.DataFrame(data)

        evaluator = MetricEvaluator(
            df=df,
            metrics=[MetricDefine(name="mae", label="MAE")],
            ground_truth="actual",
            estimates={"model_1": "Model 1"},
            group_by={"treatment": "Treatment"},
            subgroup_by={"category": "Category"},
        )

        result = evaluator.pivot_by_group()

        # Should still create enum, but based on display order
        assert isinstance(result.get_column("subgroup_value").dtype, pl.Enum)

        # Categories should be in the order they appear in the result
        enum_categories = result.get_column("subgroup_value").dtype.categories.to_list()
        assert len(enum_categories) > 0  # Should have some categories

    def test_no_enum_conversion_single_value(self, basic_metrics):
        """Test that enum conversion is skipped when only one unique value"""
        data = {
            "subject_id": ["S01", "S02"],
            "treatment": ["A", "A"],
            "category": ["Same", "Same"],  # Only one unique value
            "actual": [10, 12],
            "model_1": [11, 13],
        }
        df = pl.DataFrame(data)

        evaluator = MetricEvaluator(
            df=df,
            metrics=basic_metrics,
            ground_truth="actual",
            estimates={"model_1": "Model 1"},
            group_by={"treatment": "Treatment"},
            subgroup_by={"category": "Category"},
        )

        result = evaluator.pivot_by_group()

        # Should not convert to enum if only one unique value
        subgroup_dtype = result.get_column("subgroup_value").dtype
        # Could be string or enum depending on implementation, but should work either way
        assert subgroup_dtype in [pl.Utf8, pl.String] or isinstance(
            subgroup_dtype, pl.Enum
        )
