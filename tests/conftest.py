import polars as pl
import pytest


@pytest.fixture
def metric_sample_df() -> pl.DataFrame:
    """Wide-form dataset with two model estimates and demographic groupings."""
    return pl.DataFrame(
        {
            "subject_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "visit_id": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "treatment": ["A", "A", "A", "A", "A", "A", "B", "B", "B"],
            "age_group": [
                "young",
                "young",
                "young",
                "middle",
                "middle",
                "middle",
                "senior",
                "senior",
                "senior",
            ],
            "sex": ["M", "M", "M", "F", "F", "F", "M", "M", "M"],
            "actual": [10, 20, 30, 15, 25, 35, 12, 22, 32],
            "model_a": [8, 22, 28, 18, 24, 38, None, 19, 35],
            "model_b": [12, None, None, 13, 27, 33, 14, 25, 29],
        }
    )


@pytest.fixture
def grouped_metric_df() -> pl.DataFrame:
    """Compact dataset for scope-based tests with clear treatment groups."""
    return pl.DataFrame(
        {
            "subject_id": [1, 1, 2, 2, 3, 3],
            "actual": [10, 20, 15, 25, 12, 22],
            "model_a": [8, 22, 18, 24, 15, 19],
            "model_b": [12, 18, 13, 27, 14, 25],
            "treatment": ["A", "A", "A", "A", "B", "B"],
        }
    )


@pytest.fixture
def hierarchical_metric_df() -> pl.DataFrame:
    """Hierarchical dataset with complete model coverage for every visit."""
    return pl.DataFrame(
        {
            "subject_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "visit_id": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "actual": [10, 20, 30, 15, 25, 35, 12, 22, 32],
            "model_a": [8, 22, 28, 18, 24, 38, 15, 19, 35],
        }
    )
