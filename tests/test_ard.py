"""Tests for ARD (Analysis Results Data) structure."""

import polars as pl

from polars_eval_metrics.ard import ARD


class TestARDBasics:
    """Test basic ARD functionality."""

    def test_empty_ard(self):
        """Test creating empty ARD."""
        ard = ARD()
        assert len(ard) == 0
        assert ard.collect().shape == (0, 6)

    def test_empty_get_stats(self):
        """get_stats should work on empty ARD with fixed schema."""
        ard = ARD()
        values = ard.get_stats()
        assert values.shape == (0, 2)

    def test_from_records(self):
        """Test creating ARD from records."""
        records = [
            {
                "groups": {"treatment": "A", "site": "01"},
                "subgroups": None,
                "estimate": "model1",
                "metric": "mae",
                "stat": 3.2,
                "context": {"n": 100},
            },
            {
                "groups": {"treatment": "B", "site": "01"},
                "subgroups": {"gender": "M"},
                "estimate": "model1",
                "metric": "rmse",
                "stat": {"value": 4.5, "type": "float"},
                "context": {"n": 85},
            },
        ]

        ard = ARD(records)
        assert len(ard) == 2

        df = ard.collect()
        assert "groups" in df.columns
        assert "stat" in df.columns

    def test_filter_by_groups(self):
        """Test filtering by group values."""
        records = [
            {"groups": {"trt": "A"}, "metric": "mae", "stat": 3.2},
            {"groups": {"trt": "B"}, "metric": "mae", "stat": 2.8},
            {"groups": {"trt": "A"}, "metric": "rmse", "stat": 4.1},
        ]

        ard = ARD(records)
        filtered = ard.filter(groups={"trt": "A"})
        assert len(filtered) == 2

    def test_filter_by_metrics(self):
        """Test filtering by metrics."""
        records = [
            {"metric": "mae", "stat": 3.2},
            {"metric": "rmse", "stat": 4.5},
            {"metric": "mae", "stat": 2.8},
        ]

        ard = ARD(records)
        filtered = ard.filter(metrics="mae")
        assert len(filtered) == 2

        filtered = ard.filter(metrics=["mae", "rmse"])
        assert len(filtered) == 3

    def test_filter_missing_group_key(self):
        """Filtering by a group key that does not exist returns empty ARD."""
        records = [
            {"groups": {"trt": "A"}, "metric": "mae", "stat": 3.2},
            {"groups": {"trt": "B"}, "metric": "rmse", "stat": 4.5},
        ]

        ard = ARD(records)
        filtered = ard.filter(groups={"unknown": "value"})
        assert len(filtered) == 0

    def test_filter_subgroups_and_context(self):
        """Filtering should support subgroups and context keys."""
        records = [
            {
                "subgroups": {"gender": "F"},
                "context": {"fold": "1"},
                "metric": "mae",
                "stat": 1.2,
            },
            {
                "subgroups": {"gender": "M"},
                "context": {"fold": "2"},
                "metric": "rmse",
                "stat": 2.5,
            },
        ]

        ard = ARD(records)
        sub_filtered = ard.filter(subgroups={"gender": "F"})
        assert len(sub_filtered) == 1
        ctx_filtered = ard.filter(context={"fold": "2"})
        assert len(ctx_filtered) == 1


class TestARDTransformations:
    """Test ARD transformation methods."""

    def test_unnest_groups(self):
        """Test unnesting group columns."""
        records = [
            {"groups": {"treatment": "A", "site": "01"}, "metric": "mae", "stat": 3.2}
        ]

        ard = ARD(records)
        df = ard.unnest(["groups"])

        assert "treatment" in df.columns
        assert "site" in df.columns
        assert df["treatment"][0] == "A"

    def test_struct_harmonization(self):
        """Records with different group keys should harmonize struct schema."""
        records = [
            {
                "groups": {"trt": "A"},
                "metric": "mae",
                "stat": 3.2,
            },
            {
                "groups": {"site": "01"},
                "metric": "rmse",
                "stat": 4.5,
            },
            {
                "groups": None,
                "metric": "rmse",
                "stat": 4.1,
            },
        ]

        ard = ARD(records)
        df = ard.unnest(["groups"])

        assert "trt" in df.columns
        assert "site" in df.columns
        null_row = df.filter(pl.col("trt").is_null() & pl.col("site").is_null())
        assert null_row.height == 1

    def test_to_wide(self):
        """Test pivot to wide format."""
        records = [
            {"groups": {"trt": "A"}, "metric": "mae", "stat": 3.2},
            {"groups": {"trt": "A"}, "metric": "rmse", "stat": 4.1},
            {"groups": {"trt": "B"}, "metric": "mae", "stat": 2.8},
            {"groups": {"trt": "B"}, "metric": "rmse", "stat": 3.9},
        ]

        ard = ARD(records)
        wide = ard.to_wide()

        assert "mae" in wide.columns or '["mae"]' in str(wide.columns)
        assert wide.shape[0] == 2  # Two treatment groups

    def test_null_handling(self):
        """Test null to empty conversion."""
        records = [
            {"groups": None, "estimate": None, "metric": "mae", "stat": 3.2},
        ]

        ard = ARD(records)
        df = ard.collect()

        # Check initial state is null
        assert df["groups"][0] is None
        assert df["estimate"][0] is None

        # Convert null to empty
        with_empty = ard.with_null_as_empty()
        df_empty = with_empty.collect()
        # Note: Polars struct fill_null behavior may vary
        assert df_empty["estimate"][0] == ""

        # Convert empty back to null
        with_null = ard.with_empty_as_null()
        df_null = with_null.collect()
        assert df_null["estimate"][0] is None or df_null["estimate"][0] == ""

    def test_null_struct_round_trip(self):
        """Ensure null struct columns round-trip through empty/null helpers."""
        records = [
            {
                "groups": {"trt": "A", "site": "01"},
                "metric": "mae",
                "stat": 3.2,
            },
            {
                "groups": None,
                "metric": "rmse",
                "stat": 4.5,
            },
        ]

        ard = ARD(records)

        filled = ard.with_null_as_empty().unnest(["groups"])
        assert "trt" in filled.columns
        assert filled["trt"].null_count() == 1

        collected = ard.with_null_as_empty().with_empty_as_null().collect()
        assert collected["groups"][1] is None
        assert collected["context"][1] is None


class TestARDStatHandling:
    """Test stat value handling."""

    def test_stat_normalization(self):
        """Test different stat value types."""
        records = [
            {"metric": "count", "stat": 42},  # int
            {"metric": "mean", "stat": 3.14159},  # float
            {"metric": "label", "stat": "significant"},  # string
            {"metric": "flag", "stat": True},  # bool
            {"metric": "ci", "stat": {"lower": 2.5, "upper": 3.5}},  # struct
        ]

        ard = ARD(records)
        df = ard.collect()

        # All should be converted to stat struct
        assert df["stat"][0]["type"] == "int"
        assert df["stat"][1]["type"] == "float"
        assert df["stat"][2]["type"] == "string"
        assert df["stat"][3]["type"] == "bool"
        assert df["stat"][4]["type"] == "struct"

    def test_get_stats(self):
        """Test extracting stat values."""
        records = [
            {"metric": "mae", "stat": 3.2},
            {"metric": "rmse", "stat": 4.5},
        ]

        ard = ARD(records)

        # Get just values
        values = ard.get_stats(as_values=True)
        assert "value" in values.columns
        assert values["value"][0] == 3.2

        # Get with metadata
        with_meta = ard.get_stats(include_metadata=True)
        assert "type" in with_meta.columns
        assert "format" in with_meta.columns


class TestARDDisplay:
    """Test display and printing methods."""

    def test_repr(self):
        """Test string representation."""
        records = [
            {"groups": {"trt": "A"}, "metric": "mae", "stat": 3.2},
        ]

        ard = ARD(records)
        repr_str = repr(ard)

        assert "ARD: 1 results" in repr_str
        assert "Schema:" in repr_str

    def test_summary(self):
        """Test summary statistics."""
        records = [
            {"groups": {"trt": "A"}, "estimate": "m1", "metric": "mae", "stat": 3.2},
            {"groups": {"trt": "B"}, "estimate": "m1", "metric": "mae", "stat": 2.8},
            {"groups": {"trt": "A"}, "estimate": "m2", "metric": "rmse", "stat": 4.1},
        ]

        ard = ARD(records)
        summary = ard.summary()

        assert summary["n_rows"] == 3
        assert summary["n_metrics"] == 2
        assert summary["n_estimates"] == 2
        assert "mae" in summary["metrics"]
        assert "rmse" in summary["metrics"]
