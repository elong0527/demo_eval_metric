"""Tests for ARD (Analysis Results Data) structure."""

from __future__ import annotations

from typing import Any, Iterable

import polars as pl
import pytest

from polars_eval_metrics.ard import ARD


def stat_struct(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "type": None,
        "value_float": None,
        "value_int": None,
        "value_bool": None,
        "value_str": None,
        "value_struct": None,
        "format": None,
    }
    base.update(overrides)
    return base


def dataset(rows: Iterable[dict[str, Any]]) -> pl.DataFrame:
    normalised: list[dict[str, Any]] = []
    for row in rows:
        base = {
            "groups": None,
            "subgroups": None,
            "estimate": None,
            "metric": None,
            "label": None,
            "stat": stat_struct(),
            "context": None,
            "id": None,
        }
        base.update(row)
        normalised.append(base)
    return pl.DataFrame(normalised)


class TestARDBasics:
    """Test basic ARD functionality."""

    def test_empty_ard(self) -> None:
        ard = ARD()
        assert len(ard) == 0
        assert ard.collect().shape == (0, 8)

    def test_empty_get_stats(self) -> None:
        ard = ARD()
        values = ard.get_stats()
        assert values.shape == (0, 2)

    def test_from_frame(self) -> None:
        df = dataset(
            [
                {
                    "groups": {"treatment": "A", "site": "01"},
                    "subgroups": {"gender": None},
                    "estimate": "model1",
                    "metric": "mae",
                    "stat": stat_struct(type="float", value_float=3.2),
                    "context": {"n": 100},
                },
                {
                    "groups": {"treatment": "B", "site": "01"},
                    "subgroups": {"gender": "M"},
                    "estimate": "model1",
                    "metric": "rmse",
                    "stat": stat_struct(type="float", value_float=4.5),
                    "context": {"n": 85},
                },
            ]
        )

        ard = ARD(df)
        assert len(ard) == 2

        collected = ard.collect()
        assert "groups" in collected.columns
        assert "stat" in collected.columns

    def test_filter_by_groups(self) -> None:
        df = dataset(
            [
                {
                    "groups": {"trt": "A"},
                    "metric": "mae",
                    "stat": stat_struct(type="float", value_float=3.2),
                },
                {
                    "groups": {"trt": "B"},
                    "metric": "mae",
                    "stat": stat_struct(type="float", value_float=2.8),
                },
                {
                    "groups": {"trt": "A"},
                    "metric": "rmse",
                    "stat": stat_struct(type="float", value_float=4.1),
                },
            ]
        )

        ard = ARD(df)
        filtered = ARD(ard.lazy.filter(pl.col("groups").struct.field("trt") == "A"))
        assert len(filtered) == 2

    def test_filter_by_metrics(self) -> None:
        df = dataset(
            [
                {
                    "metric": "mae",
                    "stat": stat_struct(type="float", value_float=3.2),
                },
                {
                    "metric": "rmse",
                    "stat": stat_struct(type="float", value_float=4.5),
                },
                {
                    "metric": "mae",
                    "stat": stat_struct(type="float", value_float=2.8),
                },
            ]
        )

        ard = ARD(df)
        filtered = ARD(ard.lazy.filter(pl.col("metric") == "mae"))
        assert len(filtered) == 2

        filtered_all = ARD(ard.lazy.filter(pl.col("metric").is_in(["mae", "rmse"])))
        assert len(filtered_all) == 3

    def test_filter_missing_group_key(self) -> None:
        df = dataset(
            [
                {
                    "groups": {"trt": "A"},
                    "metric": "mae",
                    "stat": stat_struct(type="float", value_float=3.2),
                },
                {
                    "groups": {"trt": "B"},
                    "metric": "rmse",
                    "stat": stat_struct(type="float", value_float=4.5),
                },
            ]
        )

        ard = ARD(df)
        with pytest.raises(pl.exceptions.StructFieldNotFoundError):
            ard.lazy.filter(
                pl.col("groups").struct.field("unknown") == "value"
            ).collect()

    def test_filter_subgroups_and_context(self) -> None:
        df = dataset(
            [
                {
                    "subgroups": {"gender": "F"},
                    "context": {"fold": "1"},
                    "metric": "mae",
                    "stat": stat_struct(type="float", value_float=1.2),
                },
                {
                    "subgroups": {"gender": "M"},
                    "context": {"fold": "2"},
                    "metric": "rmse",
                    "stat": stat_struct(type="float", value_float=2.5),
                },
            ]
        )

        ard = ARD(df)
        sub_filtered = ARD(
            ard.lazy.filter(pl.col("subgroups").struct.field("gender") == "F")
        )
        assert len(sub_filtered) == 1
        ctx_filtered = ARD(
            ard.lazy.filter(pl.col("context").struct.field("fold") == "2")
        )
        assert len(ctx_filtered) == 1


class TestARDTransformations:
    """Test ARD transformation methods."""

    def test_unnest_groups(self) -> None:
        df = dataset(
            [
                {
                    "groups": {"treatment": "A", "site": "01"},
                    "metric": "mae",
                    "stat": stat_struct(type="float", value_float=3.2),
                }
            ]
        )

        ard = ARD(df)
        unnested = ard.unnest(["groups"])

        assert "treatment" in unnested.columns
        assert "site" in unnested.columns
        assert unnested["treatment"][0] == "A"

    def test_struct_harmonization(self) -> None:
        df = dataset(
            [
                {
                    "groups": {"trt": "A", "site": None},
                    "metric": "mae",
                    "stat": stat_struct(type="float", value_float=3.2),
                },
                {
                    "groups": {"trt": None, "site": "01"},
                    "metric": "rmse",
                    "stat": stat_struct(type="float", value_float=4.5),
                },
                {
                    "groups": {"trt": None, "site": None},
                    "metric": "rmse",
                    "stat": stat_struct(type="float", value_float=4.1),
                },
            ]
        )

        ard = ARD(df)
        unnested = ard.unnest(["groups"])

        assert {"trt", "site"}.issubset(set(unnested.columns))
        null_row = unnested.filter(pl.col("trt").is_null() & pl.col("site").is_null())
        assert null_row.height == 1

    def test_unnest_comprehensive(self) -> None:
        df = dataset(
            [
                {
                    "groups": {"trt": "A"},
                    "metric": "mae",
                    "stat": stat_struct(type="float", value_float=3.2),
                },
                {
                    "groups": {"trt": "A"},
                    "metric": "rmse",
                    "stat": stat_struct(type="float", value_float=4.1),
                },
                {
                    "groups": {"trt": "B"},
                    "metric": "mae",
                    "stat": stat_struct(type="float", value_float=2.8),
                },
                {
                    "groups": {"trt": "B"},
                    "metric": "rmse",
                    "stat": stat_struct(type="float", value_float=3.9),
                },
            ]
        )

        ard = ARD(df)
        unnested = ard.unnest(["groups"])

        assert "trt" in unnested.columns
        assert unnested.shape[0] == 4

    def test_null_handling(self) -> None:
        df = dataset(
            [
                {
                    "groups": None,
                    "estimate": None,
                    "metric": "mae",
                    "stat": stat_struct(type="float", value_float=3.2),
                }
            ]
        )

        ard = ARD(df)
        collected = ard.collect()
        assert collected["groups"][0] is None
        assert collected["estimate"][0] is None

        with_empty = ard.with_null_as_empty()
        df_empty = with_empty.collect()
        assert df_empty["estimate"][0] == ""

        with_null = ard.with_empty_as_null()
        df_null = with_null.collect()
        assert df_null["estimate"][0] in (None, "")

    def test_null_struct_round_trip(self) -> None:
        df = dataset(
            [
                {
                    "groups": {"trt": "A", "site": "01"},
                    "metric": "mae",
                    "stat": stat_struct(type="float", value_float=3.2),
                },
                {
                    "groups": None,
                    "metric": "rmse",
                    "stat": stat_struct(type="float", value_float=4.5),
                },
            ]
        )

        ard = ARD(df)
        filled = ard.with_null_as_empty().unnest(["groups"])
        assert "trt" in filled.columns
        assert filled["trt"].null_count() == 1

        round_tripped = ard.with_null_as_empty().with_empty_as_null().collect()
        assert round_tripped["groups"][1] is None
        assert round_tripped["context"][1] is None


class TestARDStatHandling:
    """Test stat value handling."""

    def test_stat_struct_contents(self) -> None:
        df = dataset(
            [
                {
                    "metric": "count",
                    "stat": stat_struct(type="int", value_int=42),
                },
                {
                    "metric": "mean",
                    "stat": stat_struct(type="float", value_float=3.14159),
                },
                {
                    "metric": "label",
                    "stat": stat_struct(type="string", value_str="significant"),
                },
                {
                    "metric": "flag",
                    "stat": stat_struct(type="bool", value_bool=True),
                },
                {
                    "metric": "ci",
                    "stat": stat_struct(
                        type="struct", value_struct={"lower": 2.5, "upper": 3.5}
                    ),
                },
            ]
        )

        ard = ARD(df)
        collected = ard.collect()

        assert collected["stat"][0]["value_int"] == 42
        assert collected["stat"][1]["value_float"] == 3.14159
        assert collected["stat"][2]["value_str"] == "significant"
        assert collected["stat"][3]["value_bool"] is True
        assert collected["stat"][4]["value_struct"] == {"lower": 2.5, "upper": 3.5}

    def test_get_stats(self) -> None:
        df = dataset(
            [
                {
                    "metric": "mae",
                    "stat": stat_struct(type="float", value_float=3.2),
                },
                {
                    "metric": "rmse",
                    "stat": stat_struct(type="float", value_float=4.5),
                },
            ]
        )

        ard = ARD(df)
        values = ard.get_stats()
        assert "value" in values.columns
        assert values["value"][0] == 3.2

        with_meta = ard.get_stats(include_metadata=True)
        assert "type" in with_meta.columns
        assert "format" in with_meta.columns


class TestARDDisplay:
    """Test display and printing methods."""

    def test_repr(self) -> None:
        df = dataset(
            [
                {
                    "groups": {"trt": "A"},
                    "metric": "mae",
                    "stat": stat_struct(type="float", value_float=3.2),
                }
            ]
        )

        ard = ARD(df)
        repr_str = repr(ard)

        assert repr_str.startswith("ARD(summary=")
        assert "'n_rows': 1" in repr_str

    def test_summary(self) -> None:
        df = dataset(
            [
                {
                    "groups": {"trt": "A"},
                    "estimate": "m1",
                    "metric": "mae",
                    "stat": stat_struct(type="float", value_float=3.2),
                },
                {
                    "groups": {"trt": "B"},
                    "estimate": "m1",
                    "metric": "mae",
                    "stat": stat_struct(type="float", value_float=2.8),
                },
                {
                    "groups": {"trt": "A"},
                    "estimate": "m2",
                    "metric": "rmse",
                    "stat": stat_struct(type="float", value_float=4.1),
                },
            ]
        )

        ard = ARD(df)
        summary = ard.summary()

        assert summary["n_rows"] == 3
        assert summary["n_metrics"] == 2
        assert summary["n_estimates"] == 2
        assert "mae" in summary["metrics"]
        assert "rmse" in summary["metrics"]
