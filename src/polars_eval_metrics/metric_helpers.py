from __future__ import annotations

"""
Helper functions for creating metrics from various sources.

Simple, functional approach to metric creation without factory classes.
"""

# pyre-strict

from typing import TYPE_CHECKING, Any

from .utils import parse_enum_value


if TYPE_CHECKING:
    from .metric_define import MetricDefine, MetricScope, MetricType


def create_metric_from_dict(config: dict[str, Any]) -> MetricDefine:
    """Create a MetricDefine from dictionary configuration."""

    _validate_metric_config(config)

    from .metric_define import MetricDefine

    return MetricDefine(
        name=config["name"],
        label=config.get("label", config["name"]),
        type=config.get("type", "across_sample"),
        scope=config.get("scope"),
        within_expr=config.get("within_expr"),
        across_expr=config.get("across_expr"),
    )


def create_metrics(configs: list[dict[str, Any]] | list[str]) -> list[MetricDefine]:
    """Create metrics from configurations or simple names."""
    if not configs:
        return []

    from .metric_define import MetricDefine

    if isinstance(configs[0], str):
        str_configs: list[str] = configs  # pyre-ignore[9]
        return [MetricDefine(name=name) for name in str_configs]

    dict_configs: list[dict[str, Any]] = configs  # pyre-ignore[9]
    return [create_metric_from_dict(config) for config in dict_configs]


# ========================================
# VALIDATION FUNCTIONS - Centralized Logic
# ========================================


def _validate_metric_config(config: dict[str, Any]) -> None:
    """Validate metric configuration dictionary."""
    if "name" not in config:
        raise ValueError("Metric configuration must include 'name'")

    if not isinstance(config["name"], str) or not config["name"].strip():
        raise ValueError("Metric name must be a non-empty string")

    if "type" in config and config["type"] is not None:
        from .metric_define import MetricType

        parse_enum_value(
            config["type"],
            MetricType,
            field="metric type",
        )

    if "scope" in config and config["scope"] is not None:
        from .metric_define import MetricScope

        parse_enum_value(
            config["scope"],
            MetricScope,
            field="metric scope",
            allow_none=True,
        )
