"""
Helper functions for creating metrics from various sources.

Simple, functional approach to metric creation without factory classes.
"""

# pyre-strict

from typing import Any

from .metric_define import MetricDefine


def create_metric_from_dict(config: dict[str, Any]) -> MetricDefine:
    """Create a MetricDefine from dictionary configuration."""
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

    if isinstance(configs[0], str):
        # Type narrow to list[str]
        str_configs: list[str] = configs  # pyre-ignore[9]
        return [MetricDefine(name=name) for name in str_configs]
    else:
        # Type narrow to list[dict[str, Any]]
        dict_configs: list[dict[str, Any]] = configs  # pyre-ignore[9]
        return [create_metric_from_dict(config) for config in dict_configs]
