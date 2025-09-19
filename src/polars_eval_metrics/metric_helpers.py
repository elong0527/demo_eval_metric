"""
Helper functions for creating metrics from various sources.

Simple, functional approach to metric creation without factory classes.
"""

# pyre-strict

from typing import Any

from .metric_define import MetricDefine


def create_metric_from_dict(config: dict[str, Any]) -> MetricDefine:
    """Create a MetricDefine from dictionary configuration."""
    # Validate configuration before creation
    _validate_metric_config(config)

    # Pure creation logic
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


# ========================================
# VALIDATION FUNCTIONS - Centralized Logic
# ========================================


def _validate_metric_config(config: dict[str, Any]) -> None:
    """Validate metric configuration dictionary"""
    if "name" not in config:
        raise ValueError("Metric configuration must include 'name'")

    if not isinstance(config["name"], str) or not config["name"].strip():
        raise ValueError("Metric name must be a non-empty string")

    # Validate type if provided
    if "type" in config and config["type"] is not None:
        valid_types = [
            "across_sample",
            "within_subject",
            "across_subject",
            "within_visit",
            "across_visit",
        ]
        if isinstance(config["type"], str) and config["type"] not in valid_types:
            raise ValueError(
                f"Invalid metric type: '{config['type']}'. Valid options: {valid_types}"
            )

    # Validate scope if provided
    if "scope" in config and config["scope"] is not None:
        valid_scopes = ["global", "model", "group"]
        if isinstance(config["scope"], str) and config["scope"] not in valid_scopes:
            raise ValueError(
                f"Invalid metric scope: '{config['scope']}'. Valid options: {valid_scopes}"
            )
