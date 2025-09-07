"""
Helper functions for creating metrics from various sources.

Simple, functional approach to metric creation without factory classes.
"""

# pyre-strict

from typing import Any

from .metric_define import MetricDefine


def create_metric_from_dict(config: dict[str, Any]) -> MetricDefine:
    """
    Create a MetricDefine from dictionary configuration.

    Handles YAML-style nested structures.

    Args:
        config: Dictionary with metric configuration

    Returns:
        MetricDefine instance

    Examples:
        # Single expression
        config = {
            'name': 'mae',
            'label': 'Mean Absolute Error',
            'type': 'across_sample',
            'within_expr': 'mae',  # Single built-in metric name
            'across_expr': 'mean'  # Single selector name
        }

        # List of expressions
        config = {
            'name': 'multi_metric',
            'type': 'across_subject',
            'within_expr': ['mae', 'rmse'],  # List of built-in metrics
            'across_expr': 'mean'
        }

        metric = create_metric_from_dict(config)
    """
    # Transform nested YAML structure to flat structure
    metric_data = {
        "name": config["name"],
        "label": config.get("label", config["name"]),
        "type": config.get("type", "across_sample"),
    }

    # Handle scope
    if "scope" in config:
        metric_data["scope"] = config["scope"]

    # Handle within_expr
    if "within_expr" in config:
        metric_data["within_expr"] = config["within_expr"]

    # Handle across_expr
    if "across_expr" in config:
        metric_data["across_expr"] = config["across_expr"]

    # Let MetricDefine handle validation and normalization
    return MetricDefine(**metric_data)


def create_metrics(configs: list[dict[str, Any]] | list[str]) -> list[MetricDefine]:
    """
    Create metrics from configurations or simple names.

    Args:
        configs: List of metric configuration dictionaries or simple metric names

    Returns:
        List of MetricDefine instances

    Examples:
        # From simple names
        metrics = create_metrics(['mae', 'rmse', 'me'])

        # From configuration dictionaries
        configs = [
            {'name': 'mae', 'label': 'Mean Absolute Error'},
            {'name': 'custom_rmse', 'label': 'Custom RMSE', 'type': 'across_subject'}
        ]
        metrics = create_metrics(configs)
    """
    if not configs:
        return []

    # Check if first item is string (names) or dict (configs)
    if isinstance(configs[0], str):
        # Simple names list
        return [MetricDefine(name=name) for name in configs]
    else:
        # Configuration dictionaries
        # pyre-ignore
        return [create_metric_from_dict(config) for config in configs]
