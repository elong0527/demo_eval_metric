"""
Polars Eval Metrics - High-performance model evaluation framework.

Simple, fast, and flexible metric evaluation using Polars lazy evaluation.
"""

# pyre-strict

from .metric_define import MetricDefine, MetricType, MetricScope
from .metric_evaluator import MetricEvaluator
from .metric_registry import MetricRegistry
from .metric_helpers import create_metrics


__all__ = [
    # Core
    "MetricDefine",
    "MetricType",
    "MetricScope",
    "create_metrics",
    "MetricRegistry",
    "MetricEvaluator",
]
