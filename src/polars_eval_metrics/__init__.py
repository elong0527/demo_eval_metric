"""
Polars Eval Metrics - High-performance model evaluation framework.

Simple, fast, and flexible metric evaluation using Polars lazy evaluation.
"""

# pyre-strict

from .metric_define import MetricDefine, MetricScope, MetricType
from .metric_evaluator import MetricEvaluator
from .metric_helpers import create_metrics
from .metric_registry import MetricRegistry


__all__ = [
    # Core
    "MetricDefine",
    "MetricType",
    "MetricScope",
    "create_metrics",
    "MetricRegistry",
    "MetricEvaluator",
]
