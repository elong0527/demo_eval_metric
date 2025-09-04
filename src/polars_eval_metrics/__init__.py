"""
Polars Eval Metrics - High-performance model evaluation framework.

Simple, fast, and flexible metric evaluation using Polars lazy evaluation.
"""

from .metric_define import MetricDefine, MetricType, MetricScope
from .metric_registry import MetricRegistry
from .metric_helpers import create_metrics
from .metric_evaluator import MetricEvaluator

__version__ = "0.1.0"

__all__ = [
    # Core
    "MetricDefine",
    "MetricType",
    "MetricScope",
    "create_metrics",
    "MetricRegistry",
    "MetricEvaluator",
]
