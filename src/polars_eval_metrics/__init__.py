"""
Polars Eval Metrics - High-performance model evaluation framework.

Simple, fast, and flexible metric evaluation using Polars lazy evaluation.
"""

# pyre-strict

from .ard import ARD
from .metric_define import MetricDefine, MetricScope, MetricType
from .metric_evaluator import MetricEvaluator
from .metric_helpers import create_metrics
from .metric_registry import MetricRegistry
from .table_formatter import ard_to_gt, ard_to_wide, pivot_to_gt


__all__ = [
    # Core
    "ARD",
    "MetricDefine",
    "MetricType",
    "MetricScope",
    "create_metrics",
    "MetricRegistry",
    "MetricEvaluator",
    # Table formatting
    "ard_to_gt",
    "ard_to_wide",
    "pivot_to_gt",
]
