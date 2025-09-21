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
# from .table_formatter import pivot_to_gt  # Disabled for ARD development


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
    # "format_pivot_table",  # Disabled for ARD development
]
