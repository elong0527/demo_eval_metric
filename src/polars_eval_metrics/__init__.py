"""
Polars Eval Metrics - High-performance model evaluation framework.

Simple, fast, and flexible metric evaluation using Polars lazy evaluation.
"""

from .core import MetricDefine, MetricType, MetricScope, create_metrics
from .evaluation import EvaluationConfig, MetricEvaluator

__version__ = "0.1.0"

__all__ = [
    # Core
    "MetricDefine",
    "MetricType",
    "MetricScope",
    "create_metrics",
    # Evaluation
    "MetricEvaluator",
    "EvaluationConfig",
]