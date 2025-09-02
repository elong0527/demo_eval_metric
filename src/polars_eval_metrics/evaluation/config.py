"""
Configuration parser for evaluation settings

Handles complete YAML configuration including metrics, columns, and filters.
"""

import yaml
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field
from ..core import MetricDefine
from ..core.metric_helpers import create_metrics


class EvaluationConfig(BaseModel):
    """Complete evaluation configuration"""

    ground_truth: str = Field(default="actual", description="Ground truth column name")
    estimates: list[str] = Field(
        default_factory=list, description="List of estimate columns"
    )
    group_by: list[str] = Field(default_factory=list, description="Grouping columns")
    metrics: list[MetricDefine] = Field(
        default_factory=list, description="List of metrics"
    )
    filter_expr: str | None = Field(default=None, description="Filter expression")

    @classmethod
    def from_yaml(cls, config: dict[str, Any] | str | Path) -> "EvaluationConfig":
        """
        Load configuration from YAML file or dictionary

        Args:
            config: YAML file path, dictionary, or string content

        Returns:
            EvaluationConfig instance
        """
        # Load config if it's a file path
        if isinstance(config, (str, Path)):
            if isinstance(config, str) and "\n" not in config:
                # It's a file path
                with open(config, "r") as f:
                    config_dict = yaml.safe_load(f)
            else:
                # It's YAML content as string
                config_dict = yaml.safe_load(config)
        else:
            config_dict = config

        # Parse metrics using helper function
        metrics = create_metrics(config_dict.get("metrics", []))

        # Create config with all settings
        return cls(
            ground_truth=config_dict.get("ground_truth", "actual"),
            estimates=config_dict.get("estimates", []),
            group_by=config_dict.get("group_by", []),
            metrics=metrics,
            filter_expr=config_dict.get("filter_expr"),
        )

    def to_evaluator_kwargs(self) -> dict[str, Any]:
        """
        Convert config to kwargs for MetricEvaluator

        Returns:
            Dictionary of kwargs for MetricEvaluator initialization
        """
        kwargs = {
            "metrics": self.metrics,
            "ground_truth": self.ground_truth,
            "estimates": self.estimates,
            "group_by": self.group_by,
        }

        # Add filter expression if present
        if self.filter_expr:
            # Note: In real usage, this would need to be compiled to pl.Expr
            # For now, we'll leave it as a string for the evaluator to handle
            kwargs["filter_expr"] = self.filter_expr

        return kwargs

    def override(self, **kwargs) -> "EvaluationConfig":
        """
        Create a new config with overridden values

        Args:
            **kwargs: Values to override

        Returns:
            New EvaluationConfig with overridden values
        """
        current_data = self.model_dump()

        # Handle metrics specially - if provided as dicts, convert them
        if "metrics" in kwargs:
            metrics_raw = kwargs.pop("metrics")
            if metrics_raw and not isinstance(metrics_raw[0], MetricDefine):
                kwargs["metrics"] = create_metrics(metrics_raw)

        current_data.update(kwargs)
        return self.__class__(**current_data)
