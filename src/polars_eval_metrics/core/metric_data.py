"""
Metric definition with Pydantic validation

Pure data model for metric configuration without any Polars dependencies.
"""

from enum import Enum
from pydantic import BaseModel, field_validator, model_validator
from typing import Self


class MetricType(Enum):
    """Metric aggregation types"""
    ACROSS_SAMPLES = "across_samples"
    ACROSS_SUBJECT = "across_subject"
    WITHIN_SUBJECT = "within_subject"
    ACROSS_VISIT = "across_visit"
    WITHIN_VISIT = "within_visit"


class SharedType(Enum):
    """Shared semantics for metric calculation optimization"""
    MODEL = "model"
    ALL = "all"
    GROUP = "group"


class MetricData(BaseModel):
    """Metric definition with validation - pure data model"""
    name: str
    label: str
    type: MetricType
    shared_by: SharedType | None = None
    agg_expr: list[str] | None = None
    select_expr: str | None = None
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate metric name is not empty"""
        if not v or not v.strip():
            raise ValueError("Metric name cannot be empty")
        return v.strip()
    
    @field_validator('label')
    @classmethod
    def validate_label(cls, v: str) -> str:
        """Validate label is not empty"""
        if not v or not v.strip():
            raise ValueError("Metric label cannot be empty")
        return v.strip()
    
    @field_validator('agg_expr')
    @classmethod
    def validate_agg_expr(cls, v: list[str] | None) -> list[str] | None:
        """Validate aggregation expressions are not empty"""
        if v is None:
            return None
        if not v:
            raise ValueError("agg_expr cannot be an empty list")
        
        for expr in v:
            if not expr or not expr.strip():
                raise ValueError("agg_expr cannot contain empty expressions")
        
        return v
    
    @field_validator('select_expr')
    @classmethod
    def validate_select_expr(cls, v: str | None) -> str | None:
        """Validate select expression is not empty"""
        if v is None:
            return None
        if not v.strip():
            raise ValueError("select_expr cannot be empty")
        return v
    
    @model_validator(mode='after')
    def validate_expressions(self) -> Self:
        """Validate expression combinations"""
        # For custom metrics, at least one expression must be provided
        is_custom = self.agg_expr is not None or self.select_expr is not None
        
        if is_custom:
            # Custom metrics must have at least one expression
            if not self.agg_expr and not self.select_expr:
                raise ValueError("Custom metrics must have at least agg_expr or select_expr")
            
            # For two-level aggregation with multiple agg_expr, select_expr is required
            if self.type in (MetricType.ACROSS_SUBJECT, MetricType.ACROSS_VISIT):
                if self.agg_expr and len(self.agg_expr) > 1 and not self.select_expr:
                    raise ValueError(
                        f"select_expr required for multiple agg expressions in {self.type.value}"
                    )
        else:
            # Built-in metrics should follow naming convention
            if ':' in self.name:
                parts = self.name.split(':', 1)
                if len(parts) != 2 or not parts[0] or not parts[1]:
                    raise ValueError(f"Invalid built-in metric name format: {self.name}")
        
        return self