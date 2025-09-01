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
    label: str | None = None
    type: MetricType
    shared_by: SharedType | None = None
    agg_expr: list[str] | None = None
    select_expr: str | None = None
    
    def __init__(self, **kwargs):
        """Initialize with default settings for easy user review"""
        # Set default label if not provided (keep as-is, no case changes)
        if 'label' not in kwargs or kwargs['label'] is None:
            kwargs['label'] = kwargs.get('name', 'Unknown Metric')
        
        super().__init__(**kwargs)
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate metric name is not empty"""
        if not v or not v.strip():
            raise ValueError("Metric name cannot be empty")
        return v.strip()
    
    @field_validator('label')
    @classmethod
    def validate_label(cls, v: str | None) -> str | None:
        """Validate label is not empty"""
        if v is None:
            return None
        if not v.strip():
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
        # ACROSS_SAMPLES cannot have selectors (aggregation functions)
        if self.type == MetricType.ACROSS_SAMPLES and ':' in self.name:
            raise ValueError("across_samples should not have aggregation expression")
        
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
    
    def get_compiled_expressions(self):
        """
        Get the actual Polars expressions that will be used for this metric.
        
        Returns:
            tuple[list[Expr], Expr | None]: (agg_expressions, select_expression)
        """
        from metric_compiler import MetricCompiler
        compiler = MetricCompiler()
        
        return compiler.compile_expressions(
            self.name, 
            self.agg_expr, 
            self.select_expr
        )
    
    
    def __str__(self) -> str:
        """Detailed string representation matching expected format"""
        lines = [f"MetricData(name='{self.name}', type={self.type.value})"]
        lines.append(f"  Label: '{self.label}'")
        lines.append(f"  Shared by: {self.shared_by.value if self.shared_by else 'none'}")
        
        try:
            # Get compiled expressions
            agg_exprs, select_expr = self.get_compiled_expressions()
            
            # Determine base metric and selector names
            if ':' in self.name:
                base_name, selector_name = self.name.split(':', 1)
            else:
                base_name = self.name
                selector_name = None
            
            # Show aggregation expressions
            if agg_exprs:
                lines.append("  Aggregation expressions:")
                for expr in agg_exprs:
                    source = "custom" if self.agg_expr else base_name
                    lines.append(f"    - [{source}] {expr}")
            else:
                lines.append("  Aggregation expressions: none")
            
            # Show selection expression
            if select_expr is not None:
                lines.append("  Selection expression:")
                if self.select_expr:
                    lines.append(f"      - [custom] {select_expr}")
                elif selector_name:
                    lines.append(f"      - [{selector_name}] {select_expr}")
                else:
                    lines.append(f"      - [{base_name}] {select_expr}")
            else:
                lines.append("  Selection expression: none")
                
        except Exception as e:
            lines.append(f"  error: {str(e)}")
        
        return '\n'.join(lines)
    
    def __repr__(self) -> str:
        """Detailed representation for interactive display"""
        return self.__str__()

