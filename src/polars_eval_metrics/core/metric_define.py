"""
Metric definition and expression preparation

This module combines metric configuration with expression compilation,
providing a single class for defining metrics and preparing Polars expressions.
"""

from enum import Enum
from pydantic import BaseModel, field_validator, model_validator
from typing import Self
import polars as pl
from .builtin import BUILTIN_METRICS, BUILTIN_SELECTORS


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


class MetricDefine(BaseModel):
    """
    Metric definition with expression preparation capabilities.
    
    This class combines metric configuration with the ability to compile
    expressions to Polars expressions, focusing solely on defining and
    preparing metrics for evaluation.
    """
    
    name: str
    label: str | None = None
    type: MetricType
    shared_by: SharedType | None = None
    agg_expr: list[str] | None = None
    select_expr: str | None = None
    
    def __init__(self, **kwargs):
        """Initialize with default label if not provided"""
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
    
    def compile_expressions(self) -> tuple[list[pl.Expr], pl.Expr | None]:
        """
        Compile this metric's expressions to Polars expressions.
        
        Returns:
            Tuple of (aggregation_expressions, selection_expression)
        """
        # Handle custom expressions
        if self.agg_expr or self.select_expr:
            result = self._compile_custom_expressions()
        else:
            # Handle built-in metrics
            result = self._compile_builtin_expressions()
        
        # For ACROSS_SAMPLES, move single expression to selection
        if self.type == MetricType.ACROSS_SAMPLES:
            agg_exprs, sel_expr = result
            if agg_exprs and not sel_expr:
                # Move aggregation to selection for ACROSS_SAMPLES
                return [], agg_exprs[0] if len(agg_exprs) == 1 else agg_exprs[0]
        
        return result
    
    def _compile_custom_expressions(self) -> tuple[list[pl.Expr], pl.Expr | None]:
        """Compile custom metric expressions"""
        agg_exprs = []
        if self.agg_expr:
            agg_exprs = [self._evaluate_expression(expr) for expr in self.agg_expr]
        
        select_pl_expr = None
        if self.select_expr:
            select_pl_expr = self._evaluate_expression(self.select_expr)
        
        # If only select_expr provided (no agg_expr), use it as single aggregation
        if not agg_exprs and select_pl_expr is not None:
            return [select_pl_expr], None
        
        return agg_exprs, select_pl_expr
    
    def _compile_builtin_expressions(self) -> tuple[list[pl.Expr], pl.Expr | None]:
        """Compile built-in metric expressions"""
        parts = (self.name + ':').split(':')[:2]
        agg_name, select_name = parts[0], parts[1] if parts[1] else None
        
        # Get built-in aggregation expression (already a Polars expression)
        agg_expr = BUILTIN_METRICS.get(agg_name)
        if agg_expr is None:
            raise ValueError(f"Unknown built-in metric: {agg_name}")
        
        # Get selector expression if specified (already a Polars expression)
        select_expr = None
        if select_name:
            select_expr = BUILTIN_SELECTORS.get(select_name)
            if select_expr is None:
                raise ValueError(f"Unknown built-in selector: {select_name}")
            # If there's a selector, return as aggregation + selection
            return [agg_expr], select_expr
        
        # No selector: this is likely ACROSS_SAMPLES, return as selection only
        return [], agg_expr
    
    def _evaluate_expression(self, expr_str: str) -> pl.Expr:
        """Convert string expression to Polars expression"""
        namespace = {
            'pl': pl,
            'col': pl.col,
            'lit': pl.lit,
            'len': pl.len,
            'struct': pl.struct,
        }
        
        try:
            return eval(expr_str, namespace, {})
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression '{expr_str}': {e}")
    
    def get_pl_chain(self) -> str:
        """
        Get a string representation of the Polars LazyFrame chain for this metric.
        
        Returns:
            String showing the LazyFrame operations that would be executed
        """
        agg_exprs, select_expr = self.compile_expressions()
        
        chain_lines = ["(", "  pl.LazyFrame"]
        
        # Determine the chain based on metric type
        if self.type == MetricType.ACROSS_SAMPLES:
            # Simple aggregation across all samples
            if select_expr is not None:
                chain_lines.append(f"  .select({select_expr})")
            elif agg_exprs:
                chain_lines.append(f"  .select({agg_exprs[0]})")
                
        elif self.type == MetricType.WITHIN_SUBJECT:
            # Group by subject, then aggregate
            chain_lines.append("  .group_by('subject_id')")
            if select_expr is not None:
                chain_lines.append(f"  .agg({select_expr})")
            elif agg_exprs:
                chain_lines.append(f"  .agg({agg_exprs[0]})")
                
        elif self.type == MetricType.ACROSS_SUBJECT:
            # Two-level: group by subject, aggregate, then aggregate across
            if agg_exprs:
                chain_lines.append("  .group_by('subject_id')")
                chain_lines.append(f"  .agg({agg_exprs[0]})")
            if select_expr is not None:
                chain_lines.append(f"  .select({select_expr})")
                
        elif self.type == MetricType.WITHIN_VISIT:
            # Group by subject and visit
            chain_lines.append("  .group_by(['subject_id', 'visit_id'])")
            if select_expr is not None:
                chain_lines.append(f"  .agg({select_expr})")
            elif agg_exprs:
                chain_lines.append(f"  .agg({agg_exprs[0]})")
                
        elif self.type == MetricType.ACROSS_VISIT:
            # Two-level: group by visit, aggregate, then aggregate across
            if agg_exprs:
                chain_lines.append("  .group_by(['subject_id', 'visit_id'])")
                chain_lines.append(f"  .agg({agg_exprs[0]})")
            if select_expr is not None:
                chain_lines.append(f"  .select({select_expr})")
        
        chain_lines.append(")")
        
        return '\n'.join(chain_lines)
    
    def __str__(self) -> str:
        """String representation for display"""
        lines = [f"MetricDefine(name='{self.name}', type={self.type.value})"]
        lines.append(f"  Label: '{self.label}'")
        lines.append(f"  Shared by: {self.shared_by.value if self.shared_by else 'none'}")
        
        try:
            agg_exprs, select_expr = self.compile_expressions()
            
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
                    lines.append(f"    - [custom] {select_expr}")
                elif selector_name:
                    lines.append(f"    - [{selector_name}] {select_expr}")
                else:
                    lines.append(f"    - [{base_name}] {select_expr}")
            else:
                lines.append("  Selection expression: none")
            
            # Add the LazyFrame chain
            lines.append("")
            lines.append(self.get_pl_chain())
                
        except Exception as e:
            lines.append(f"  Error compiling expressions: {str(e)}")
        
        return '\n'.join(lines)
    
    def __repr__(self) -> str:
        """Representation for interactive display"""
        return self.__str__()