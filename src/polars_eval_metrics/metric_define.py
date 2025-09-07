"""
Metric definition and expression preparation

This module combines metric configuration with expression compilation,
providing a single class for defining metrics and preparing Polars expressions.
"""

# pyre-strict
import textwrap

from enum import Enum
from typing import Self

import polars as pl
from pydantic import BaseModel, field_validator, model_validator, ConfigDict

from .metric_registry import MetricRegistry


class MetricType(Enum):
    """Metric aggregation types"""

    ACROSS_SAMPLES = "across_samples"
    ACROSS_SUBJECT = "across_subject"
    WITHIN_SUBJECT = "within_subject"
    ACROSS_VISIT = "across_visit"
    WITHIN_VISIT = "within_visit"


class MetricScope(Enum):
    """Scope for metric calculation - determines at what level the metric is computed"""

    GLOBAL = "global"  # Calculate once for entire dataset
    MODEL = "model"  # Calculate per model only, ignoring groups
    GROUP = "group"  # Calculate per group only, ignoring models


class MetricDefine(BaseModel):
    """
    Metric definition with hierarchical expression support.

    This class defines metrics with support for two-level aggregation patterns:
    - within_expr: Expressions for within-entity aggregation (e.g., within subject/visit)
    - across_expr: Expressions for across-entity aggregation or final metric computation

    Attributes:
        name: Metric identifier
        label: Display name for the metric
        type: Aggregation type (ACROSS_SAMPLES, WITHIN_SUBJECT, ACROSS_SUBJECT, etc.)
        scope: Calculation scope (GLOBAL, MODEL, GROUP) - orthogonal to aggregation type
        within_expr: Expression(s) for within-entity aggregation:
                    - Used in WITHIN_SUBJECT, ACROSS_SUBJECT, WITHIN_VISIT, ACROSS_VISIT
                    - Not used in ACROSS_SAMPLES (which operates directly on samples)
        across_expr: Expression for across-entity aggregation or final computation:
                    - For ACROSS_SAMPLES: Applied directly to error columns
                    - For ACROSS_SUBJECT/VISIT: Summarizes within_expr results across entities
                    - For WITHIN_SUBJECT/VISIT: Not used (within_expr is final)

    Note: within_expr and across_expr are distinct from group_by/subgroup_by which control
          analysis stratification (e.g., by treatment, age, sex) and apply to ALL metric types.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # pyre-ignore
    name: str
    label: str | None = None
    type: MetricType = MetricType.ACROSS_SAMPLES
    scope: MetricScope | None = None
    within_expr: list[str | pl.Expr] | None = None
    across_expr: str | pl.Expr | None = None
    registry: MetricRegistry | None = None

    def __init__(self, **kwargs: object) -> None:
        """Initialize with default label if not provided"""
        if "label" not in kwargs or kwargs["label"] is None:
            kwargs["label"] = kwargs.get("name", "Unknown Metric")
        # Extract registry if provided, but don't store in model
        registry = kwargs.pop("registry", None)
        super().__init__(**kwargs)
        # Set _registry after Pydantic initialization
        self._registry = registry

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate metric name is not empty"""
        if not v or not v.strip():
            raise ValueError("Metric name cannot be empty")
        return v.strip()

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: str | None) -> str | None:
        """Validate label is not empty"""
        if v is None:
            return None
        if not v.strip():
            raise ValueError("Metric label cannot be empty")
        return v.strip()

    @field_validator("type", mode="before")
    @classmethod
    def validate_type(cls, v) -> MetricType:
        """Convert string to MetricType enum if needed"""
        if isinstance(v, MetricType):
            return v
        if isinstance(v, str):
            # Try to convert string to enum
            try:
                # First try exact match
                return MetricType(v)
            except ValueError:
                # Try case-insensitive match with underscores
                v_normalized = v.lower().replace("-", "_")
                for member in MetricType:
                    if member.value == v_normalized:
                        return member
                # List valid options in error message
                valid_options = [m.value for m in MetricType]
                raise ValueError(
                    f"Invalid metric type: '{v}'. Valid options are: {', '.join(valid_options)}"
                )
        raise ValueError(f"type must be a MetricType enum or string, got {type(v)}")

    @field_validator("scope", mode="before")
    @classmethod
    def validate_scope(cls, v) -> MetricScope | None:
        """Convert string to MetricScope enum if needed"""
        if v is None:
            return None
        if isinstance(v, MetricScope):
            return v
        if isinstance(v, str):
            # Try to convert string to enum
            try:
                # First try exact match
                return MetricScope(v)
            except ValueError:
                # Try case-insensitive match
                v_normalized = v.lower()
                for member in MetricScope:
                    if member.value == v_normalized:
                        return member
                # List valid options in error message
                valid_options = [m.value for m in MetricScope]
                raise ValueError(
                    f"Invalid metric scope: '{v}'. Valid options are: {', '.join(valid_options)}"
                )
        raise ValueError(
            f"scope must be a MetricScope enum, string, or None, got {type(v)}"
        )

    @field_validator("within_expr", mode="before")
    @classmethod
    def normalize_within_expr(cls, v) -> object:
        """Convert single string to list before validation"""
        if v is None:
            return None
        if isinstance(v, str):
            return [v]  # Convert single string to list
        if isinstance(v, pl.Expr):
            return [v]  # Convert single expression to list
        return v  # Already a list or something else

    @field_validator("within_expr")
    @classmethod
    def validate_within_expr(
        cls, v: list[str | pl.Expr] | None
    ) -> list[str | pl.Expr] | None:
        """Validate within-entity aggregation expressions list"""
        if v is None:
            return None

        if not isinstance(v, list):
            raise ValueError(
                f"within_expr must be a list after normalization, got {type(v)}"
            )

        if not v:
            raise ValueError("within_expr cannot be an empty list")

        for i, item in enumerate(v):
            if isinstance(item, str):
                if not item.strip():
                    raise ValueError(
                        f"within_expr[{i}]: Built-in metric name cannot be empty"
                    )
            elif not isinstance(item, pl.Expr):
                raise ValueError(
                    f"within_expr[{i}] must be a string (built-in name) or Polars expression, got {type(item)}"
                )

        return v

    @field_validator("across_expr")
    @classmethod
    def validate_across_expr(cls, v: str | pl.Expr | None) -> str | pl.Expr | None:
        """Validate across-entity expression - can be built-in selector name or Polars expression"""
        if v is None:
            return None

        # If it's a string, validate it's not empty (will check if valid built-in during compile)
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("Built-in selector name cannot be empty")
            return v.strip()

        # Otherwise it must be a Polars expression
        if not isinstance(v, pl.Expr):
            raise ValueError(
                f"across_expr must be a string (built-in selector) or Polars expression, got {type(v)}"
            )
        return v

    @model_validator(mode="after")
    def validate_expressions(self) -> Self:
        """Validate expression combinations"""
        is_custom = self.within_expr is not None or self.across_expr is not None

        if is_custom:
            # Custom metrics must have at least one expression
            if self.within_expr is None and self.across_expr is None:
                raise ValueError(
                    "Custom metrics must have at least within_expr or across_expr"
                )

            # For two-level aggregation with multiple within_expr, across_expr is required
            if self.type in (MetricType.ACROSS_SUBJECT, MetricType.ACROSS_VISIT):
                if (
                    self.within_expr
                    and len(self.within_expr) > 1
                    and self.across_expr is None
                ):
                    raise ValueError(
                        f"across_expr required for multiple within expressions in {self.type.value}"
                    )
        else:
            # Built-in metrics should follow naming convention
            if ":" in self.name:
                parts = self.name.split(":", 1)
                if len(parts) != 2 or not parts[0] or not parts[1]:
                    raise ValueError(
                        f"Invalid built-in metric name format: {self.name}"
                    )

        return self

    def compile_expressions(
        self, registry: MetricRegistry | None = None
    ) -> tuple[list[pl.Expr], pl.Expr | None]:
        """
        Compile this metric's expressions to Polars expressions.

        Args:
            registry: Optional, kept for backward compatibility (not used).

        Returns:
            Tuple of (aggregation_expressions, selection_expression)
        """
        # Handle custom expressions
        if self.within_expr is not None or self.across_expr is not None:
            result = self._compile_custom_expressions()
        else:
            # Handle built-in metrics
            result = self._compile_builtin_expressions()

        # For ACROSS_SAMPLES, move single expression to selection
        if self.type == MetricType.ACROSS_SAMPLES:
            agg_exprs, sel_expr = result
            if agg_exprs and sel_expr is None:
                # Move aggregation to selection for ACROSS_SAMPLES
                return [], agg_exprs[0] if len(agg_exprs) == 1 else agg_exprs[0]

        return result

    def _compile_custom_expressions(self) -> tuple[list[pl.Expr], pl.Expr | None]:
        """Return custom metric expressions - handle built-in names and Polars expressions in list"""
        within_exprs = []

        # Handle within_expr - always a list after normalization
        if self.within_expr is not None:
            # List - resolve each item
            for item in self.within_expr:
                if isinstance(item, str):
                    # Built-in metric name
                    try:
                        builtin_expr = MetricRegistry.get_metric(item)
                        within_exprs.append(builtin_expr)
                    except ValueError:
                        raise ValueError(
                            f"Unknown built-in metric in within_expr: {item}"
                        )
                else:
                    # Already a Polars expression
                    within_exprs.append(item)

        # Handle across_expr - can be string (built-in selector) or expression
        across_pl_expr = None
        if self.across_expr is not None:
            if isinstance(self.across_expr, str):
                # Built-in selector name
                try:
                    across_pl_expr = MetricRegistry.get_summary(self.across_expr)
                except ValueError:
                    raise ValueError(
                        f"Unknown built-in selector in across_expr: {self.across_expr}"
                    )
            else:
                # Already a Polars expression
                across_pl_expr = self.across_expr

        # If only across_expr provided (no within_expr), use it as single aggregation
        if len(within_exprs) == 0 and across_pl_expr is not None:
            return [across_pl_expr], None

        return within_exprs, across_pl_expr

    def _compile_builtin_expressions(self) -> tuple[list[pl.Expr], pl.Expr | None]:
        """Compile built-in metric expressions"""
        parts = (self.name + ":").split(":")[:2]
        agg_name, select_name = parts[0], parts[1] if parts[1] else None

        # Get built-in aggregation expression (already a Polars expression)
        try:
            agg_expr = MetricRegistry.get_metric(agg_name)
        except ValueError:
            raise ValueError(f"Unknown built-in metric: {agg_name}")

        # Get selector expression if specified (already a Polars expression)
        select_expr = None
        if select_name:
            try:
                select_expr = MetricRegistry.get_summary(select_name)
            except ValueError:
                raise ValueError(f"Unknown built-in selector: {select_name}")
            # If there's a selector, return as aggregation + selection
            return [agg_expr], select_expr

        # No selector: this is likely ACROSS_SAMPLES, return as selection only
        return [], agg_expr

    def get_pl_chain(self, registry: MetricRegistry | None = None) -> str:
        """
        Get a string representation of the Polars LazyFrame chain for this metric.

        Args:
            registry: Optional registry to use for resolving expressions.

        Returns:
            String showing the LazyFrame operations that would be executed
        """
        agg_exprs, select_expr = self.compile_expressions(registry)

        chain_lines = ["(", "  pl.LazyFrame"]

        # Helper to clean and format Polars expressions
        def clean_expr(expr_str: str) -> str:
            """Remove Polars internal representation artifacts"""
            # Remove "dyn type:" prefixes
            import re

            cleaned = re.sub(r"dyn \w+:\s*", "", expr_str)
            # Remove outer brackets if they wrap the entire expression
            if cleaned.startswith("[") and cleaned.endswith("]"):
                cleaned = cleaned[1:-1]
            # Clean up nested brackets
            cleaned = cleaned.replace("[(", "(").replace(")]", ")")
            return cleaned

        def format_expr(expr, max_width=70) -> str:
            """Format a single expression with text wrapping"""
            expr_str = clean_expr(str(expr))

            # If short enough, return as-is
            if len(expr_str) <= max_width:
                return expr_str

            # Use textwrap for longer expressions
            return textwrap.fill(
                expr_str,
                width=max_width,
                subsequent_indent="      ",
                break_long_words=False,
                break_on_hyphens=False,
            )

        # Helper to format multiple expressions with proper indentation
        def format_exprs(exprs, indent="    ") -> str:
            if len(exprs) == 1:
                return format_expr(exprs[0])
            else:
                # Format as multi-line list for readability
                lines = ["["]
                for i, expr in enumerate(exprs):
                    formatted = format_expr(expr)
                    # Add comma except for last item
                    comma = "," if i < len(exprs) - 1 else ""

                    # Handle multi-line expressions
                    expr_lines = formatted.split("\n")
                    if len(expr_lines) == 1:
                        lines.append(f"{indent}  {formatted}{comma}")
                    else:
                        lines.append(f"{indent}  {expr_lines[0]}")
                        for line in expr_lines[1:]:
                            lines.append(f"{indent}  {line}")
                        lines[-1] += comma  # Add comma to last line

                lines.append(f"{indent}]")
                return "\n".join(lines)

        # Determine the chain based on metric type
        if self.type == MetricType.ACROSS_SAMPLES:
            # Simple aggregation across all samples
            if select_expr is not None:
                chain_lines.append(f"  .select({format_expr(select_expr)})")
            elif agg_exprs:
                if len(agg_exprs) == 1:
                    chain_lines.append(f"  .select({format_expr(agg_exprs[0])})")
                else:
                    chain_lines.append("  .select(")
                    chain_lines.append(format_exprs(agg_exprs))
                    chain_lines.append("  )")

        elif self.type == MetricType.WITHIN_SUBJECT:
            # Group by subject, then aggregate
            chain_lines.append("  .group_by('subject_id')")
            if select_expr is not None:
                chain_lines.append(f"  .agg({format_expr(select_expr)})")
            elif agg_exprs:
                if len(agg_exprs) == 1:
                    chain_lines.append(f"  .agg({format_expr(agg_exprs[0])})")
                else:
                    chain_lines.append("  .agg(")
                    chain_lines.append(format_exprs(agg_exprs))
                    chain_lines.append("  )")

        elif self.type == MetricType.ACROSS_SUBJECT:
            # Two-level: group by subject, aggregate, then aggregate across
            if agg_exprs:
                chain_lines.append("  .group_by('subject_id')")
                if len(agg_exprs) == 1:
                    chain_lines.append(f"  .agg({format_expr(agg_exprs[0])})")
                else:
                    chain_lines.append("  .agg(")
                    chain_lines.append(format_exprs(agg_exprs))
                    chain_lines.append("  )")
            if select_expr is not None:
                chain_lines.append(f"  .select({format_expr(select_expr)})")

        elif self.type == MetricType.WITHIN_VISIT:
            # Group by subject and visit
            chain_lines.append("  .group_by(['subject_id', 'visit_id'])")
            if select_expr is not None:
                chain_lines.append(f"  .agg({format_expr(select_expr)})")
            elif agg_exprs:
                if len(agg_exprs) == 1:
                    chain_lines.append(f"  .agg({format_expr(agg_exprs[0])})")
                else:
                    chain_lines.append("  .agg(")
                    chain_lines.append(format_exprs(agg_exprs))
                    chain_lines.append("  )")

        elif self.type == MetricType.ACROSS_VISIT:
            # Two-level: group by visit, aggregate, then aggregate across
            if agg_exprs:
                chain_lines.append("  .group_by(['subject_id', 'visit_id'])")
                if len(agg_exprs) == 1:
                    chain_lines.append(f"  .agg({format_expr(agg_exprs[0])})")
                else:
                    chain_lines.append("  .agg(")
                    chain_lines.append(format_exprs(agg_exprs))
                    chain_lines.append("  )")
            if select_expr is not None:
                chain_lines.append(f"  .select({format_expr(select_expr)})")

        chain_lines.append(")")

        return "\n".join(chain_lines)

    def __str__(self) -> str:
        """String representation for display"""
        lines = [f"MetricDefine(name='{self.name}', type={self.type.value})"]
        lines.append(f"  Label: '{self.label}'")
        if self.scope is not None:
            lines.append(f"  Scope: {self.scope.value}")

        try:
            agg_exprs, select_expr = self.compile_expressions()

            # Determine base metric and selector names
            if ":" in self.name:
                base_name, selector_name = self.name.split(":", 1)
            else:
                base_name = self.name
                selector_name = None

            # Show within-entity expressions (only if they exist)
            if agg_exprs:
                lines.append("  Within-entity expressions:")
                for i, expr in enumerate(agg_exprs):
                    # Determine source for each expression
                    if self.within_expr is not None and i < len(self.within_expr):
                        item = self.within_expr[i]
                        source = item if isinstance(item, str) else "custom"
                    else:
                        source = base_name  # From metric name
                    lines.append(f"    - [{source}] {expr}")

            # Show across-entity expression (only if it exists)
            if select_expr is not None:
                lines.append("  Across-entity expression:")
                # Determine source for selection expression
                if isinstance(self.across_expr, str):
                    source = self.across_expr  # Built-in selector name
                elif self.across_expr is not None:
                    source = "custom"  # Custom expression
                elif selector_name:
                    source = selector_name  # From metric name's selector part
                else:
                    source = base_name  # From metric name
                lines.append(f"    - [{source}] {select_expr}")

            # Add the LazyFrame chain
            lines.append("")
            lines.append(self.get_pl_chain())

        except Exception as e:
            lines.append(f"  Error compiling expressions: {str(e)}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Representation for interactive display"""
        return self.__str__()
