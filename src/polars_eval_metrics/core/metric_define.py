"""
Metric definition and expression preparation

This module combines metric configuration with expression compilation,
providing a single class for defining metrics and preparing Polars expressions.
"""

from enum import Enum
from pydantic import BaseModel, field_validator, model_validator, ConfigDict
from typing import Self
import polars as pl
import textwrap
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
    Metric definition with expression preparation capabilities.

    This class combines metric configuration with the ability to compile
    expressions to Polars expressions, focusing solely on defining and
    preparing metrics for evaluation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    label: str | None = None
    type: MetricType = MetricType.ACROSS_SAMPLES
    scope: MetricScope | None = None
    agg_expr: list[str | pl.Expr] | None = None
    select_expr: str | pl.Expr | None = None
    registry: MetricRegistry | None = None

    def __init__(self, **kwargs):
        """Initialize with default label if not provided"""
        if "label" not in kwargs or kwargs["label"] is None:
            kwargs["label"] = kwargs.get("name", "Unknown Metric")
        # Extract registry if provided, but don't store in model
        self._registry = kwargs.pop("registry", None)
        super().__init__(**kwargs)

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

    @field_validator("agg_expr", mode="before")
    @classmethod
    def normalize_agg_expr(cls, v):
        """Convert single string to list before validation"""
        if v is None:
            return None
        if isinstance(v, str):
            return [v]  # Convert single string to list
        if isinstance(v, pl.Expr):
            return [v]  # Convert single expression to list
        return v  # Already a list or something else

    @field_validator("agg_expr")
    @classmethod
    def validate_agg_expr(
        cls, v: list[str | pl.Expr] | None
    ) -> list[str | pl.Expr] | None:
        """Validate aggregation expressions list"""
        if v is None:
            return None

        if not isinstance(v, list):
            raise ValueError(
                f"agg_expr must be a list after normalization, got {type(v)}"
            )

        if not v:
            raise ValueError("agg_expr cannot be an empty list")

        for i, item in enumerate(v):
            if isinstance(item, str):
                if not item.strip():
                    raise ValueError(
                        f"agg_expr[{i}]: Built-in metric name cannot be empty"
                    )
            elif not isinstance(item, pl.Expr):
                raise ValueError(
                    f"agg_expr[{i}] must be a string (built-in name) or Polars expression, got {type(item)}"
                )

        return v

    @field_validator("select_expr")
    @classmethod
    def validate_select_expr(cls, v: str | pl.Expr | None) -> str | pl.Expr | None:
        """Validate select expression - can be built-in selector name or Polars expression"""
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
                f"select_expr must be a string (built-in selector) or Polars expression, got {type(v)}"
            )
        return v

    @model_validator(mode="after")
    def validate_expressions(self) -> Self:
        """Validate expression combinations"""
        is_custom = self.agg_expr is not None or self.select_expr is not None

        if is_custom:
            # Custom metrics must have at least one expression
            if self.agg_expr is None and self.select_expr is None:
                raise ValueError(
                    "Custom metrics must have at least agg_expr or select_expr"
                )

            # For two-level aggregation with multiple agg_expr, select_expr is required
            if self.type in (MetricType.ACROSS_SUBJECT, MetricType.ACROSS_VISIT):
                if (
                    self.agg_expr
                    and len(self.agg_expr) > 1
                    and self.select_expr is None
                ):
                    raise ValueError(
                        f"select_expr required for multiple agg expressions in {self.type.value}"
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
        self, registry: MetricRegistry = None
    ) -> tuple[list[pl.Expr], pl.Expr | None]:
        """
        Compile this metric's expressions to Polars expressions.

        Args:
            registry: Optional registry to use for resolving expressions.
                     If not provided, uses the registry from initialization or global registry.

        Returns:
            Tuple of (aggregation_expressions, selection_expression)
        """
        # Use provided registry, or instance registry, or create default
        reg = registry or self._registry or MetricRegistry()

        # Handle custom expressions
        if self.agg_expr is not None or self.select_expr is not None:
            result = self._compile_custom_expressions(reg)
        else:
            # Handle built-in metrics
            result = self._compile_builtin_expressions(reg)

        # For ACROSS_SAMPLES, move single expression to selection
        if self.type == MetricType.ACROSS_SAMPLES:
            agg_exprs, sel_expr = result
            if agg_exprs and sel_expr is None:
                # Move aggregation to selection for ACROSS_SAMPLES
                return [], agg_exprs[0] if len(agg_exprs) == 1 else agg_exprs[0]

        return result

    def _compile_custom_expressions(
        self, registry: MetricRegistry
    ) -> tuple[list[pl.Expr], pl.Expr | None]:
        """Return custom metric expressions - handle built-in names and Polars expressions in list"""
        agg_exprs = []

        # Handle agg_expr - always a list after normalization
        if self.agg_expr is not None:
            # List - resolve each item
            for item in self.agg_expr:
                if isinstance(item, str):
                    # Built-in metric name
                    try:
                        builtin_expr = registry.get_metric(item)
                        agg_exprs.append(builtin_expr)
                    except ValueError:
                        raise ValueError(f"Unknown built-in metric in agg_expr: {item}")
                else:
                    # Already a Polars expression
                    agg_exprs.append(item)

        # Handle select_expr - can be string (built-in selector) or expression
        select_pl_expr = None
        if self.select_expr is not None:
            if isinstance(self.select_expr, str):
                # Built-in selector name
                try:
                    select_pl_expr = registry.get_selector(self.select_expr)
                except ValueError:
                    raise ValueError(
                        f"Unknown built-in selector in select_expr: {self.select_expr}"
                    )
            else:
                # Already a Polars expression
                select_pl_expr = self.select_expr

        # If only select_expr provided (no agg_expr), use it as single aggregation
        if len(agg_exprs) == 0 and select_pl_expr is not None:
            return [select_pl_expr], None

        return agg_exprs, select_pl_expr

    def _compile_builtin_expressions(
        self, registry: MetricRegistry
    ) -> tuple[list[pl.Expr], pl.Expr | None]:
        """Compile built-in metric expressions"""
        parts = (self.name + ":").split(":")[:2]
        agg_name, select_name = parts[0], parts[1] if parts[1] else None

        # Get built-in aggregation expression (already a Polars expression)
        try:
            agg_expr = registry.get_metric(agg_name)
        except ValueError:
            raise ValueError(f"Unknown built-in metric: {agg_name}")

        # Get selector expression if specified (already a Polars expression)
        select_expr = None
        if select_name:
            try:
                select_expr = registry.get_selector(select_name)
            except ValueError:
                raise ValueError(f"Unknown built-in selector: {select_name}")
            # If there's a selector, return as aggregation + selection
            return [agg_expr], select_expr

        # No selector: this is likely ACROSS_SAMPLES, return as selection only
        return [], agg_expr

    def get_pl_chain(self, registry: MetricRegistry = None) -> str:
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

        def format_expr(expr, max_width=70):
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
        def format_exprs(exprs, indent="    "):
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

            # Show aggregation expressions (only if they exist)
            if agg_exprs:
                lines.append("  Aggregation expressions:")
                for i, expr in enumerate(agg_exprs):
                    # Determine source for each expression
                    if self.agg_expr is not None and i < len(self.agg_expr):
                        item = self.agg_expr[i]
                        source = item if isinstance(item, str) else "custom"
                    else:
                        source = base_name  # From metric name
                    lines.append(f"    - [{source}] {expr}")

            # Show selection expression (only if it exists)
            if select_expr is not None:
                lines.append("  Selection expression:")
                # Determine source for selection expression
                if isinstance(self.select_expr, str):
                    source = self.select_expr  # Built-in selector name
                elif self.select_expr is not None:
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
