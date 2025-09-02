"""
Error Expression Registry

This module provides an extensible registry system for error expressions,
allowing users to register custom error types that work seamlessly with
the existing metric evaluation framework.
"""

import polars as pl
from typing import Callable, Dict, Any


class ErrorExpressions:
    """
    Registry for error expression functions.
    
    Manages built-in and custom error types that can be used in metric definitions.
    Supports parameterized error functions for hyperparameter optimization scenarios.
    """
    
    # Class-level registry to store error functions
    _registry: Dict[str, Callable[[str, str], pl.Expr]] = {}
    
    @classmethod
    def register(cls, name: str, func: Callable[[str, str, ...], pl.Expr]) -> None:
        """
        Register a custom error expression function.
        
        Args:
            name: Name of the error type (e.g., 'buffer_error')
            func: Function that takes (estimate, ground_truth, *args) and returns pl.Expr
                  The function should return a Polars expression that computes the error
        
        Example:
            def buffer_error_func(estimate: str, ground_truth: str, threshold: float = 0.5) -> pl.Expr:
                return (pl.col(estimate) - pl.col(ground_truth)).abs() <= threshold
            
            ErrorExpressions.register('buffer_error', buffer_error_func)
        """
        cls._registry[name] = func
    
    @classmethod
    def get_expression(cls, name: str, estimate: str, ground_truth: str, **params) -> pl.Expr:
        """
        Get a Polars expression for the specified error type.
        
        Args:
            name: Name of the error type
            estimate: Estimate column name  
            ground_truth: Ground truth column name
            **params: Additional parameters for parameterized error functions
        
        Returns:
            Polars expression that computes the error
            
        Raises:
            ValueError: If the error type is not registered
        """
        if name not in cls._registry:
            raise ValueError(f"Error type '{name}' not registered. Available types: {list(cls._registry.keys())}")
        
        func = cls._registry[name]
        return func(estimate, ground_truth, **params)
    
    @classmethod
    def list_available(cls) -> list[str]:
        """List all available error types."""
        return list(cls._registry.keys())
    
    @classmethod
    def generate_error_columns(cls, estimate: str, ground_truth: str, 
                              error_types: list[str] | None = None,
                              error_params: Dict[str, Dict[str, Any]] | None = None) -> list[pl.Expr]:
        """
        Generate error column expressions for specified error types.
        
        Args:
            estimate: Estimate column name
            ground_truth: Ground truth column name  
            error_types: List of error types to generate. If None, generates all built-in types.
                        Users should provide unique names for parameterized error types.
            error_params: Dictionary mapping error types to their parameters
        
        Returns:
            List of Polars expressions with aliases matching the error_types names
        """
        if error_types is None:
            # Default to built-in error types for backward compatibility
            error_types = ['error', 'absolute_error', 'squared_error', 'percent_error', 'absolute_percent_error']
        
        error_params = error_params or {}
        expressions = []
        
        for error_type in error_types:
            params = error_params.get(error_type, {})
            expr = cls.get_expression(error_type, estimate, ground_truth, **params)
            expressions.append(expr.alias(error_type))
        
        return expressions


# Built-in error expression functions
def _error(estimate: str, ground_truth: str) -> pl.Expr:
    """Basic error: estimate - ground_truth"""
    return pl.col(estimate) - pl.col(ground_truth)

def _absolute_error(estimate: str, ground_truth: str) -> pl.Expr:
    """Absolute error: |estimate - ground_truth|"""
    error = pl.col(estimate) - pl.col(ground_truth)
    return error.abs()

def _squared_error(estimate: str, ground_truth: str) -> pl.Expr:
    """Squared error: (estimate - ground_truth)^2"""
    error = pl.col(estimate) - pl.col(ground_truth)
    return error ** 2

def _percent_error(estimate: str, ground_truth: str) -> pl.Expr:
    """Percent error: (estimate - ground_truth) / ground_truth * 100"""
    error = pl.col(estimate) - pl.col(ground_truth)
    return pl.when(pl.col(ground_truth) != 0).then(
        error / pl.col(ground_truth) * 100
    ).otherwise(None)

def _absolute_percent_error(estimate: str, ground_truth: str) -> pl.Expr:
    """Absolute percent error: |(estimate - ground_truth) / ground_truth| * 100"""
    error = pl.col(estimate) - pl.col(ground_truth)
    return pl.when(pl.col(ground_truth) != 0).then(
        (error / pl.col(ground_truth) * 100).abs()
    ).otherwise(None)


# Register all built-in error types
ErrorExpressions.register('error', _error)
ErrorExpressions.register('absolute_error', _absolute_error)
ErrorExpressions.register('squared_error', _squared_error)
ErrorExpressions.register('percent_error', _percent_error)
ErrorExpressions.register('absolute_percent_error', _absolute_percent_error)