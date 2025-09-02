"""
Helper functions for creating metrics from various sources.

Simple, functional approach to metric creation without factory classes.
"""

from typing import Any
from .metric_define import MetricDefine


def create_metric_from_dict(config: dict[str, Any]) -> MetricDefine:
    """
    Create a MetricDefine from dictionary configuration.
    
    Handles YAML-style nested structures.
    
    Args:
        config: Dictionary with metric configuration
        
    Returns:
        MetricDefine instance
        
    Example:
        config = {
            'name': 'mae',
            'label': 'Mean Absolute Error', 
            'type': 'across_samples',
            'agg': {'expr': 'absolute_error.mean()'},
            'select': {'expr': 'value.mean()'}
        }
        metric = create_metric_from_dict(config)
    """
    # Transform nested YAML structure to flat structure
    metric_data = {
        'name': config['name'],
        'label': config.get('label', config['name']),
        'type': config.get('type', 'across_samples')
    }
    
    # Handle scope
    if 'scope' in config:
        metric_data['scope'] = config['scope']
    
    # Parse nested agg expressions
    agg_config = config.get('agg', {})
    if isinstance(agg_config, dict) and 'expr' in agg_config:
        raw_expr = agg_config['expr']
        metric_data['agg_expr'] = [raw_expr] if isinstance(raw_expr, str) else raw_expr
    
    # Parse nested select expression
    select_config = config.get('select', {})
    if isinstance(select_config, dict) and 'expr' in select_config:
        metric_data['select_expr'] = select_config['expr']
    
    # Let MetricDefine handle validation and normalization
    return MetricDefine(**metric_data)


def create_metrics(configs: list[dict[str, Any]] | list[str]) -> list[MetricDefine]:
    """
    Create metrics from configurations or simple names.
    
    Args:
        configs: List of metric configuration dictionaries or simple metric names
        
    Returns:
        List of MetricDefine instances
        
    Examples:
        # From simple names
        metrics = create_metrics(['mae', 'rmse', 'bias'])
        
        # From configuration dictionaries  
        configs = [
            {'name': 'mae', 'label': 'Mean Absolute Error'},
            {'name': 'custom_rmse', 'label': 'Custom RMSE', 'type': 'across_subject'}
        ]
        metrics = create_metrics(configs)
    """
    if not configs:
        return []
    
    # Check if first item is string (names) or dict (configs)
    if isinstance(configs[0], str):
        # Simple names list
        return [MetricDefine(name=name) for name in configs]
    else:
        # Configuration dictionaries
        return [create_metric_from_dict(config) for config in configs]
