"""
Factory for creating Metric instances from YAML configuration

Handles YAML parsing and metric initialization logic.
"""

from typing import Any
from .metric_define import MetricDefine, MetricType, MetricScope


class MetricFactory:
    """Factory for creating MetricDefine instances from various sources"""
    
    @staticmethod
    def from_yaml(config: dict[str, Any]) -> MetricDefine:
        """
        Create MetricDefine from YAML configuration dict
        
        This method handles the YAML-specific parsing logic and ensures
        the resulting Metric is in a complete, valid state.
        
        Args:
            config: Dictionary from YAML configuration
            
        Returns:
            MetricDefine instance with proper defaults
        """
        # Extract basic fields
        metric_data = {
            'name': config['name'],
            'label': config.get('label', config['name']),
            'type': MetricType(config.get('type', 'across_samples'))
        }
        
        # Parse scope if present (also support old 'shared_by' for compatibility)
        scope_key = 'scope' if 'scope' in config else 'shared_by'
        if scope_key in config:
            scope_value = config[scope_key]
            if scope_value is not None:
                # Map old values to new ones if needed
                if scope_value == 'all':
                    scope_value = 'global'
                metric_data['scope'] = MetricScope(scope_value)
        
        # Parse agg expressions (normalize to list)
        agg_config = config.get('agg', {})
        if isinstance(agg_config, dict) and 'expr' in agg_config:
            raw_expr = agg_config['expr']
            metric_data['agg_expr'] = [raw_expr] if isinstance(raw_expr, str) else raw_expr
        
        # Parse select expression
        select_config = config.get('select', {})
        if isinstance(select_config, dict) and 'expr' in select_config:
            metric_data['select_expr'] = select_config['expr']
        
        # Create the metric with validation
        metric = MetricDefine(**metric_data)
        
        return metric
    
    @staticmethod
    def from_config(config: dict[str, Any]) -> list[MetricDefine]:
        """
        Create a list of MetricDefine objects from a configuration dictionary
        
        Args:
            config: Configuration dictionary containing 'metrics' key with list of metric configs
            
        Returns:
            List of MetricDefine instances
            
        Example:
            config = {
                'metrics': [
                    {'name': 'mae', 'label': 'Mean Absolute Error'},
                    {'name': 'rmse', 'label': 'Root Mean Squared Error'}
                ]
            }
            metrics = MetricFactory.from_config(config)
        """
        if 'metrics' not in config:
            raise ValueError("Configuration must contain 'metrics' key")
        
        metric_configs = config['metrics']
        if not isinstance(metric_configs, list):
            raise ValueError("'metrics' must be a list of metric configurations")
        
        return [MetricFactory.from_yaml(m) for m in metric_configs]