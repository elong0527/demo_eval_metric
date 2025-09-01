"""
Factory for creating Metric instances from YAML configuration

Handles YAML parsing and metric initialization logic.
"""

from typing import Any
from .metric_data import MetricData, MetricType, SharedType


class MetricFactory:
    """Factory for creating MetricData instances from various sources"""
    
    @staticmethod
    def from_yaml(config: dict[str, Any]) -> MetricData:
        """
        Create MetricData from YAML configuration dict
        
        This method handles the YAML-specific parsing logic and ensures
        the resulting Metric is in a complete, valid state.
        
        Args:
            config: Dictionary from YAML configuration
            
        Returns:
            MetricData instance with proper defaults
        """
        # Extract basic fields
        metric_data = {
            'name': config['name'],
            'label': config.get('label', config['name']),
            'type': MetricType(config.get('type', 'across_samples'))
        }
        
        # Parse shared_by if present
        if 'shared_by' in config:
            shared_by_value = config['shared_by']
            if shared_by_value is not None:
                metric_data['shared_by'] = SharedType(shared_by_value)
        
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
        metric = MetricData(**metric_data)
        
        # Post-process to ensure completeness for downstream processing
        metric = MetricFactory._ensure_complete_expressions(metric)
        
        return metric
    
    @staticmethod
    def _ensure_complete_expressions(metric: MetricData) -> MetricData:
        """
        Ensure expressions are complete for downstream processing
        
        This method returns a new metric with default expressions filled in
        where necessary, eliminating the need for null checks in downstream code.
        
        Args:
            metric: The metric to process
            
        Returns:
            MetricData with complete expressions
        """
        # For two-level aggregations, ensure select_expr exists
        if metric.type in (MetricType.ACROSS_SUBJECT, MetricType.ACROSS_VISIT):
            if metric.agg_expr and not metric.select_expr:
                # Default to mean for two-level aggregation
                return metric.model_copy(update={'select_expr': "mean"})
        
        return metric
    
    @staticmethod
    def from_dict(data: dict[str, Any]) -> MetricData:
        """
        Create MetricData from a simple dictionary
        
        Args:
            data: Dictionary with metric fields
            
        Returns:
            MetricData instance
        """
        # Convert string types to enums if needed
        if 'type' in data and isinstance(data['type'], str):
            data['type'] = MetricType(data['type'])
        
        if 'shared_by' in data and isinstance(data['shared_by'], str):
            data['shared_by'] = SharedType(data['shared_by'])
        
        return MetricData(**data)