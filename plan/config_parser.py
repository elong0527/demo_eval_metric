"""
YAML Configuration Parser for Metrics

This module handles parsing YAML configuration files and converting them
into Metric objects for evaluation.
"""

import yaml
from pathlib import Path
from typing import Any
from metric import Metric, MetricType, SharedType


class ConfigParser:
    """Parse YAML configuration into metric definitions"""
    
    def __init__(self, config_path: str | Path | dict):
        """
        Initialize parser with configuration
        
        Args:
            config_path: Path to YAML file or dict configuration
        """
        self.config = self._load_config(config_path)
        self.columns = self._parse_columns()
        self.metrics = self._parse_metrics()
    
    def _load_config(self, config_input: str | Path | dict) -> dict:
        """Load configuration from file or dict"""
        if isinstance(config_input, dict):
            return config_input
        
        with open(config_input, 'r') as f:
            return yaml.safe_load(f)
    
    def _parse_columns(self) -> dict[str, list[str]]:
        """Parse column configurations"""
        columns = self.config.get('columns', {})
        return {
            'group': columns.get('group', []),
            'subgroup': columns.get('subgroup', []),
            'subject': columns.get('subject', 'subject_id'),
            'visit': columns.get('visit', 'visit_id'),
        }
    
    def _parse_metrics(self) -> list[Metric]:
        """Parse metric definitions from configuration"""
        metrics = []
        
        for metric_config in self.config.get('metrics', []):
            metric = self._parse_single_metric(metric_config)
            metrics.append(metric)
        
        return metrics
    
    def _parse_single_metric(self, config: dict[str, Any]) -> Metric:
        """Parse a single metric configuration"""
        # Extract basic fields
        metric_data = {
            'name': config['name'],
            'label': config.get('label', config['name']),
            'type': MetricType(config.get('type', 'across_samples'))
        }
        
        # Parse shared_by if present
        if 'shared_by' in config:
            shared_value = config['shared_by']
            if shared_value in ['model', 'all', 'group']:
                metric_data['shared_by'] = SharedType(shared_value)
        
        # Parse aggregation expressions
        agg_config = config.get('agg', {})
        if isinstance(agg_config, dict) and 'expr' in agg_config:
            raw_expr = agg_config['expr']
            metric_data['agg_expr'] = [raw_expr] if isinstance(raw_expr, str) else raw_expr
        
        # Parse select expression
        select_config = config.get('select', {})
        if isinstance(select_config, dict) and 'expr' in select_config:
            metric_data['select_expr'] = select_config['expr']
        
        return Metric(**metric_data)
    
    def get_group_columns(self) -> list[str]:
        """Get group column names"""
        return self.columns['group']
    
    def get_subgroup_columns(self) -> list[str]:
        """Get subgroup column names"""
        return self.columns['subgroup']
    
    def get_all_analysis_columns(self) -> list[str]:
        """Get all analysis columns (group + subgroup)"""
        return self.columns['group'] + self.columns['subgroup']