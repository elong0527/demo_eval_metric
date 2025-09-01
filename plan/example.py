"""
Example using YAML configuration file with the YAML to Polars expression translator
"""

import polars as pl
from yaml_to_pl_expr import YAMLToPolarsTranslator
from data import get_sample_data

# Configure Polars display
pl.Config.set_tbl_cols(100)
pl.Config.set_fmt_str_lengths(100)
pl.Config.set_tbl_width_chars(200)

# Load YAML configuration from file
yaml_config_path = "evaluation_schema.yaml"

# Initialize translator with YAML file
translator = YAMLToPolarsTranslator(yaml_config_path)

# Get sample data
df = get_sample_data()

# Generate Polars expressions for each metric
expressions = {}
for metric in translator.metrics:
    expr_str = translator.get_polars_expression(metric)
    expressions[metric.name] = {
        "label": metric.label,
        "type": metric.type.value,
        "expression": expr_str,
        "shared_by": metric.shared_by
    }

print(expressions)

print(translator.get_polars_expression(metric))

metric = translator.metrics[4]
print(translator.get_polars_expression(metric))