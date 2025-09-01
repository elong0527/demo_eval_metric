"""
Example demonstrating YAML to Polars expression translation
"""

import polars as pl
from yaml_to_pl_lazy import YAMLToPolarsLazyTranslator
from data import get_sample_data

# Configure Polars display
pl.Config.set_tbl_cols(100)
pl.Config.set_fmt_str_lengths(100)
pl.Config.set_tbl_width_chars(200)

# Initialize translator with YAML configuration
translator = YAMLToPolarsLazyTranslator("evaluation_schema.yaml")

# Get sample data
df = get_sample_data()

# Example 1: Compute basic metrics
lf_prep = translator.prepare_dataframe(df, "aval", "model1")

# Single metric computation
mae_metric = next(m for m in translator.metrics if m.name == "mae:mean")
mae_result = translator.get_lazy_expression(lf_prep, mae_metric).collect()

# Example 2: Compute all metrics for a model
results = {}
for metric in translator.metrics[:5]:  # First 5 metrics as example
    lf_prep = translator.prepare_dataframe(df, "aval", "model1")
    result_lf = translator.get_lazy_expression(lf_prep, metric)
    results[metric.name] = result_lf.collect()["value"][0]

# Example 3: Compare models
model_comparison = {}
for model in ["model1", "model2"]:
    lf_prep = translator.prepare_dataframe(df, "aval", model)
    result = translator.get_lazy_expression(lf_prep, mae_metric).collect()
    model_comparison[model] = result["value"][0]

# Example 4: Custom expression metric (weighted mean)
weighted_metric = next(m for m in translator.metrics if m.name == "mae:wmean")
lf_prep = translator.prepare_dataframe(df, "aval", "model1")
weighted_result = translator.get_lazy_expression(lf_prep, weighted_metric).collect()

# Example 5: Within-subject metrics
within_subject_metric = next(m for m in translator.metrics if m.name == "mae")
lf_prep = translator.prepare_dataframe(df, "aval", "model1")
translator.get_lazy_expression(lf_prep, within_subject_metric).explain()

# Example 6: Get optimized query plan
query_plan = translator.get_lazy_expression(lf_prep, mae_metric).explain(optimized=True)

# Example 7: Batch compute multiple metrics efficiently
batch_results = []
lf_prep = translator.prepare_dataframe(df, "aval", "model1")
for metric in translator.metrics:
    if metric.type.value == "across_samples":
        result_lf = translator.get_lazy_expression(lf_prep, metric)
        batch_results.append(result_lf)

# Collect all at once for efficiency
collected_batch = [lf.collect() for lf in batch_results]

# Example 8: Access metric metadata
metric_info = [
    {
        "name": m.name,
        "label": m.label,
        "type": m.type.value,
        "shared_by": m.shared_by.value if m.shared_by else None,
        "has_custom_expr": bool(m.agg_expr or m.select_expr)
    }
    for m in translator.metrics
]

# Example 9: Group-based analysis (if groups were defined in data)
# This would use the use_group and use_subgroup parameters
# result_with_groups = translator.get_lazy_expression(
#     lf_prep, mae_metric, use_group=True
# ).collect()