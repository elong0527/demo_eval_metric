"""
Example demonstrating YAML to Polars expression translation

This module shows how to use the YAMLToPolarsLazyTranslator for model evaluation.
"""

from yaml_to_pl_lazy import YAMLToPolarsLazyTranslator
from data import get_sample_data
from metric import MetricType


# ============================================================================
# Basic Usage
# ============================================================================

# Initialize translator with YAML configuration
translator = YAMLToPolarsLazyTranslator("evaluation_schema.yaml")

# Load sample data
df = get_sample_data()


# ============================================================================
# Example 1: Single Metric Computation
# ============================================================================

# Prepare dataframe with error columns
lf_prep = translator.prepare_dataframe(df, ground_truth="aval", estimate="model1")

# Find and compute a specific metric
mae_mean_metric = next(m for m in translator.metrics if m.name == "mae:mean")
mae_mean_result = translator.get_lazy_expression(lf_prep, mae_mean_metric).collect()


# ============================================================================
# Example 2: Batch Metric Computation
# ============================================================================

# Compute multiple metrics efficiently using lazy evaluation
def compute_metrics_for_model(df, model_col, metrics_list):
    """Compute multiple metrics for a single model"""
    lf_prep = translator.prepare_dataframe(df, "aval", model_col)
    
    results = {}
    for metric in metrics_list:
        result_lf = translator.get_lazy_expression(lf_prep, metric)
        results[metric.name] = result_lf.collect()["value"][0]
    
    return results

# Select metrics to compute
selected_metrics = translator.metrics[:5]
model1_results = compute_metrics_for_model(df, "model1", selected_metrics)


# ============================================================================
# Example 3: Model Comparison
# ============================================================================

def compare_models(df, metric, model_columns):
    """Compare multiple models using the same metric"""
    comparison = {}
    
    for model in model_columns:
        lf_prep = translator.prepare_dataframe(df, "aval", model)
        result = translator.get_lazy_expression(lf_prep, metric).collect()
        comparison[model] = result["value"][0]
    
    return comparison

# Compare models using MAE
mae_metric = next(m for m in translator.metrics if m.name == "mae:mean")
model_comparison = compare_models(df, mae_metric, ["model1", "model2"])


# ============================================================================
# Example 4: Different Aggregation Levels
# ============================================================================

# Within-subject metrics (individual level)
within_subject_metric = next(
    m for m in translator.metrics 
    if m.type == MetricType.WITHIN_SUBJECT
)
within_subject_results = translator.get_lazy_expression(
    lf_prep, within_subject_metric
).collect()

# Across-subject metrics (population level)
across_subject_metric = next(
    m for m in translator.metrics 
    if m.type == MetricType.ACROSS_SUBJECT
)
across_subject_results = translator.get_lazy_expression(
    lf_prep, across_subject_metric
).collect()


# ============================================================================
# Example 5: Custom Metrics
# ============================================================================

# Custom weighted mean metric
weighted_metric = next(m for m in translator.metrics if m.name == "mae:wmean")
weighted_result = translator.get_lazy_expression(lf_prep, weighted_metric).collect()

# Custom threshold metric
threshold_metric = next(m for m in translator.metrics if m.name == "ae:pct_threshold")
threshold_result = translator.get_lazy_expression(lf_prep, threshold_metric).collect()


# ============================================================================
# Example 6: Optimized Batch Processing
# ============================================================================

def batch_compute_by_type(df, model_col, metric_type):
    """Batch compute all metrics of a specific type"""
    lf_prep = translator.prepare_dataframe(df, "aval", model_col)
    
    # Filter metrics by type
    filtered_metrics = [m for m in translator.metrics if m.type == metric_type]
    
    # Build lazy expressions
    lazy_results = [
        translator.get_lazy_expression(lf_prep, metric)
        for metric in filtered_metrics
    ]
    
    # Collect all at once for efficiency
    collected = [lf.collect() for lf in lazy_results]
    
    # Return as dictionary
    return {
        filtered_metrics[i].name: collected[i]["value"][0]
        for i in range(len(filtered_metrics))
    }

# Compute all across-samples metrics
across_samples_results = batch_compute_by_type(df, "model1", MetricType.ACROSS_SAMPLES)


# ============================================================================
# Example 7: Query Plan Inspection
# ============================================================================

# Get the optimized query plan for debugging/optimization
query_plan = translator.get_lazy_expression(lf_prep, mae_mean_metric).explain(optimized=True)


# ============================================================================
# Example 8: Metadata Access
# ============================================================================

# Access metric metadata programmatically
def get_metrics_summary():
    """Get summary of all available metrics"""
    return [
        {
            "name": m.name,
            "label": m.label,
            "type": m.type.value,
            "is_two_level": m.type in [MetricType.ACROSS_SUBJECT, MetricType.ACROSS_VISIT],
            "has_custom_expr": bool(m.agg_expr)
        }
        for m in translator.metrics
    ]

metrics_summary = get_metrics_summary()


# ============================================================================
# Example 9: Group-based Analysis (if columns exist in data)
# ============================================================================

# Check if group columns exist in the data before using them
available_cols = df.columns

# Filter to only use existing columns
valid_group_cols = [col for col in translator.group_cols if col in available_cols]
valid_subgroup_cols = [col for col in translator.subgroup_cols if col in available_cols]

# If valid group columns exist, use them for analysis
if valid_group_cols:
    # Update translator's columns temporarily
    original_group_cols = translator.group_cols
    translator.group_cols = valid_group_cols
    
    grouped_result = translator.get_lazy_expression(
        lf_prep, mae_mean_metric, use_group=True
    ).collect()
    
    # Restore original columns
    translator.group_cols = original_group_cols

# For demonstration: Use existing columns as groups
# Since sample data has 'treatment' and 'gender' columns
if "treatment" in available_cols:
    translator.group_cols = ["treatment"]
    grouped_by_treatment = translator.get_lazy_expression(
        lf_prep, mae_mean_metric, use_group=True
    ).collect()
    translator.group_cols = []  # Reset


# ============================================================================
# Example 10: Integration with External Systems
# ============================================================================

def export_metrics_to_dict(df, model_col, metrics_list):
    """Export metrics in a format suitable for external systems"""
    lf_prep = translator.prepare_dataframe(df, "aval", model_col)
    
    output = {
        "model": model_col,
        "metrics": {}
    }
    
    for metric in metrics_list:
        result = translator.get_lazy_expression(lf_prep, metric).collect()
        output["metrics"][metric.name] = {
            "value": result["value"][0],
            "label": metric.label,
            "type": metric.type.value
        }
    
    return output

# Export results for model1
export_data = export_metrics_to_dict(df, "model1", translator.metrics[:3])