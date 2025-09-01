"""Example demonstrating MetricFactory usage"""

import sys
sys.path.insert(0, '../src')

from polars_eval_metrics import MetricFactory, MetricEvaluator
from data_generator import generate_sample_data

# Example 1: Create metrics from YAML config
config = {
    'metrics': [
        {'name': 'mae', 'type': 'across_samples', 'label': 'Mean Absolute Error'},
        {'name': 'mse', 'type': 'across_samples', 'label': 'Mean Squared Error'},
        {'name': 'bias', 'type': 'across_samples', 'label': 'Bias'},
        {
            'name': 'custom_rmse',
            'type': 'across_samples',
            'label': 'Custom RMSE',
            'agg': {'expr': 'pl.col("squared_error").mean().sqrt().alias("value")'}
        }
    ]
}

# Parse metrics using MetricFactory
metrics = [MetricFactory.from_yaml(m) for m in config['metrics']]

print(f"Created {len(metrics)} metrics:")
for metric in metrics:
    print(f"  - {metric.name}: {metric.label}")

# Example 2: Use metrics with evaluator
df = generate_sample_data(n_subjects=4, n_visits=3)

evaluator = MetricEvaluator(
    df=df,
    metrics=metrics,
    ground_truth="actual",
    estimates=["model1", "model2"],
    group_by=["treatment"]
)

# Evaluate all metrics
results = evaluator.evaluate_all()
print(f"\nResults shape: {results.shape}")
print("\nSample results:")
print(results.select(["metric", "estimate", "treatment", "value"]).head(10))