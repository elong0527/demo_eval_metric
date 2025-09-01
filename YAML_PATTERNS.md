# YAML Patterns and Processing Strategies

## Overview
This document describes patterns for YAML configuration and corresponding processing strategies for model evaluation pipelines.

## Metric Types

The `type` field in metrics determines the aggregation hierarchy and grouping strategy. Each type answers a different analytical question:

### Type Selection Guide

| Type | Scenario |
|------|----------|
| `across_samples` | Sample-wide statistics |
| `across_subject` | Per-subject statistics |
| `within_subject` | Subject-specific metrics |
| `across_visit` | Per-visit statistics |
| `within_visit` | Subject-visit-specific metrics |


### Core Metric Types

#### 1. `across_samples` (No Grouping)
**Question:** What is the metric across all data points?

```yaml
metrics:
  - name: "mae"
    type: across_samples
```

**Processing:**
```python
df.select(pl.col("absolute_error").mean())  # Direct aggregation, no groups
```

#### 2. `across_subject` (Two-Level: Subject then Population)
**Question:** What is the average across subjects?
```yaml
metrics:
  - name: "mae:mean"
    type: across_subject
```
**Processing:**
```python
# Step 1: Calculate per subject
df.group_by("subject_id").agg(
    pl.col("absolute_error").mean().alias("mae")
).select(  # Step 2: Average across subjects
    pl.col("mae").mean()
)
```

#### 3. `within_subject` (Group by Subject)
**Question:** What is the metric for each subject?
```yaml
metrics:
  - name: "mae"
    type: within_subject
```
**Processing:**
```python
df.group_by("subject_id").agg(pl.col("absolute_error").mean())
```

#### 4. `across_visit` (Group by Subject and Visit)
**Question:** What is the metric at each subject and visit?
```yaml
metrics:
  - name: "mae"
    type: across_visit
```
**Processing:**
```python
# Step 1: Calculate per subject and visit
df.group_by("subject_id", "visit_id').agg(
    pl.col("absolute_error").mean().alias("mae")
).select(  # Step 2: Average across subjects and visit
    pl.col("mae").mean()
)
```

#### 5. `within_visit` (Group by Subject and Visit)
**Question:** What is the metric for each subject-visit combination?
```yaml
metrics:
  - name: "mae"
    type: within_visit
```
**Processing:**
```python
df.group_by(["subject_id", "visit_id"]).agg(pl.col("absolute_error").mean())
```

## 1. Metric Naming Patterns

### Pattern A: Function Composition

```yaml
metrics:
  - name: "mae:mean"      # aggregation:selection
```

**Processing Strategy:**
- Split on `:` to extract aggregation and selection functions
- Map to pre-registered Polars expressions
- Chain operations: `group_by().agg(mae).select(mean)`

### Pattern B: Simple Names with Type
```yaml
metrics:
  - name: "n_subject"
    type: across_samples
    shared_by: model
```

**Processing Strategy:**
- Use `type` to determine grouping strategy
- Use `shared_by` to optimize calculation frequency
- Single function lookup in registry

### Pattern C: Custom Expressions with agg.expr and select.expr
```yaml
metrics:
  # Simple select expression (direct aggregation)
  - name: "ae:pct_threshold"
    label: "Percent of samples with absolute error < 1"
    type: across_samples
    select:
      expr: (pl.col("absolute_error") < 1).mean() * 100
  
  # Multiple aggregation expressions with select
  - name: "mae:wmean"
    label: "Weighted Mean of per subject MAE"
    type: across_subject
    agg:
      expr: 
        - mae.alias("_value")
        - pl.col("weight").mean().alias("_weight")
    select:
      expr: "(pl.col('_value') * pl.col('_weight')).sum() / pl.col('_weight').sum()"
```

**Processing Strategy:**

#### agg.expr Pattern (Aggregation Expressions)
- `agg.expr`: Can be a single expression or list of expressions
- For list: All expressions are computed in the aggregation step
- Built-in metrics (like `mae`) can be used directly in expressions

```python
# Single expression
agg:
  expr: pl.col("absolute_error").mean()

# Multiple expressions (list)
agg:
  expr:
    - mae.alias("_value")  # Built-in metric reference
    - pl.col("weight").mean().alias("_weight")
```

#### select.expr Pattern (Selection/Post-aggregation)
- `select.expr`: Applied after first-level aggregation
- When used alone: Acts as the primary aggregation expression
- When used with `agg.expr`: Transforms the aggregated results

```python
# When select.expr is used alone (no agg.expr)
if select_expr and not agg_expr:
    return select_expr  # Used as main aggregation

# When both agg.expr and select.expr are present
# Step 1: First-level aggregation with agg.expr
first_agg = lf.group_by(first_group).agg(agg_expr_list)

# Step 2: Second-level selection with select.expr
result = first_agg.group_by(second_group).agg(
    select_expr.alias("value")
)

## 2. Aggregation Level Patterns

### Pattern A: Hierarchical Aggregation

```yaml
metrics:
  - name: "mae:mean"
    type: across_subject    # subject → population
```

**Processing Flow:**
```python
df.group_by("subject_id").agg(mae)  # First level
  .select(mean)                      # Second level
```

### Pattern B: Direct Aggregation
```yaml
metrics:
  - name: "total_samples"
    type: across_samples    # No grouping
```

**Processing Flow:**
```python
df.select(pl.count())  # Direct calculation
```

### Pattern C: Visit-Based Aggregation
```yaml
metrics:
  - name: "rmse:mean"
    type: across_visit
```

**Processing Flow:**
```python
df.group_by("visit_id").agg(rmse)
  .select(mean)
```

## 3. Shared Semantics Patterns

### Pattern A: Universal Metrics
```yaml
metrics:
  - name: "total_subject"
    shared_by: all  # Same value everywhere
```

**Processing Strategy:**
- Calculate once at dataset level
- Replicate value for all groups/models
- Cache result to avoid recalculation

### Pattern B: Model-Agnostic Metrics
```yaml
metrics:
  - name: "n_subject"
    shared_by: model  # Same across models, varies by group
```

**Processing Strategy:**
- Calculate per group using ground_truth
- Apply same value to all model estimates
- Optimize by computing once per group

### Pattern C: Model-Specific Metrics
```yaml
metrics:
  - name: "mae"
    shared_by: none  # Default - unique per model/group
```

**Processing Strategy:**
- Calculate for each estimate × group combination
- No optimization possible
- Standard processing pipeline

## 4. Filter Application Patterns

### Pattern A: Global Filters
```yaml
filter:
  - adlb.anafl == True
  - adsv.enrlfl == True
```

**Processing Strategy:**
- Apply additional filter before metric calculation
- Combine with global filters using AND logic
- Create filtered subset for this metric only

## 5. Group/Subgroup Patterns

### Pattern A: Group Analysis (Direct Group By)
```yaml
columns:
  group: [region, treatment_arm]
  subgroup: [age_group, gender]
```

**Group Processing Strategy:**
```python
# Groups are added directly to group_by
def apply_group_analysis(df, group_cols, metric_expr):
    """Groups modify the aggregation directly"""
    
    # For subject-level metrics with groups
    return df.group_by(["subject_id"] + group_cols).agg(
        metric_expr
    ).group_by(group_cols).agg(
        pl.col("_value").mean()  # Aggregate across subjects within group
    )
    
    # Example: MAE per subject, then averaged by region and treatment_arm
    # df.group_by(["subject_id", "region", "treatment_arm"]).agg(mae)
    #   .group_by(["region", "treatment_arm"]).agg(mean)
```

### Pattern B: Subgroup Analysis (Marginal Calculations)
```yaml
columns:
  subgroup: [age_group, gender, race]  # Each analyzed independently
```

**Subgroup Processing Strategy:**
```python
# Subgroups create separate analysis branches using group_by
def apply_subgroup_analysis(df, subgroup_cols, metric_expr, base_grouping=None):
    """Each subgroup creates a separate marginal analysis using group_by"""
    
    results = []
    base_grouping = base_grouping or []
    
    # Each subgroup independently (marginal)
    for subgroup_col in subgroup_cols:
        # Group by the subgroup column (plus any base grouping)
        grouping_cols = base_grouping + [subgroup_col]
        
        # Apply metric calculation with group_by
        result = df.group_by(grouping_cols).agg(
            metric_expr.alias("_value")
        )
        
        # Add subgroup identifiers
        result = result.with_columns(
            pl.lit(subgroup_col).alias("subgroup_var")
        ).rename({subgroup_col: "subgroup_level"})
        
        results.append(result)
    
    return pl.concat(results, how="diagonal")

### Pattern C: Combined Group + Subgroup
```yaml
columns:
  group: [treatment_arm]          # Direct grouping
  subgroup: [age_group, gender]   # Marginal within each group
```

**Combined Processing Strategy:**
```python
# No need for a separate function - just use apply_subgroup_analysis directly
# When groups are present, pass them as base_grouping:

result = apply_subgroup_analysis(
    df, 
    subgroup_cols=['age_group', 'gender'], 
    metric_expr=mae_expr,
    base_grouping=['treatment_arm']  # Groups become the base
)

# This produces:
# df.group_by(['treatment_arm', 'age_group']).agg(metric)  # First subgroup
# df.group_by(['treatment_arm', 'gender']).agg(metric)     # Second subgroup
# Each with subgroup_var and subgroup_level columns added
```

## 6. Multiple Estimates Pattern

### Pattern A: Named Estimates
```yaml
columns:
  estimates: [baseline_model, enhanced_model, ensemble_model]
```

**Processing Strategy:**
```python
for estimate_name in estimates:
    # Process each model separately
    result = calculate_metrics(df, ground_truth, estimate_name)
    result = result.with_columns(
        pl.lit(estimate_name).alias("model")
    )
```

## 7. Output Structure Patterns

### Pattern A: Long Format (Preferred for Analysis)
```yaml
output:
  format: long
  columns: [group, subgroup, metric, estimate, value]
```

**Result Structure:**
```
| group | subgroup | metric | estimate | value |
|-------|----------|--------|----------|-------|
| A     | Male     | mae    | model1   | 0.5   |
| A     | Male     | mae    | model2   | 0.4   |
```

### Pattern B: Wide Format (For Reporting)
```yaml
output:
  format: wide
  pivot_on: estimate
  values: [mae, rmse]
```

**Result Structure:**
```
| group | subgroup | mae_model1 | mae_model2 | rmse_model1 | rmse_model2 |
|-------|----------|------------|------------|-------------|-------------|
| A     | Male     | 0.5        | 0.4        | 0.7         | 0.6         |
```

## 8. Performance Optimization Patterns

### Pattern A: Lazy Evaluation
```yaml
execution:
  lazy: true
```

**Processing Strategy:**
```python
# Build lazy query plan
lazy_df = pl.scan_parquet("data.parquet")
lazy_result = lazy_df.filter(...).group_by(...).agg(...)

# Collect in batches
results = []
for batch in lazy_result.collect_batches(10000):
    results.append(process_batch(batch))
```
