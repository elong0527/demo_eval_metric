"""
Sample clinical trial data for demonstrating the evaluation framework
"""

import polars as pl


def get_sample_data() -> pl.DataFrame:
    """
    Generate sample clinical trial data with multiple subjects, visits, and treatments
    
    Returns:
        DataFrame with clinical trial structure
    """
    return pl.DataFrame({
        "subject_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        "visit_id":   [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
        "treatment":  ["A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B"],
        "gender":     ["M", "M", "M", "F", "F", "F", "M", "M", "M", "F", "F", "F"],
        "aval":       [10, 12, 11, 15, 14, 13, 20, 22, 21, 25, 24, 23],
        "model1":     [11, 13, 12, 14, 15, 14, 21, 23, 22, 24, 25, 24],
        "model2":     [10.5, 12.5, 11.5, 14.5, 14.5, 13.5, 20.5, 22.5, 21.5, 24.5, 24.5, 23.5],
        "weight":     [1.0, 1.5, 1.0, 2.0, 1.5, 1.0, 1.0, 2.0, 1.5, 1.0, 1.0, 1.5]
    })


def get_larger_sample_data(n_subjects: int = 100, n_visits: int = 5) -> pl.DataFrame:
    """
    Generate larger sample clinical trial data for performance testing
    
    Args:
        n_subjects: Number of subjects to generate
        n_visits: Number of visits per subject
        
    Returns:
        DataFrame with synthetic clinical trial data
    """
    import numpy as np
    np.random.seed(42)
    
    data = []
    for subject_id in range(1, n_subjects + 1):
        # Assign treatment group
        treatment = "A" if subject_id <= n_subjects // 2 else "B"
        gender = "M" if np.random.random() > 0.5 else "F"
        
        # Base value for this subject
        base_value = np.random.normal(20 if treatment == "A" else 25, 3)
        
        for visit_id in range(1, n_visits + 1):
            # True value with some progression
            aval = base_value + np.random.normal(visit_id * 0.5, 1)
            
            # Model predictions with different error patterns
            model1 = aval + np.random.normal(1, 0.5)  # Systematic bias
            model2 = aval + np.random.normal(0, 0.8)  # No bias, more variance
            
            data.append({
                "subject_id": subject_id,
                "visit_id": visit_id,
                "treatment": treatment,
                "gender": gender,
                "aval": round(aval, 2),
                "model1": round(model1, 2),
                "model2": round(model2, 2)
            })
    
    return pl.DataFrame(data)


if __name__ == "__main__":
    df = get_sample_data()
    print(df)