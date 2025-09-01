"""Sample data generator for documentation examples."""

import polars as pl
import random
from typing import Optional

def get_sample_data(
    n_subjects: int = 10,
    n_visits: int = 3,
    n_treatments: int = 2,
    seed: Optional[int] = 42
) -> pl.DataFrame:
    """Generate sample data for examples."""
    if seed is not None:
        random.seed(seed)
    
    data = []
    for subject_id in range(1, n_subjects + 1):
        treatment = f"Treatment_{(subject_id % n_treatments) + 1}"
        gender = "M" if subject_id % 2 == 0 else "F"
        
        for visit_id in range(1, n_visits + 1):
            # Generate actual and model predictions
            base_value = 100 + subject_id * 5 + visit_id * 10
            noise = random.gauss(0, 5)
            
            data.append({
                "subject_id": f"SUBJ{subject_id:03d}",
                "visit_id": visit_id,
                "treatment": treatment,
                "gender": gender,
                "aval": base_value + noise,  # actual value
                "model1": base_value + random.gauss(2, 3),  # model 1 prediction
                "model2": base_value + random.gauss(-1, 4),  # model 2 prediction
            })
    
    return pl.DataFrame(data)

def generate_sample_data(
    n_subjects: int = 10,
    n_visits: int = 3,
    n_groups: int = 2,
    seed: Optional[int] = 42
) -> pl.DataFrame:
    """Generate sample data with specified parameters."""
    if seed is not None:
        random.seed(seed)
    
    data = []
    for subject_id in range(1, n_subjects + 1):
        treatment = f"Treatment_{(subject_id % n_groups) + 1}"
        
        for visit_id in range(1, n_visits + 1):
            # Generate actual and model predictions
            base_value = 100 + subject_id * 5 + visit_id * 10
            noise = random.gauss(0, 5)
            
            data.append({
                "subject_id": f"SUBJ{subject_id:03d}",
                "visit_id": visit_id,
                "treatment": treatment,
                "actual": base_value + noise,
                "model1": base_value + random.gauss(2, 3),
                "model2": base_value + random.gauss(-1, 4),
            })
    
    return pl.DataFrame(data)