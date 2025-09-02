"""
Sample data generation for examples and testing.
"""

import polars as pl


def generate_sample_data(
    n_subjects: int = 13,
    n_visits: int = 5,
    n_groups: int = 3,
) -> pl.DataFrame:
    """
    Generate sample data for testing metrics.

    Args:
        n_subjects: Number of subjects
        n_visits: Number of visits per subject
        n_groups: Number of treatment groups
        seed: Random seed for reproducibility

    Returns:
        DataFrame with sample data
    """
    # Create base structure
    subjects = list(range(1, n_subjects + 1))
    visits = list(range(1, n_visits + 1))
    groups = [chr(65 + i) for i in range(n_groups)]  # A, B, C, ...

    # Generate combinations
    data = []
    races = ["White", "Black", "Asian", "Hispanic"]

    for subject in subjects:
        group = groups[(subject - 1) % n_groups]
        gender = "M" if subject % 2 == 0 else "F"
        race = races[(subject - 1) % len(races)]

        for visit in visits:
            # Generate values with some pattern
            base_value = 10 + subject * 5 + visit * 2

            data.append(
                {
                    "subject_id": subject,
                    "visit_id": visit,
                    "treatment": group,
                    "gender": gender,
                    "race": race,
                    "actual": float(base_value),
                    "model1": float(base_value + (subject % 3) - 0.2),
                    "model2": float(base_value - (visit % 2) + 0.3),
                    "weight": 1.0 + (subject % 3) * 0.1,
                }
            )

    return pl.DataFrame(data)
