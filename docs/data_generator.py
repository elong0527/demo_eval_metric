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

    The values follow a deterministic pattern derived from the subject and visit
    identifiers, so no random seed configuration is required.

    Args:
        n_subjects: Number of subjects
        n_visits: Number of visits per subject
        n_groups: Number of treatment groups

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


def create_clinical_trial_data():
    """
    Create sample clinical trial dataset for documentation examples.

    Returns a DataFrame with subject-level data including demographics,
    treatment assignments, and model predictions with realistic patterns.
    """
    return pl.DataFrame(
        {
            "subject_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "visit_id": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "treatment": ["A", "A", "A", "A", "A", "A", "B", "B", "B"],
            "age_group": [
                "young",
                "young",
                "young",
                "middle",
                "middle",
                "middle",
                "senior",
                "senior",
                "senior",
            ],
            "sex": ["M", "M", "M", "F", "F", "F", "M", "M", "M"],
            "actual": [10, 20, 30, 15, 25, 35, 12, 22, 32],
            "model_1": [8, 22, 28, 18, 24, 38, None, 19, 35],
            "model_2": [12, None, None, 13, 27, 33, 14, 25, 29],
        }
    )


def create_expanded_clinical_data():
    """
    Create expanded clinical trial dataset for comprehensive pivot examples.

    Returns a larger DataFrame with more subjects, regions, and prediction patterns
    suitable for demonstrating pivot functionality across different scopes.
    """
    # Base subject and visit structure
    subjects = list(range(1, 13))  # 12 subjects
    visits = [1, 2, 3]  # 3 visits each

    # Create combinations
    subject_ids = []
    visit_ids = []
    treatments = []
    regions = []
    age_groups = []
    sex_values = []

    for subject in subjects:
        for visit in visits:
            subject_ids.append(subject)
            visit_ids.append(visit)

            # Treatment assignment (6 subjects per treatment)
            treatments.append("A" if subject <= 6 else "B")

            # Region assignment (3 subjects per region per treatment)
            if subject in [1, 2, 3, 7, 8, 9]:
                regions.append("North")
            else:
                regions.append("South")

            # Age group pattern
            if subject in [1, 4, 7, 10]:
                age_groups.append("Young")
            elif subject in [2, 5, 8, 11]:
                age_groups.append("Middle")
            else:
                age_groups.append("Senior")

            # Sex pattern
            sex_values.append("M" if subject % 2 == 1 else "F")

    # Generate actual values with realistic progression
    actual_values = []
    for i, (subject, visit) in enumerate(zip(subject_ids, visit_ids)):
        base_value = 10 + (subject - 1) * 2  # Subject-specific baseline
        visit_effect = (visit - 1) * 5  # Visit progression
        actual_values.append(base_value + visit_effect)

    # Generate model predictions with different accuracy patterns
    model_1_predictions = []
    model_2_predictions = []

    for i, actual in enumerate(actual_values):
        subject = subject_ids[i]
        visit = visit_ids[i]

        # Model 1: Generally accurate with small random errors
        if subject == 7 and visit == 1:  # One missing value
            model_1_predictions.append(None)
        else:
            error_1 = (subject % 3 - 1) * 1.5  # Small systematic error
            model_1_predictions.append(actual + error_1)

        # Model 2: Less accurate with larger errors, some missing values
        if subject in [2] and visit in [2, 3]:  # Some missing values
            model_2_predictions.append(None)
        else:
            error_2 = (subject % 4 - 1.5) * 2.5  # Larger systematic error
            model_2_predictions.append(actual + error_2)

    return pl.DataFrame(
        {
            "subject_id": subject_ids,
            "visit_id": visit_ids,
            "treatment": treatments,
            "region": regions,
            "age_group": age_groups,
            "sex": sex_values,
            "actual": actual_values,
            "model_1": model_1_predictions,
            "model_2": model_2_predictions,
        }
    )
