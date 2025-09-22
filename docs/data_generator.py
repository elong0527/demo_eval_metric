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
    Generate sample data for testing metrics and documentation examples.

    The values follow deterministic patterns derived from the subject, visit, and
    treatment identifiers, so no random seed configuration is required.

    Args:
        n_subjects: Number of subjects
        n_visits: Number of visits per subject
        n_groups: Number of treatment groups

    Returns:
        DataFrame with sample data that includes demographic columns, weights,
        and model predictions with controlled missing values.
    """
    if n_subjects < 1:
        msg = "n_subjects must be at least 1"
        raise ValueError(msg)
    if n_visits < 1:
        msg = "n_visits must be at least 1"
        raise ValueError(msg)
    if n_groups < 1:
        msg = "n_groups must be at least 1"
        raise ValueError(msg)

    subjects = list(range(1, n_subjects + 1))
    visits = list(range(1, n_visits + 1))
    groups = [chr(65 + i) for i in range(n_groups)]  # A, B, C, ...

    races = ["White", "Black", "Asian", "Hispanic"]
    regions = ["North", "South", "East", "West"]
    age_groups = ["Young", "Middle", "Senior"]

    mid_subject = subjects[len(subjects) // 2]
    missing_model2_subject = subjects[1] if len(subjects) > 1 else subjects[0]

    records: list[dict[str, object]] = []

    for subject in subjects:
        treatment = groups[(subject - 1) % n_groups]
        gender = "M" if subject % 2 == 0 else "F"
        race = races[(subject - 1) % len(races)]
        region = regions[(subject - 1) % len(regions)]
        age_group = age_groups[(subject - 1) % len(age_groups)]

        subject_baseline = 12 + subject * 3

        for visit in visits:
            visit_effect = (visit - 1) * 4
            actual = subject_baseline + visit_effect

            if subject == mid_subject and visit == 1:
                model1 = None
            else:
                error_1 = (subject % 4 - 1.5) * 1.2 + (visit - 1) * 0.4
                model1 = actual + error_1

            if subject == missing_model2_subject and visit == n_visits:
                model2 = None
            else:
                error_2 = (visit % 3 - 1) * 2.1 + (subject % 2) * 1.0
                model2 = actual + error_2

            records.append(
                {
                    "subject_id": subject,
                    "visit_id": visit,
                    "treatment": treatment,
                    "gender": gender,
                    "race": race,
                    "region": region,
                    "age_group": age_group,
                    "actual": float(actual),
                    "model1": None if model1 is None else float(model1),
                    "model2": None if model2 is None else float(model2),
                    "weight": float(1.0 + (subject % 3) * 0.1),
                }
            )

    return pl.DataFrame(records)
