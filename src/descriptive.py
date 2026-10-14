"""Create the descriptive tables reported in the manuscript."""

from __future__ import annotations

import pandas as pd

from .modeling import OUTCOMES


CONTINUOUS = [
    "Age",
    "GAD7_Anxiety",
    "PHQ9_Depression",
    "MDQ_Mania",
    "SCI_Insomnia",
    "PSS_Stress",
    "SBQ_Suicidal_Ideation",
    "P16_Psychotic_Exp_Sum",
    "UCLA3_Loneliness",
]
CATEGORICAL = ["Sex", "Course_Type", "Ethnicity", "Institution"]


def create_descriptive_tables(data: pd.DataFrame, cfg: dict) -> None:
    out = cfg["outputs"]["tables_dir"]
    rows = []
    for column in CONTINUOUS:
        values = data[column].dropna()
        rows.append(
            {
                "variable": column,
                "level": "",
                "n": len(values),
                "mean_sd": f"{values.mean():.2f} ± {values.std():.2f}",
                "median_iqr": (
                    f"{values.median():.2f} "
                    f"[{values.quantile(.25):.2f}, {values.quantile(.75):.2f}]"
                ),
                "n_pct": "",
            }
        )
    for column in CATEGORICAL:
        for level, count in data[column].value_counts(dropna=False).items():
            rows.append(
                {
                    "variable": column,
                    "level": level,
                    "n": len(data),
                    "mean_sd": "",
                    "median_iqr": "",
                    "n_pct": f"{count} ({100 * count / len(data):.1f}%)",
                }
            )
    pd.DataFrame(rows).to_csv(out / "table1_sample_characteristics.csv", index=False)

    depression, anxiety, suicide = [data[column].astype("Int64") for column in OUTCOMES]
    indicators = {
        "depression_risk": depression,
        "anxiety_risk": anxiety,
        "suicidal_behavior_risk": suicide,
        "depression_and_anxiety": depression * anxiety,
        "depression_and_suicide": depression * suicide,
        "anxiety_and_suicide": anxiety * suicide,
        "all_three": depression * anxiety * suicide,
    }
    prevalence = [
        {
            "outcome": name,
            "n": int(values.sum(skipna=True)),
            "denominator": int(values.notna().sum()),
            "prevalence_pct": 100 * values.mean(skipna=True),
        }
        for name, values in indicators.items()
    ]
    pd.DataFrame(prevalence).to_csv(out / "table2_outcome_prevalence.csv", index=False)
