from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
SEED = 2026

OUTCOMES = {
    "depression_risk": "PHQ9_Depression",
    "anxiety_risk": "GAD7_Anxiety",
    "suicidal_behavior_risk": "SBQ_Suicidal_Ideation",
}


def first_col(df: pd.DataFrame, startswith: str) -> str:
    matches = [c for c in df.columns if str(c).startswith(startswith)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one column starting with {startswith!r}, found {matches}"
        )
    return matches[0]


def numbered_col(df: pd.DataFrame, number: int) -> str:
    pattern = re.compile(rf"^{number}[.\uff0e]")
    matches = [c for c in df.columns if pattern.match(str(c))]
    if len(matches) != 1:
        raise ValueError(f"Expected one column numbered {number}, found {matches}")
    return matches[0]


def cols_numbered(df: pd.DataFrame, start: int, end: int) -> list[str]:
    return [numbered_col(df, i) for i in range(start, end + 1)]


def sum_mapped(
    df: pd.DataFrame, columns: list[str], maps: list[dict[str, int]]
) -> pd.Series:
    scored = []
    for column, mapping in zip(columns, maps, strict=True):
        values = df[column].map(mapping)
        bad = sorted(df.loc[values.isna(), column].dropna().astype(str).unique())
        if bad:
            raise ValueError(f"Unmapped values in {column}: {bad}")
        scored.append(values)
    return pd.concat(scored, axis=1).sum(axis=1, min_count=len(columns))


def score_questionnaire(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    age_col = first_col(df, "(1)A1")
    eligible_col = first_col(df, "A6")
    chinese_col = first_col(df, "A7")
    age = df[age_col].astype(str).str.extract(r"(\d+)", expand=False).astype(float)

    # Sleep Condition Indicator (SCI-8): all eight items are scored 4 (best)
    # to 0 (worst), yielding a 0--32 total.
    sci_cols = [first_col(df, f"B{i}") for i in range(1, 9)]
    sci_maps = [
        {"0–15分钟": 4, "16–30分钟": 3, "31–45分钟": 2, "46–60分钟": 1, "≥61分钟": 0},
        {"0–15分钟": 4, "16–30分钟": 3, "31–45分钟": 2, "46–60分钟": 1, "≥61分钟": 0},
        {"0–1晚": 4, "2晚": 3, "3晚": 2, "4晚": 1, "5–7晚": 0},
        {"非常好": 4, "好": 3, "一般": 2, "差": 1, "非常差": 0},
        {"完全没有": 4, "一点": 3, "有些": 2, "很大": 1, "非常大": 0},
        {"完全没有": 4, "一点": 3, "有些": 2, "很大": 1, "非常大": 0},
        {"完全没有": 4, "一点": 3, "有些": 2, "很大": 1, "非常大": 0},
        {
            "没有问题/小于1个月": 4,
            "1–2个月": 3,
            "3–6个月": 2,
            "7–12个月": 1,
            "大于1年": 0,
        },
    ]
    sci = sum_mapped(df, sci_cols, sci_maps)

    # PSS-10: 0--4 frequency scoring; items 4, 5, 7 and 8 are reverse scored.
    pss_cols = [numbered_col(df, i) for i in range(1, 11)]
    forward_05 = {"从不": 0, "几乎从不": 1, "有时": 2, "经常": 3, "非常经常": 4}
    reverse_05 = {k: 4 - v for k, v in forward_05.items()}
    pss = sum_mapped(
        df,
        pss_cols,
        [reverse_05 if i in {4, 5, 7, 8} else forward_05 for i in range(1, 11)],
    )

    # UCLA Loneliness Scale version 3: 1--4 scoring; positive items are reversed.
    ucla_cols = cols_numbered(df, 11, 30)
    forward_14 = {"从不": 1, "很少": 2, "有时": 3, "经常": 4}
    reverse_14 = {k: 5 - v for k, v in forward_14.items()}
    positive = {11, 15, 16, 19, 20, 25, 26, 29, 30}
    ucla = sum_mapped(
        df,
        ucla_cols,
        [reverse_14 if i in positive else forward_14 for i in range(11, 31)],
    )

    symptom_map = {"完全不会": 0, "几天": 1, "一半以上的日子": 2, "几乎每天": 3}
    phq_cols = cols_numbered(df, 31, 39)
    gad_cols = cols_numbered(df, 41, 47)
    phq = sum_mapped(df, phq_cols, [symptom_map] * 9)
    gad = sum_mapped(df, gad_cols, [symptom_map] * 7)

    sbq_cols = [
        first_col(df, "49."),
        first_col(df, "50. "),
        first_col(df, "50.\u4f60"),
        first_col(df, "51."),
    ]
    sbq_maps = [
        {
            "从来没有": 1,
            "有过一闪而过的想法": 2,
            "至少一次制定了自杀的计划": 3,
            "曾经尝试自杀": 4,
        },
        {
            "从来没有": 1,
            "很少（1次）": 2,
            "有时（2次）": 3,
            "经常（3~4次）": 4,
            "非常频繁（5次或更多）": 5,
        },
        {"没有": 1, "是的，有过1次": 2, "是的，不止1次": 3},
        {
            "永远不可能": 0,
            "非常不可能": 1,
            "不可能": 2,
            "相当不可能": 3,
            "可能": 4,
            "相当可能": 5,
        },
    ]
    sbq = sum_mapped(df, sbq_cols, sbq_maps)

    scored = pd.DataFrame(
        {
            "source_row": np.arange(2, len(df) + 2),
            "response_id": df["\u5e8f\u53f7"].astype(str),
            "submitted_at": pd.to_datetime(df["\u63d0\u4ea4\u7b54\u5377\u65f6\u95f4"]),
            "eligible_current_student": df[eligible_col].eq("是"),
            "eligible_chinese": df[chinese_col].eq("是"),
            "Age": age,
            "Sex": df[first_col(df, "A2")].map({"男": 1.0, "女": 2.0}),
            # The Chinese vocational-college course, ethnicity and institution have no
            # development-cohort equivalents. Out-of-range codes are intentionally
            # passed to the fitted one-hot encoder and therefore treated as unseen.
            "Course_Type": 6.0,
            "Ethnicity": 19.0,
            "Institution": 7.0,
            "SCI_Insomnia": sci,
            "PSS_Stress": pss,
            "UCLA3_Loneliness": ucla,
            "PHQ9_Depression": phq,
            "GAD7_Anxiety": gad,
            "SBQ_Suicidal_Ideation": sbq,
        }
    )
    scored["depression_risk"] = scored["PHQ9_Depression"].ge(10).astype(int)
    scored["anxiety_risk"] = scored["GAD7_Anxiety"].ge(10).astype(int)
    scored["suicidal_behavior_risk"] = scored["SBQ_Suicidal_Ideation"].ge(7).astype(int)
    scored["psychological_multimorbidity_burden"] = scored[list(OUTCOMES)].sum(axis=1)
    return scored


def calibration_stats(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    z = logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000).fit(z, y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    intercept, slope = calibration_stats(y, p)
    return {
        "auroc": roc_auc_score(y, p),
        "auprc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "sensitivity": recall_score(y, pred, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "ppv": precision_score(y, pred, zero_division=0),
        "npv": tn / (tn + fn) if tn + fn else np.nan,
        "f1": f1_score(y, pred, zero_division=0),
        "n_flagged": int(pred.sum()),
        "pct_flagged": 100 * pred.mean(),
    }


def bootstrap_metrics(
    y: np.ndarray, p: np.ndarray, threshold: float, n_resamples: int = 2000
) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    names = [
        "auroc",
        "auprc",
        "brier",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "pct_flagged",
    ]
    draws: dict[str, list[float]] = {name: [] for name in names}
    for _ in range(n_resamples):
        ix = rng.integers(0, len(y), len(y))
        yy, pp = y[ix], p[ix]
        if len(np.unique(yy)) < 2:
            continue
        pred = (pp >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(yy, pred, labels=[0, 1]).ravel()
        row = {
            "auroc": roc_auc_score(yy, pp),
            "auprc": average_precision_score(yy, pp),
            "brier": brier_score_loss(yy, pp),
            "sensitivity": recall_score(yy, pred, zero_division=0),
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "ppv": precision_score(yy, pred, zero_division=0),
            "npv": tn / (tn + fn) if tn + fn else np.nan,
            "pct_flagged": 100 * pred.mean(),
        }
        for name in names:
            if np.isfinite(row[name]):
                draws[name].append(float(row[name]))
    out = {}
    for name, values in draws.items():
        out[f"{name}_ci_low"] = float(np.percentile(values, 2.5))
        out[f"{name}_ci_high"] = float(np.percentile(values, 97.5))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score and externally validate the frozen low-burden models."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the private Wenjuanxing text-export XLSX file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "pilot_validation",
        help="Directory for aggregate tables and restricted row-level output.",
    )
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_excel(args.input, dtype=object)
    scored = score_questionnaire(raw)
    included = scored[
        scored["eligible_current_student"]
        & scored["eligible_chinese"]
        & scored["Age"].ge(18)
    ].copy()

    thresholds = pd.read_csv(
        ROOT / "outputs" / "tables" / "table_safety_thresholds.csv"
    )
    rows = []
    features = [
        "Age",
        "Sex",
        "Course_Type",
        "Ethnicity",
        "Institution",
        "SCI_Insomnia",
        "PSS_Stress",
        "UCLA3_Loneliness",
    ]
    for outcome in OUTCOMES:
        model_name = "elastic_net"
        model = joblib.load(
            ROOT / "outputs" / "models" / f"{outcome}__low_burden__{model_name}.joblib"
        )
        p = model.predict_proba(included[features])[:, 1]
        y = included[outcome].to_numpy(dtype=int)
        safety = thresholds.loc[
            (thresholds["outcome"] == outcome)
            & (thresholds["feature_set"] == "low_burden")
            & (thresholds["model"] == model_name)
            & (thresholds["threshold_type"] == "safety"),
            "threshold",
        ].item()
        for threshold_type, threshold in [
            ("default", 0.5),
            ("prespecified_safety", float(safety)),
        ]:
            row = {
                "outcome": outcome,
                "feature_set": "low_burden",
                "model": model_name,
                "threshold_type": threshold_type,
                "threshold": threshold,
                "n": len(y),
                "events": int(y.sum()),
                "prevalence_pct": 100 * y.mean(),
            }
            row.update(metrics(y, p, threshold))
            row.update(bootstrap_metrics(y, p, threshold))
            rows.append(row)
        included[f"predicted_probability__{outcome}"] = p

    summary = pd.DataFrame(rows)
    feasibility = pd.DataFrame(
        [
            {"measure": "exported_responses", "value": len(scored)},
            {"measure": "eligible_complete_responses", "value": len(included)},
            {
                "measure": "excluded_not_current_student",
                "value": int((~scored.eligible_current_student).sum()),
            },
            {
                "measure": "excluded_could_not_read_simplified_chinese",
                "value": int((~scored.eligible_chinese).sum()),
            },
            {"measure": "excluded_age_under_18", "value": int(scored.Age.lt(18).sum())},
            {
                "measure": "complete_required_low_burden_and_outcome_items",
                "value": int(
                    included[features + list(OUTCOMES)].notna().all(axis=1).sum()
                ),
            },
            {
                "measure": "median_platform_completion_seconds",
                "value": float(
                    raw.loc[included.index, "\u6240\u7528\u65f6\u95f4"]
                    .astype(str)
                    .str.extract(r"(\d+)", expand=False)
                    .astype(float)
                    .median()
                ),
            },
            {
                "measure": "questionnaire_length_acceptable_agree_or_strongly_agree_pct",
                "value": 100
                * raw.loc[included.index, first_col(raw, "53.")]
                .isin(["同意", "非常同意"])
                .mean(),
            },
            {
                "measure": "questionnaire_caused_obvious_discomfort_agree_or_strongly_agree_pct",
                "value": 100
                * raw.loc[included.index, first_col(raw, "54.")]
                .isin(["同意", "非常同意"])
                .mean(),
            },
            {
                "measure": "would_complete_similar_questionnaire_again_agree_or_strongly_agree_pct",
                "value": 100
                * raw.loc[included.index, first_col(raw, "55.")]
                .isin(["同意", "非常同意"])
                .mean(),
            },
        ]
    )
    sample = pd.DataFrame(
        [
            {
                "characteristic": "Age, median [IQR]",
                "value": f"{included.Age.median():.1f} [{included.Age.quantile(.25):.1f}, {included.Age.quantile(.75):.1f}]",
            },
            {
                "characteristic": "Female, n (%)",
                "value": f"{int((included.Sex == 2).sum())} ({100*(included.Sex == 2).mean():.1f}%)",
            },
            {
                "characteristic": "SCI-8, median [IQR]",
                "value": f"{included.SCI_Insomnia.median():.1f} [{included.SCI_Insomnia.quantile(.25):.1f}, {included.SCI_Insomnia.quantile(.75):.1f}]",
            },
            {
                "characteristic": "PSS-10, median [IQR]",
                "value": f"{included.PSS_Stress.median():.1f} [{included.PSS_Stress.quantile(.25):.1f}, {included.PSS_Stress.quantile(.75):.1f}]",
            },
            {
                "characteristic": "UCLA-20, median [IQR]",
                "value": f"{included.UCLA3_Loneliness.median():.1f} [{included.UCLA3_Loneliness.quantile(.25):.1f}, {included.UCLA3_Loneliness.quantile(.75):.1f}]",
            },
            {
                "characteristic": "PHQ-9, median [IQR]",
                "value": f"{included.PHQ9_Depression.median():.1f} [{included.PHQ9_Depression.quantile(.25):.1f}, {included.PHQ9_Depression.quantile(.75):.1f}]",
            },
            {
                "characteristic": "GAD-7, median [IQR]",
                "value": f"{included.GAD7_Anxiety.median():.1f} [{included.GAD7_Anxiety.quantile(.25):.1f}, {included.GAD7_Anxiety.quantile(.75):.1f}]",
            },
            {
                "characteristic": "SBQ-R, median [IQR]",
                "value": f"{included.SBQ_Suicidal_Ideation.median():.1f} [{included.SBQ_Suicidal_Ideation.quantile(.25):.1f}, {included.SBQ_Suicidal_Ideation.quantile(.75):.1f}]",
            },
        ]
    )

    restricted = included.drop(columns=["source_row", "response_id", "submitted_at"])
    restricted.to_csv(out / "pilot_scored_restricted.csv", index=False)
    summary.to_csv(out / "table_pilot_low_burden_validation.csv", index=False)
    feasibility.to_csv(out / "table_pilot_feasibility.csv", index=False)
    sample.to_csv(out / "table_pilot_sample_characteristics.csv", index=False)
    print(feasibility.to_string(index=False))
    print(sample.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
