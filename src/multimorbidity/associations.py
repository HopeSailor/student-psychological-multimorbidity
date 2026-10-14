from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2, norm

OUTCOMES = ["depression_risk", "anxiety_risk", "suicidal_behavior_risk"]
OUTCOME_LABELS = {
    "depression_risk": "Depressive symptom risk",
    "anxiety_risk": "Anxiety symptom risk",
    "suicidal_behavior_risk": "SBQ-R risk indicator",
}


def contingency_associations(data):
    rows = []
    for a, b in itertools.combinations(OUTCOMES, 2):
        tab = pd.crosstab(data[a], data[b]).reindex(
            index=[0, 1], columns=[0, 1], fill_value=0
        )
        n00, n01, n10, n11 = tab.loc[0, 0], tab.loc[0, 1], tab.loc[1, 0], tab.loc[1, 1]
        # Haldane correction only if needed.
        cells = np.array([n00, n01, n10, n11], dtype=float)
        if np.any(cells == 0):
            cells += 0.5
        n00, n01, n10, n11 = cells
        lor = np.log((n11 * n00) / (n10 * n01))
        se = np.sqrt(np.sum(1 / cells))
        phi = np.corrcoef(data[a], data[b])[0, 1]
        rows.append(
            {
                "outcome_1": a,
                "outcome_2": b,
                "phi": phi,
                "odds_ratio": np.exp(lor),
                "ci_low": np.exp(lor - 1.96 * se),
                "ci_high": np.exp(lor + 1.96 * se),
            }
        )
    return pd.DataFrame(rows)


def pattern_tables(data):
    p8 = (
        data.groupby(["risk_pattern_8"] + OUTCOMES, dropna=False)
        .size()
        .reset_index(name="n")
    )
    p8["percent"] = 100 * p8.n / len(data)
    order = [
        "No elevated risk",
        "Single internalising risk",
        "Depression-anxiety multimorbidity",
        "SBQ-R indicator only",
        "SBQ-R + one internalising risk",
        "All three elevated domains",
    ]
    p6 = (
        data["multimorbidity_profile_6"]
        .value_counts()
        .reindex(order)
        .rename_axis("profile")
        .reset_index(name="n")
    )
    p6["percent"] = 100 * p6.n / len(data)
    burden = (
        data["psychological_multimorbidity_burden"]
        .value_counts()
        .sort_index()
        .rename_axis("burden")
        .reset_index(name="n")
    )
    burden["percent"] = 100 * burden.n / len(data)
    return p8.sort_values("n", ascending=False), p6, burden


def burden_gradients(data, markers):
    rows = []
    for label, col in markers.items():
        for burden, g in data.groupby("psychological_multimorbidity_burden"):
            x = g[col].dropna()
            rows.append(
                {
                    "marker": label,
                    "variable": col,
                    "burden": int(burden),
                    "n": len(x),
                    "mean": x.mean(),
                    "sd": x.std(),
                    "median": x.median(),
                    "q1": x.quantile(0.25),
                    "q3": x.quantile(0.75),
                    "se": x.std() / np.sqrt(len(x)),
                }
            )
    return pd.DataFrame(rows)


def gee_shared_specific(data, markers):
    """Joint GEE estimates shared structure and formally tests outcome-specific effects."""
    d = data.copy().reset_index(drop=True)
    d["student_id"] = np.arange(len(d))
    zcols = []
    for label, col in markers.items():
        z = f"z_{label}"
        d[z] = (d[col] - d[col].mean()) / d[col].std()
        zcols.append(z)
    long = d.melt(
        id_vars=["student_id", "Age", "Sex", "Course_Type"] + zcols,
        value_vars=OUTCOMES,
        var_name="outcome",
        value_name="risk",
    )
    long["outcome"] = pd.Categorical(long.outcome, categories=OUTCOMES, ordered=True)
    formula = (
        "risk ~ C(outcome) * ("
        + " + ".join(zcols)
        + ") + Age + C(Sex) + C(Course_Type)"
    )
    model = smf.gee(
        formula,
        "student_id",
        long,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    )
    fit = model.fit(maxiter=200)
    params = fit.params
    cov = fit.cov_params()
    names = list(params.index)
    rows = []
    for label in markers:
        base = f"z_{label}"
        interaction_names = [f"C(outcome)[T.{o}]:{base}" for o in OUTCOMES[1:]]
        idx = [names.index(n) for n in interaction_names if n in names]
        if idx:
            b = params.iloc[idx].to_numpy()
            V = cov.iloc[idx, idx].to_numpy()
            stat = float(b @ np.linalg.pinv(V) @ b)
            phet = chi2.sf(stat, len(idx))
        else:
            phet = np.nan
        for outcome in OUTCOMES:
            contrast = np.zeros(len(names))
            contrast[names.index(base)] = 1
            if outcome != OUTCOMES[0]:
                n = f"C(outcome)[T.{outcome}]:{base}"
                if n in names:
                    contrast[names.index(n)] = 1
            est = float(contrast @ params.to_numpy())
            se = float(np.sqrt(contrast @ cov.to_numpy() @ contrast))
            rows.append(
                {
                    "marker": label,
                    "outcome": outcome,
                    "odds_ratio": np.exp(est),
                    "ci_low": np.exp(est - 1.96 * se),
                    "ci_high": np.exp(est + 1.96 * se),
                    "p_value": 2 * norm.sf(abs(est / se)),
                    "heterogeneity_p": phet,
                }
            )
    return pd.DataFrame(rows), fit, long


def profile_marker_summary(data, markers):
    z = data.copy()
    labels = []
    for label, col in markers.items():
        z[label] = (z[col] - z[col].mean()) / z[col].std()
        labels.append(label)
    order = [
        "No elevated risk",
        "Single internalising risk",
        "Depression-anxiety multimorbidity",
        "SBQ-R indicator only",
        "SBQ-R + one internalising risk",
        "All three elevated domains",
    ]
    return z.groupby("multimorbidity_profile_6")[labels].mean().reindex(order)


def run_associations(data, cfg):
    markers = cfg["multimorbidity"]["markers"]
    out = cfg["outputs"]["tables_dir"]
    p8, p6, burden = pattern_tables(data)
    p8.to_csv(out / "table_multimorbidity_patterns_8.csv", index=False)
    p6.to_csv(out / "table_multimorbidity_profiles_6.csv", index=False)
    burden.to_csv(out / "table_multimorbidity_burden.csv", index=False)
    contingency_associations(data).to_csv(
        out / "table_pairwise_outcome_associations.csv", index=False
    )
    burden_gradients(data, markers).to_csv(
        out / "table_risk_burden_gradients.csv", index=False
    )
    effects, fit, _ = gee_shared_specific(data, markers)
    effects.to_csv(out / "table_shared_outcome_specific_markers.csv", index=False)
    (cfg["outputs"]["reports_dir"] / "gee_shared_specific_summary.txt").write_text(
        fit.summary().as_text(), encoding="utf-8"
    )
    profile_marker_summary(data, markers).to_csv(
        out / "table_profile_marker_zscores.csv"
    )
    return {
        "patterns8": p8,
        "profiles6": p6,
        "burden": burden,
        "effects": effects,
        "profile_markers": profile_marker_summary(data, markers),
    }
