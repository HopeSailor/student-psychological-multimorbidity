from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import npv, specificity

OUTCOMES = ["depression_risk", "anxiety_risk", "suicidal_behavior_risk"]


def load_inputs(cfg):
    data = pd.read_csv(cfg["data"]["processed_dir"] / "analysis_dataset.csv")
    manifest = json.loads(
        (cfg["data"]["processed_dir"] / "feature_manifest.json").read_text()
    )
    return data, manifest


def preprocess(X, scale=True):
    cat = [c for c in X if c in {"Sex", "Course_Type", "Ethnicity", "Institution"}]
    num = [c for c in X if c not in cat]
    nsteps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        nsteps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(nsteps), num),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                cat,
            ),
        ],
        verbose_feature_names_out=False,
    )


def model_specs(X, fast=False):
    return {
        "logistic": (
            Pipeline(
                [
                    ("pre", preprocess(X, True)),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=5000, class_weight="balanced", random_state=2026
                        ),
                    ),
                ]
            ),
            {"clf__C": [0.1, 1, 10]},
        ),
        "elastic_net": (
            Pipeline(
                [
                    ("pre", preprocess(X, True)),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=5000,
                            solver="saga",
                            penalty="elasticnet",
                            class_weight="balanced",
                            random_state=2026,
                        ),
                    ),
                ]
            ),
            {"clf__C": [0.1, 1, 10], "clf__l1_ratio": [0.1, 0.5, 0.9]},
        ),
        "random_forest": (
            Pipeline(
                [
                    ("pre", preprocess(X, False)),
                    (
                        "clf",
                        RandomForestClassifier(
                            class_weight="balanced", random_state=2026, n_jobs=-1
                        ),
                    ),
                ]
            ),
            {
                "clf__n_estimators": [300] if fast else [300, 500],
                "clf__max_depth": [3, 8, None],
                "clf__min_samples_leaf": [1, 5],
            },
        ),
    }


def threshold_metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "sensitivity": recall_score(y, pred, zero_division=0),
        "specificity": specificity(y, pred),
        "ppv": precision_score(y, pred, zero_division=0),
        "npv": npv(y, pred),
        "f1": f1_score(y, pred, zero_division=0),
        "n_flagged": int(pred.sum()),
        "pct_flagged": 100 * pred.mean(),
        "false_negative_rate": fn / (fn + tp) if fn + tp else np.nan,
        "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan,
    }


def choose_thresholds(y, p, sensitivity_target=0.8):
    fpr, tpr, th = roc_curve(y, p)
    finite = np.isfinite(th)
    th, tpr, fpr = th[finite], tpr[finite], fpr[finite]
    youden = th[np.argmax(tpr - fpr)]
    valid = np.where(tpr >= sensitivity_target)[0]
    safety = th[valid[np.argmax(1 - fpr[valid])]] if len(valid) else th[np.argmax(tpr)]
    return {"default": 0.5, "youden": float(youden), "safety": float(safety)}


def calibration_stats(y, p):
    z = logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000).fit(z, y)
    return float(m.intercept_[0]), float(m.coef_[0, 0])


def evaluate(y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    ci, cs = calibration_stats(y, p)
    return {
        "auroc": roc_auc_score(y, p),
        "auprc": average_precision_score(y, p),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "sensitivity": recall_score(y, pred),
        "specificity": specificity(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "npv": npv(y, pred),
        "f1": f1_score(y, pred),
        "brier": brier_score_loss(y, p),
        "calibration_intercept": ci,
        "calibration_slope": cs,
    }


def bootstrap_ci(y, p, n=500, seed=2026):
    rng = np.random.default_rng(seed)
    vals = []
    y = np.asarray(y)
    p = np.asarray(p)
    for _ in range(n):
        ix = rng.integers(0, len(y), len(y))
        yy, pp = y[ix], p[ix]
        if len(np.unique(yy)) < 2:
            continue
        pred = pp >= 0.5
        vals.append(
            [
                roc_auc_score(yy, pp),
                average_precision_score(yy, pp),
                recall_score(yy, pred),
                specificity(yy, pred),
                f1_score(yy, pred),
                brier_score_loss(yy, pp),
            ]
        )
    a = np.asarray(vals)
    names = ["auroc", "auprc", "sensitivity", "specificity", "f1", "brier"]
    return {
        f"{name}_{side}": float(np.percentile(a[:, i], q))
        for i, name in enumerate(names)
        for side, q in (("ci_low", 2.5), ("ci_high", 97.5))
    }


def train_all(cfg):
    data, manifest = load_inputs(cfg)
    seed = cfg["project"]["random_seed"]
    cv = StratifiedKFold(cfg["project"]["cv_folds"], shuffle=True, random_state=seed)
    perf, thresholds, boot, cvrows, predictions = [], [], [], [], {}
    for outcome in OUTCOMES:
        eligible = data[outcome].notna()
        outcome_data = (
            data.loc[eligible].reset_index().rename(columns={"index": "source_index"})
        )
        y = outcome_data[outcome].astype(int)
        train_idx, test_idx = train_test_split(
            np.arange(len(y)),
            test_size=cfg["project"]["test_size"],
            random_state=seed,
            stratify=y,
        )
        for feature_set in ["low_burden", "extended"]:
            features = (
                manifest[feature_set]
                if feature_set == "low_burden"
                else manifest[feature_set][outcome]
            )
            X = outcome_data[features]
            for model_name, (pipe, grid) in model_specs(
                X, cfg["modeling"].get("fast_mode", False)
            ).items():
                search = GridSearchCV(
                    pipe,
                    grid,
                    scoring="roc_auc",
                    cv=cv,
                    n_jobs=cfg["modeling"]["n_jobs"],
                    return_train_score=False,
                )
                search.fit(X.iloc[train_idx], y.iloc[train_idx])
                p = search.predict_proba(X.iloc[test_idx])[:, 1]
                row = {
                    "outcome": outcome,
                    "feature_set": feature_set,
                    "model": model_name,
                    "cv_auroc": search.best_score_,
                    "best_params": json.dumps(search.best_params_),
                }
                row.update(evaluate(y.iloc[test_idx], p))
                perf.append(row)
                b = {
                    "outcome": outcome,
                    "feature_set": feature_set,
                    "model": model_name,
                }
                b.update(
                    bootstrap_ci(
                        y.iloc[test_idx].to_numpy(),
                        p,
                        cfg["project"]["bootstrap_resamples"],
                        seed,
                    )
                )
                boot.append(b)
                for i, params in enumerate(search.cv_results_["params"]):
                    cvrows.append(
                        {
                            "outcome": outcome,
                            "feature_set": feature_set,
                            "model": model_name,
                            "params": json.dumps(params),
                            "mean_test_auroc": search.cv_results_["mean_test_score"][i],
                            "rank": search.cv_results_["rank_test_score"][i],
                        }
                    )
                for label, t in choose_thresholds(
                    y.iloc[test_idx], p, cfg["modeling"]["safety_sensitivity_target"]
                ).items():
                    tr = {
                        "outcome": outcome,
                        "feature_set": feature_set,
                        "model": model_name,
                        "threshold_type": label,
                    }
                    tr.update(threshold_metrics(y.iloc[test_idx], p, t))
                    thresholds.append(tr)
                key = f"{outcome}__{feature_set}__{model_name}"
                predictions[key] = {
                    "test_idx": outcome_data.iloc[test_idx]["source_index"].to_numpy(),
                    "y": y.iloc[test_idx].to_numpy(),
                    "p": p,
                    "features": features,
                    "estimator": search.best_estimator_,
                }
                joblib.dump(
                    search.best_estimator_,
                    cfg["outputs"]["models_dir"] / f"{key}.joblib",
                )
    perf = pd.DataFrame(perf)
    # Parsimonious selection: within 0.01 AUROC of best, minimize Brier, then prefer logistic.
    selected = []
    for outcome in OUTCOMES:
        g = perf[perf.outcome == outcome]
        competitive = g[g.auroc >= g.auroc.max() - 0.01].copy()
        competitive["simple"] = competitive.model.map(
            {"logistic": 0, "elastic_net": 1, "random_forest": 2}
        )
        selected.append(competitive.sort_values(["brier", "simple"]).iloc[0])
    selected = pd.DataFrame(selected).drop(columns="simple")
    out = cfg["outputs"]["tables_dir"]
    perf.to_csv(out / "table_model_performance_all.csv", index=False)
    for outcome, n in zip(OUTCOMES, ["depression", "anxiety", "suicide"]):
        perf[perf.outcome == outcome].to_csv(
            out / f"table_model_performance_{n}.csv", index=False
        )
    pd.DataFrame(thresholds).to_csv(out / "table_safety_thresholds.csv", index=False)
    pd.DataFrame(boot).to_csv(
        out / "table_bootstrap_confidence_intervals.csv", index=False
    )
    pd.DataFrame(cvrows).to_csv(out / "table_complete_cv_results.csv", index=False)
    selected.to_csv(out / "table_selected_models.csv", index=False)
    joblib.dump(predictions, cfg["outputs"]["models_dir"] / "test_predictions.joblib")
    return perf, selected, predictions, data


def post_evaluation(cfg, selected, predictions, data):
    tabs = cfg["outputs"]["tables_dir"]
    calrows = []
    dcarows = []
    shprows = []
    subrows = []
    for _, row in selected.iterrows():
        outcome = row.outcome
        key = f"{outcome}__{row.feature_set}__{row.model}"
        z = predictions[key]
        y, p = z["y"], z["p"]
        ci, cs = calibration_stats(y, p)
        calrows.append(
            {
                "outcome": outcome,
                "model": row.model,
                "feature_set": row.feature_set,
                "brier": brier_score_loss(y, p),
                "intercept": ci,
                "slope": cs,
            }
        )
        for t in np.arange(0.01, 0.501, 0.01):
            pred = p >= t
            nb = ((pred & (y == 1)).sum() / len(y)) - (
                (pred & (y == 0)).sum() / len(y)
            ) * t / (1 - t)
            prev = np.mean(y)
            all_nb = prev - (1 - prev) * t / (1 - t)
            dcarows += [
                {
                    "outcome": outcome,
                    "threshold": t,
                    "strategy": "model-guided support",
                    "net_benefit": nb,
                },
                {
                    "outcome": outcome,
                    "threshold": t,
                    "strategy": "support-all",
                    "net_benefit": all_nb,
                },
                {
                    "outcome": outcome,
                    "threshold": t,
                    "strategy": "support-none",
                    "net_benefit": 0,
                },
            ]
        X = data.loc[z["test_idx"], z["features"]]
        est = z["estimator"]
        pi = permutation_importance(
            est, X, y, scoring="roc_auc", n_repeats=15, random_state=2026, n_jobs=-1
        )
        imp = (
            pd.DataFrame(
                {
                    "outcome": outcome,
                    "feature": z["features"],
                    "mean_abs_importance": pi.importances_mean,
                    "importance_sd": pi.importances_std,
                }
            )
            .sort_values("mean_abs_importance", ascending=False)
            .head(10)
        )
        shprows.append(imp)
        for subgroup in ["Sex", "Course_Type", "Institution", "Ethnicity"]:
            vals = data.loc[z["test_idx"], subgroup].reset_index(drop=True)
            for level in vals.dropna().unique():
                m = (vals == level).to_numpy()
                yy = y[m]
                pp = p[m]
                if (
                    len(yy) < cfg["modeling"]["min_subgroup_n"]
                    or yy.sum() < cfg["modeling"]["min_subgroup_cases"]
                    or (len(yy) - yy.sum()) < cfg["modeling"]["min_subgroup_cases"]
                ):
                    continue
                tm = threshold_metrics(yy, pp, 0.5)
                subrows.append(
                    {
                        "outcome": outcome,
                        "subgroup": subgroup,
                        "level": level,
                        "n": len(yy),
                        "positive_cases": int(yy.sum()),
                        "negative_cases": int(len(yy) - yy.sum()),
                        "auroc": roc_auc_score(yy, pp),
                        "auprc": average_precision_score(yy, pp),
                        "sensitivity": tm["sensitivity"],
                        "specificity": tm["specificity"],
                        "false_negative_rate": tm["false_negative_rate"],
                        "false_positive_rate": tm["false_positive_rate"],
                        "brier": brier_score_loss(yy, pp),
                    }
                )
    pd.DataFrame(calrows).to_csv(tabs / "table_calibration_metrics.csv", index=False)
    pd.DataFrame(dcarows).to_csv(tabs / "table_decision_curve_summary.csv", index=False)
    pd.concat(shprows).to_csv(
        tabs / "table_feature_importance_top10_combined.csv", index=False
    )
    for imp, outcome in zip(shprows, OUTCOMES):
        imp.to_csv(
            tabs
            / f"table_permutation_importance_{'suicide' if outcome.startswith('suicidal') else outcome.split('_')[0]}.csv",
            index=False,
        )
    subs = pd.DataFrame(subrows)
    subs.to_csv(tabs / "table_subgroup_safety_combined.csv", index=False)
    for outcome, n in zip(OUTCOMES, ["depression", "anxiety", "suicide"]):
        subs[subs.outcome == outcome].to_csv(
            tabs / f"table_subgroup_safety_{n}.csv", index=False
        )
