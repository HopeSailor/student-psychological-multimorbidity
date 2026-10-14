from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    hamming_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.multioutput import ClassifierChain
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

OUTCOMES = ["depression_risk", "anxiety_risk", "suicidal_behavior_risk"]


def preprocessor(X):
    cats = [c for c in X if c in {"Sex", "Course_Type", "Ethnicity", "Institution"}]
    nums = [c for c in X if c not in cats]
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                nums,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                cats,
            ),
        ],
        verbose_feature_names_out=False,
    )


def metrics(y, p, method):
    pred = (p >= 0.5).astype(int)
    row = {
        "method": method,
        "macro_auroc": roc_auc_score(y, p, average="macro"),
        "macro_auprc": average_precision_score(y, p, average="macro"),
        "hamming_loss": hamming_loss(y, pred),
        "exact_match_accuracy": accuracy_score(y, pred),
        "mean_brier": np.mean([brier_score_loss(y[:, j], p[:, j]) for j in range(3)]),
    }
    for j, n in enumerate(["depression", "anxiety", "suicide"]):
        row[f"{n}_auroc"] = roc_auc_score(y[:, j], p[:, j])
        row[f"{n}_auprc"] = average_precision_score(y[:, j], p[:, j])
    return row


def run_joint_prediction(data, cfg):
    """Cross-fitted comparison; chains consume predicted, never observed, upstream labels at test time."""
    features = [
        "Age",
        "Sex",
        "Course_Type",
        "Ethnicity",
        "Institution",
        "SCI_Insomnia",
        "PSS_Stress",
        "MDQ_Mania",
        "P16_Psychotic_Exp_Sum",
        "UCLA3_Loneliness",
    ]
    X = data[features]
    y = data[OUTCOMES].astype(int).to_numpy()
    strata = np.array([int("".join(map(str, row)), 2) for row in y])
    cv = StratifiedKFold(
        cfg["multimorbidity"]["joint_cv_folds"],
        shuffle=True,
        random_state=cfg["project"]["random_seed"],
    )
    splits = list(cv.split(X, strata))
    base = Pipeline(
        [
            ("pre", preprocessor(X)),
            ("clf", LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced")),
        ]
    )
    independent = np.zeros_like(y, dtype=float)
    for j in range(3):
        independent[:, j] = cross_val_predict(
            clone(base), X, y[:, j], cv=splits, method="predict_proba", n_jobs=1
        )[:, 1]
    # The preprocessing step is shared and fitted within each fold; three chain orders test label-dependence robustness.
    rows = [metrics(y, independent, "Independent logistic models")]
    pred_store = {"independent": independent}
    for order in ([0, 1, 2], [1, 0, 2], [2, 0, 1]):
        chain = Pipeline(
            [
                ("pre", preprocessor(X)),
                (
                    "chain",
                    ClassifierChain(
                        LogisticRegression(
                            C=0.1, max_iter=5000, class_weight="balanced"
                        ),
                        order=list(order),
                        cv=3,
                        random_state=2026,
                    ),
                ),
            ]
        )
        # Manual out-of-fold loop avoids sklearn's class-order enforcement bug for
        # multilabel estimators while preserving strictly unseen-fold predictions.
        p = np.zeros_like(y, dtype=float)
        for train_idx, test_idx in splits:
            fitted = clone(chain).fit(X.iloc[train_idx], y[train_idx])
            p[test_idx] = fitted.predict_proba(X.iloc[test_idx])
        label = "Classifier chain " + "→".join(str(x + 1) for x in order)
        rows.append(metrics(y, p, label))
        pred_store[label] = p
    result = pd.DataFrame(rows)
    result.to_csv(
        cfg["outputs"]["tables_dir"] / "table_joint_multilabel_prediction.csv",
        index=False,
    )
    np.savez_compressed(
        cfg["outputs"]["models_dir"] / "joint_cross_fitted_predictions.npz",
        y=y,
        **{f"p{i}": p for i, p in enumerate(pred_store.values())},
    )
    return result
