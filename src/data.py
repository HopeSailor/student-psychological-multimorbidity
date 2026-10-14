"""Download, audit, and prepare the public development dataset."""

from __future__ import annotations

import json
from hashlib import md5
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from .config import ensure_dirs, load_config, load_first_dataset


TOTALS = {
    "sci": "SCI_Insomnia",
    "gad": "GAD7_Anxiety",
    "phq": "PHQ9_Depression",
    "pss": "PSS_Stress",
    "mdq": "MDQ_Mania",
    "sbq": "SBQ_Suicidal_Ideation",
    "pq16": "P16_Psychotic_Exp_Sum",
    "ucla": "UCLA3_Loneliness",
}
DEMOGRAPHICS = ["Age", "Sex", "Course_Type", "Ethnicity", "Institution"]
LOW_BURDEN = DEMOGRAPHICS + [TOTALS[name] for name in ("sci", "pss", "ucla")]
EXTENDED = {
    "depression_risk": DEMOGRAPHICS
    + [TOTALS[name] for name in ("gad", "mdq", "sci", "pss", "pq16", "ucla")],
    "anxiety_risk": DEMOGRAPHICS
    + [TOTALS[name] for name in ("phq", "mdq", "sci", "pss", "pq16", "ucla")],
    "suicidal_behavior_risk": DEMOGRAPHICS
    + [TOTALS[name] for name in ("gad", "mdq", "sci", "pss", "pq16", "ucla")],
}


def download_public_data(cfg: dict) -> None:
    """Download the fixed Figshare record and verify each file checksum."""
    raw_dir = cfg["data"]["raw_dir"]
    if any(raw_dir.glob("*.xlsx")):
        return
    article_id = cfg["data"]["figshare_article_id"]
    request = Request(
        f"https://api.figshare.com/v2/articles/{article_id}",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    metadata = json.load(urlopen(request))
    for item in metadata["files"]:
        if Path(item["name"]).suffix.lower() not in {".xlsx", ".pdf"}:
            continue
        target = raw_dir / item["name"]
        request = Request(item["download_url"], headers={"User-Agent": "Mozilla/5.0"})
        target.write_bytes(urlopen(request).read())
        if md5(target.read_bytes()).hexdigest() != item["computed_md5"]:
            raise RuntimeError(f"Checksum mismatch for {target.name}")


def prepare_analysis_data(cfg: dict) -> pd.DataFrame:
    """Construct outcomes, multimorbidity profiles, and final feature manifests."""
    data, _ = load_first_dataset(cfg["data"]["raw_dir"])
    data = data.copy()
    required = list(TOTALS.values()) + DEMOGRAPHICS
    missing = [column for column in required if column not in data]
    if missing:
        raise KeyError(f"Required variables not found: {missing}")

    data["depression_risk"] = (
        data[TOTALS["phq"]].ge(10).where(data[TOTALS["phq"]].notna()).astype("Int64")
    )
    data["anxiety_risk"] = (
        data[TOTALS["gad"]].ge(10).where(data[TOTALS["gad"]].notna()).astype("Int64")
    )
    data["suicidal_behavior_risk"] = (
        data[TOTALS["sbq"]].ge(7).where(data[TOTALS["sbq"]].notna()).astype("Int64")
    )
    outcomes = ["depression_risk", "anxiety_risk", "suicidal_behavior_risk"]
    data["psychological_multimorbidity_burden"] = (
        data[outcomes].sum(axis=1, min_count=3).astype("Int64")
    )

    complete = data[outcomes].notna().all(axis=1)
    bits = pd.Series(pd.NA, index=data.index, dtype="string")
    bits.loc[complete] = (
        data.loc[complete, outcomes].astype(int).astype(str).agg("".join, axis=1)
    )
    data["risk_pattern_8"] = bits.map(
        {
            "000": "No elevated risk",
            "100": "Depression only",
            "010": "Anxiety only",
            "001": "SBQ-R indicator only",
            "110": "Depression + anxiety",
            "101": "Depression + SBQ-R indicator",
            "011": "Anxiety + SBQ-R indicator",
            "111": "All three elevated domains",
        }
    )
    data["multimorbidity_profile_6"] = bits.map(
        {
            "000": "No elevated risk",
            "100": "Single internalising risk",
            "010": "Single internalising risk",
            "110": "Depression-anxiety multimorbidity",
            "001": "SBQ-R indicator only",
            "101": "SBQ-R + one internalising risk",
            "011": "SBQ-R + one internalising risk",
            "111": "All three elevated domains",
        }
    )

    phq8_items = [f"PHQ_{index}" for index in range(1, 9)]
    data["PHQ8_sensitivity"] = data[phq8_items].sum(axis=1, min_count=8)
    keep = list(
        dict.fromkeys(
            required
            + phq8_items
            + ["PHQ_9", "PHQ8_sensitivity"]
            + outcomes
            + [
                "psychological_multimorbidity_burden",
                "risk_pattern_8",
                "multimorbidity_profile_6",
            ]
        )
    )
    prepared = data[[column for column in keep if column in data]].copy()
    prepared.to_csv(cfg["data"]["processed_dir"] / "analysis_dataset.csv", index=False)

    manifest = {
        "low_burden": LOW_BURDEN,
        "extended": EXTENDED,
        "exclusions": {
            "depression_risk": [
                "PHQ items",
                "PHQ9_Depression",
                "depression-derived labels",
            ],
            "anxiety_risk": ["GAD-7 items", "GAD7_Anxiety", "anxiety-derived labels"],
            "suicidal_behavior_risk": [
                "SBQ-R items",
                "SBQ_Suicidal_Ideation",
                "PHQ_9",
                "PHQ9_Depression",
            ],
        },
    }
    (cfg["data"]["processed_dir"] / "feature_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "outcome": outcome,
                "feature_set": feature_set,
                "features": "; ".join(features),
            }
            for outcome in EXTENDED
            for feature_set, features in (
                ("low_burden", LOW_BURDEN),
                ("extended", EXTENDED[outcome]),
            )
        ]
    ).to_csv(cfg["outputs"]["tables_dir"] / "table_feature_sets.csv", index=False)
    return prepared


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    download_public_data(cfg)
    prepare_analysis_data(cfg)


if __name__ == "__main__":
    main()
