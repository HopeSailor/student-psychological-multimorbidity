"""Run the complete manuscript analysis from the public development dataset."""

from __future__ import annotations

import warnings

from .config import ensure_dirs, load_config
from .data import download_public_data, prepare_analysis_data
from .descriptive import create_descriptive_tables
from .figures import make_all
from .modeling import post_evaluation, train_all
from .multimorbidity.associations import run_associations
from .multimorbidity.joint_models import run_joint_prediction


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    cfg = load_config()
    ensure_dirs(cfg)
    download_public_data(cfg)
    data = prepare_analysis_data(cfg)

    create_descriptive_tables(data, cfg)
    _, selected, predictions, data = train_all(cfg)
    post_evaluation(cfg, selected, predictions, data)

    complete = data.dropna(
        subset=["depression_risk", "anxiety_risk", "suicidal_behavior_risk"]
    ).copy()
    associations = run_associations(complete, cfg)
    joint_models = run_joint_prediction(complete, cfg)
    make_all(associations, joint_models, cfg, data)


if __name__ == "__main__":
    main()
