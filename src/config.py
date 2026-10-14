from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    with open(path or ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for section in ("data", "outputs"):
        for key, value in cfg[section].items():
            if key.endswith("_dir"):
                cfg[section][key] = ROOT / value
    return cfg


def ensure_dirs(cfg: dict) -> None:
    for section in ("data", "outputs"):
        for key, value in cfg[section].items():
            if key.endswith("_dir"):
                Path(value).mkdir(parents=True, exist_ok=True)


def get_logger(name: str, cfg: dict) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        fh = logging.FileHandler(cfg["outputs"]["logs_dir"] / f"{name}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def load_first_dataset(raw_dir: Path) -> tuple[pd.DataFrame, Path]:
    files = sorted(
        p for p in raw_dir.iterdir() if p.suffix.lower() in {".csv", ".xlsx"}
    )
    if not files:
        raise FileNotFoundError(f"No CSV or XLSX dataset found in {raw_dir}")
    p = files[0]
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    else:
        df = pd.read_excel(p)
    return df, p


def safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def specificity(y, pred) -> float:
    y, pred = np.asarray(y), np.asarray(pred)
    neg = y == 0
    return float(np.mean(pred[neg] == 0)) if neg.any() else np.nan


def npv(y, pred) -> float:
    y, pred = np.asarray(y), np.asarray(pred)
    denom = np.sum(pred == 0)
    return float(np.sum((pred == 0) & (y == 0)) / denom) if denom else np.nan
