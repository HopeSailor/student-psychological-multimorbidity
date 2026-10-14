# Student Psychological Multimorbidity

Reproducible analysis code for **Psychological Multimorbidity and Risk
Stratification among University Students: A Safety-Oriented Machine Learning
Study**.

The repository contains only the final analysis used in the manuscript. It
does not contain questionnaire responses, intermediate datasets, fitted model
objects, generated tables, or figures.

## Analysis structure

```text
src/
├── data.py                 # public-data download and variable construction
├── descriptive.py          # manuscript descriptive tables
├── modeling.py             # single-outcome models and evaluation
├── multimorbidity/
│   ├── associations.py      # patterns, burden gradients, and GEE models
│   └── joint_models.py      # independent and classifier-chain models
├── pilot_validation.py     # independent Chinese pilot validation
├── figure_style.py         # shared journal figure style
├── figures.py              # final main and supplementary figures
└── run_analysis.py         # complete development-data workflow
```

## Reproduce the development analysis

Python 3.12 was used for the manuscript. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.run_analysis
```

The workflow downloads the fixed public Figshare record, verifies its file
checksums, constructs the prespecified outcomes and feature sets, fits the
models, and writes the manuscript tables and figures under `outputs/`.

The source dataset is available from Figshare:
[10.6084/m9.figshare.24052236.v2](https://doi.org/10.6084/m9.figshare.24052236.v2).

## Reproduce the pilot validation

The Chinese pilot questionnaire contains sensitive mental-health and
suicide-related responses and is not included in this repository. After
obtaining an authorised copy of the original Wenjuanxing text-export workbook,
run:

```bash
python -m src.pilot_validation --input /path/to/questionnaire_export.xlsx
```

This command applies the frozen low-burden elastic-net models and the
development-sample thresholds. Generated row-level and aggregate outputs are
written under `outputs/pilot_validation/` and are excluded from version
control.

De-identified pilot data may be requested from the corresponding author. Any
release is subject to institutional review, applicable ethics requirements,
and a data-use agreement; access is not guaranteed. Requests should describe
the research purpose, variables required, analysis plan, data-security
arrangements, and proposed deletion date.

## Data protection

`data/` and `outputs/` are excluded by `.gitignore`. Do not commit raw survey
exports, IP addresses, timestamps, direct or indirect identifiers, row-level
scores, or fitted artifacts derived from restricted participant data.

## License

Code is released under the MIT License. The public dataset remains subject to
the terms specified by its original authors and repository.
