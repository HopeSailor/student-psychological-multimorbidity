from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

from .figure_style import COLORS, clean_axis, panel_label, save_figure, set_nature_style

# Explicit publication/export contract (also mirrored by the shared helper).
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
EXPORT_FORMATS = (".svg", ".pdf", ".tiff", ".png")
RASTER_DPI = 600

OUTCOMES = ["depression_risk", "anxiety_risk", "suicidal_behavior_risk"]
SHORT = {
    "depression_risk": "Depression",
    "anxiety_risk": "Anxiety",
    "suicidal_behavior_risk": "SBQ-R risk indicator",
}


def figure1_framework(cfg):
    set_nature_style()
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    ax.axis("off")
    blocks = [
        ("Public student\nmental-health data", 0.06, COLORS["grey"]),
        ("Three self-report\nrisk outcomes", 0.25, COLORS["navy"]),
        ("Psychological risk\nmultimorbidity", 0.44, COLORS["purple"]),
        ("Shared/outcome-\nspecific markers", 0.63, COLORS["orange"]),
        ("Joint modelling,\ncalibration,\nthreshold safety", 0.82, COLORS["green"]),
    ]
    for text, x, c in blocks:
        ax.text(
            x,
            0.60,
            text,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=6.5,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=.45", fc=c, ec="none"),
        )
    for i in range(len(blocks) - 1):
        ax.annotate(
            "",
            xy=(blocks[i + 1][1] - 0.075, 0.60),
            xytext=(blocks[i][1] + 0.075, 0.60),
            xycoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->", lw=1, color=COLORS["black"]),
        )
    save_figure(
        fig, cfg["outputs"]["figures_dir"] / "figure1_safety_multimorbidity_framework"
    )


def figure2_upset(patterns, cfg):
    set_nature_style()
    p = patterns.sort_values("n", ascending=False).reset_index(drop=True)
    fig = plt.figure(figsize=(7.2, 4.7))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1.3, 4], height_ratios=[3, 1.6], hspace=0.05, wspace=0.08
    )
    ax_set = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_dot = fig.add_subplot(gs[1, 1])
    ax_empty = fig.add_subplot(gs[1, 0])
    ax_empty.axis("off")
    preval = [p.loc[p[o] == 1, "n"].sum() for o in OUTCOMES]
    ax_set.barh(range(3), preval, color=COLORS["light_grey"])
    ax_set.set_yticks(range(3), [SHORT[o] for o in OUTCOMES])
    ax_set.invert_xaxis()
    ax_set.set_xlabel("Set size")
    clean_axis(ax_set)
    panel_label(ax_set, "a")
    x = np.arange(len(p))
    ax_bar.bar(x, p.n, color=COLORS["navy"], width=0.72)
    for i, n in enumerate(p.n):
        ax_bar.text(i, n + 10, str(n), ha="center", va="bottom", fontsize=6)
    ax_bar.set_ylabel("Intersection size")
    ax_bar.set_xticks([])
    clean_axis(ax_bar)
    panel_label(ax_bar, "b", x=-0.06)
    for i, row in p.iterrows():
        ys = []
        for j, o in enumerate(OUTCOMES):
            active = int(row[o]) == 1
            ax_dot.scatter(
                i,
                j,
                s=25,
                color=COLORS["black"] if active else COLORS["light_grey"],
                zorder=3,
            )
            if active:
                ys.append(j)
        if len(ys) > 1:
            ax_dot.plot([i, i], [min(ys), max(ys)], color=COLORS["black"], lw=1.3)
    ax_dot.set_yticks(range(3), [SHORT[o] for o in OUTCOMES])
    ax_dot.set_xticks(x, [str(i + 1) for i in x])
    ax_dot.set_xlabel("Risk-pattern rank")
    ax_dot.set_ylim(-0.5, 2.5)
    ax_dot.invert_yaxis()
    clean_axis(ax_dot)
    save_figure(fig, cfg["outputs"]["figures_dir"] / "figure2_multimorbidity_upset")


def figure3_burden_gradients(grad, cfg):
    set_nature_style()
    labels = ["stress", "sleep", "loneliness", "psychotic_experiences", "mania"]
    display = {
        "stress": "Stress",
        "sleep": "Poor sleep\n(32 - SCI)",
        "loneliness": "Loneliness",
        "psychotic_experiences": "Psychotic experiences",
        "mania": "Mania",
    }
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.25), sharex=True)
    for k, (ax, label) in enumerate(zip(axes, labels)):
        g = grad[grad.marker == label].sort_values("burden").copy()
        y = 32 - g["mean"] if label == "sleep" else g["mean"]
        ax.errorbar(
            g.burden,
            y,
            yerr=1.96 * g.se,
            marker="o",
            ms=3.5,
            capsize=2,
            color=COLORS["navy"],
        )
        ax.set_title(display[label])
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xlabel("Risk burden")
        clean_axis(ax)
        panel_label(ax, chr(97 + k), x=-0.2)
        if k == 0:
            ax.set_ylabel("Mean score (95% CI)")
    save_figure(fig, cfg["outputs"]["figures_dir"] / "figure3_risk_burden_gradients")


def figure4_forest(effects, cfg):
    set_nature_style()
    markers = list(effects.marker.unique())
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.25), sharey=True)
    cols = [COLORS["navy"], COLORS["orange"], COLORS["red"]]
    for k, (ax, outcome, c) in enumerate(zip(axes, OUTCOMES, cols)):
        g = (
            effects[effects.outcome == outcome]
            .set_index("marker")
            .reindex(markers)
            .reset_index()
        )
        y = np.arange(len(g))
        if np.any(g[["odds_ratio", "ci_low", "ci_high"]].to_numpy() <= 0):
            raise ValueError(
                "Odds ratios and confidence limits must be strictly positive for a log axis"
            )
        ax.errorbar(
            g.odds_ratio,
            y,
            xerr=[g.odds_ratio - g.ci_low, g.ci_high - g.odds_ratio],
            fmt="o",
            ms=3.5,
            capsize=2,
            color=c,
        )
        ax.axvline(1, color=COLORS["grey"], lw=0.7, ls="--")
        ax.set_xscale("log")
        ax.set_xlim(0.28, 7.2)
        ax.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 4]))
        ax.xaxis.set_major_formatter(FixedFormatter(["0.5", "1", "2", "4"]))
        ax.minorticks_off()
        ax.set_xlabel("Odds ratio per 1 SD")
        ax.set_title(SHORT[outcome])
        clean_axis(ax)
        panel_label(ax, chr(97 + k), x=-0.18)
        display = {
            "sleep": "Sleep condition",
            "stress": "Stress",
            "mania": "Mania",
            "psychotic_experiences": "Psychotic experiences",
            "loneliness": "Loneliness",
        }
        ax.set_yticks(y, [display.get(m, m.replace("_", " ").title()) for m in markers])
        ax.invert_yaxis()
    save_figure(
        fig, cfg["outputs"]["figures_dir"] / "figure4_shared_outcome_specific_markers"
    )


def figure5_profiles(profile_markers, profiles, cfg):
    set_nature_style()
    profile_markers = profile_markers.copy()
    profile_markers["sleep"] = -profile_markers["sleep"]
    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1.1, 2.2]}
    )
    axes[0].barh(
        range(len(profiles)),
        profiles.n,
        color=[
            COLORS["grey"],
            COLORS["navy"],
            COLORS["purple"],
            COLORS["gold"],
            COLORS["orange"],
            COLORS["red"],
        ],
    )
    axes[0].set_yticks(
        range(len(profiles)),
        profiles.profile.str.replace(" psychological", "", regex=False),
    )
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Students (n)")
    clean_axis(axes[0])
    panel_label(axes[0], "a")
    im = axes[1].imshow(
        profile_markers.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-1.2, vmax=1.2
    )
    display = {
        "sleep": "Poor sleep",
        "stress": "Stress",
        "mania": "Mania",
        "psychotic_experiences": "Psychotic experiences",
        "loneliness": "Loneliness",
    }
    axes[1].set_xticks(
        range(profile_markers.shape[1]),
        [display.get(x, x.replace("_", " ").title()) for x in profile_markers.columns],
        rotation=35,
        ha="right",
    )
    axes[1].set_yticks(range(profile_markers.shape[0]), [""] * profile_markers.shape[0])
    axes[1].tick_params(length=0)
    panel_label(axes[1], "b", x=-0.08)
    cb = fig.colorbar(im, ax=axes[1], fraction=0.035, pad=0.03)
    cb.set_label("Standardised mean (z)")
    save_figure(fig, cfg["outputs"]["figures_dir"] / "figure5_multimorbidity_profiles")


def figure6_joint_prediction(results, cfg):
    set_nature_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    r = results.copy()
    labels = ["Independent"] + [f"Chain {i}" for i in range(1, len(r))]
    x = np.arange(len(r))
    axes[0].plot(x, r.macro_auroc, "o-", color=COLORS["navy"], label="Macro-AUROC")
    axes[0].plot(x, r.macro_auprc, "s-", color=COLORS["orange"], label="Macro-AUPRC")
    axes[0].set_ylim(0.82, 0.89)
    axes[0].set_ylabel("Discrimination")
    axes[0].legend(frameon=False)
    clean_axis(axes[0])
    panel_label(axes[0], "a")
    axes[1].plot(
        x,
        r.exact_match_accuracy,
        "o-",
        color=COLORS["green"],
        label="Exact match accuracy",
    )
    axes[1].plot(
        x, 1 - r.hamming_loss, "s-", color=COLORS["purple"], label="1 − Hamming loss"
    )
    axes[1].set_ylim(0.45, 1)
    axes[1].set_ylabel("Joint-label performance")
    axes[1].legend(frameon=False)
    clean_axis(axes[1])
    panel_label(axes[1], "b")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=25, ha="right")
    save_figure(
        fig, cfg["outputs"]["figures_dir"] / "figure6_joint_multilabel_prediction"
    )


def figure7_model_validation(cfg):
    set_nature_style()
    tabs = cfg["outputs"]["tables_dir"]
    perf = pd.read_csv(tabs / "table_model_performance_all.csv")
    cal = pd.read_csv(tabs / "table_calibration_metrics.csv")
    outcomes = [
        ("depression_risk", "Depression"),
        ("anxiety_risk", "Anxiety"),
        ("suicidal_behavior_risk", "SBQ-R indicator"),
    ]
    fig = plt.figure(figsize=(7.2, 4.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.48, wspace=0.35)
    colors = {
        "logistic": COLORS["navy"],
        "elastic_net": COLORS["orange"],
        "random_forest": COLORS["grey"],
    }
    markers = {"low_burden": "o", "extended": "s"}
    for j, (outcome, title) in enumerate(outcomes):
        ax = fig.add_subplot(gs[0, j])
        g = perf[perf.outcome == outcome]
        ybase = np.arange(3)
        for fs, off in [("low_burden", -0.12), ("extended", 0.12)]:
            h = (
                g[g.feature_set == fs]
                .set_index("model")
                .reindex(["logistic", "elastic_net", "random_forest"])
            )
            for i, m in enumerate(h.index):
                ax.scatter(
                    h.loc[m, "auroc"],
                    i + off,
                    s=22,
                    marker=markers[fs],
                    color=colors[m],
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=3,
                )
        ax.set_yticks(
            ybase,
            ["Logistic", "Elastic net", "Random forest"] if j == 0 else ["", "", ""],
        )
        lo = max(0.72, g.auroc.min() - 0.03)
        ax.set_xlim(lo, min(1, g.auroc.max() + 0.03))
        ax.set_xlabel("Test AUROC")
        ax.set_title(title)
        clean_axis(ax)
        panel_label(ax, chr(97 + j), x=-0.18)
    # Calibration intercept and slope as two compact validation panels plus legend.
    ax_i = fig.add_subplot(gs[1, 0])
    ax_s = fig.add_subplot(gs[1, 1])
    ax_l = fig.add_subplot(gs[1, 2])
    ax_l.axis("off")
    cs = [COLORS["navy"], COLORS["orange"], COLORS["red"]]
    labs = [x[1] for x in outcomes]
    for k, ((outcome, _), c) in enumerate(zip(outcomes, cs)):
        r = cal[cal.outcome == outcome].iloc[0]
        ax_i.scatter(k, r.intercept, color=c, s=25)
        ax_s.scatter(k, r.slope, color=c, s=25)
    ax_i.axhline(0, color=COLORS["grey"], ls="--", lw=0.8)
    ax_s.axhline(1, color=COLORS["grey"], ls="--", lw=0.8)
    for ax, title, letter in [
        (ax_i, "Calibration intercept", "d"),
        (ax_s, "Calibration slope", "e"),
    ]:
        ax.set_xticks(
            range(3), ["Depression", "Anxiety", "SBQ-R"], rotation=25, ha="right"
        )
        ax.set_title(title)
        clean_axis(ax)
        panel_label(ax, letter, x=-0.18)
    handles = [
        Line2D(
            [0],
            [0],
            marker=markers[fs],
            color="none",
            markerfacecolor=COLORS["black"],
            label=fs.replace("_", " ").title(),
        )
        for fs in markers
    ]
    handles += [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=colors[m],
            label=m.replace("_", " ").title(),
        )
        for m in colors
    ]
    ax_l.legend(handles=handles, loc="center left", frameon=False, title="Encoding")
    save_figure(fig, cfg["outputs"]["figures_dir"] / "figure7_model_validation")


def figure8_safety_utility(cfg):
    set_nature_style()
    tabs = cfg["outputs"]["tables_dir"]
    dca = pd.read_csv(tabs / "table_decision_curve_summary.csv")
    sel = pd.read_csv(tabs / "table_selected_models.csv")
    th = pd.read_csv(tabs / "table_safety_thresholds.csv")
    outcomes = [
        ("depression_risk", "Depression"),
        ("anxiety_risk", "Anxiety"),
        ("suicidal_behavior_risk", "SBQ-R indicator"),
    ]
    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1], hspace=0.48, wspace=0.34)
    for j, (outcome, title) in enumerate(outcomes):
        ax = fig.add_subplot(gs[0, j])
        g = dca[dca.outcome == outcome]
        for strategy, c, ls in [
            ("model-guided support", COLORS["navy"], "-"),
            ("support-all", COLORS["orange"], "--"),
            ("support-none", COLORS["grey"], ":"),
        ]:
            h = g[g.strategy == strategy]
            ax.plot(
                h.threshold,
                h.net_benefit,
                color=c,
                ls=ls,
                label=strategy.replace("support", "support").title(),
            )
        ax.axhline(0, color=COLORS["light_grey"], lw=0.6)
        ax.set_title(title)
        ax.set_xlabel("Threshold probability")
        if j == 0:
            ax.set_ylabel("Net benefit")
            ax.legend(frameon=False, fontsize=5.5)
        clean_axis(ax)
        panel_label(ax, chr(97 + j), x=-0.18)
        ax2 = fig.add_subplot(gs[1, j])
        sr = sel[sel.outcome == outcome].iloc[0]
        h = th[
            (th.outcome == outcome)
            & (th.feature_set == sr.feature_set)
            & (th.model == sr.model)
            & (th.threshold_type == "safety")
        ].iloc[0]
        vals = [h.sensitivity, h.specificity, h.ppv, h.npv, h.pct_flagged / 100]
        bars = ax2.bar(
            np.arange(5),
            vals,
            color=[
                COLORS["navy"],
                COLORS["blue"],
                COLORS["orange"],
                COLORS["teal"],
                COLORS["pale_red"],
            ],
            edgecolor=COLORS["black"],
            linewidth=0.4,
        )
        ax2.set_ylim(0, 1)
        ax2.set_xticks(
            range(5),
            ["Sens.", "Spec.", "PPV", "NPV", "Flagged"],
            rotation=35,
            ha="right",
        )
        if j == 0:
            ax2.set_ylabel("Proportion")
        ax2.text(
            0.98,
            0.94,
            f"threshold = {h.threshold:.2f}",
            transform=ax2.transAxes,
            ha="right",
            va="top",
            fontsize=6,
        )
        clean_axis(ax2)
        panel_label(ax2, chr(100 + j), x=-0.18)
    save_figure(fig, cfg["outputs"]["figures_dir"] / "figure8_safety_utility")


def supplementary1_correlations(data, cfg):
    set_nature_style()
    cols = [
        "GAD7_Anxiety",
        "PHQ9_Depression",
        "SCI_Insomnia",
        "PSS_Stress",
        "MDQ_Mania",
        "SBQ_Suicidal_Ideation",
        "P16_Psychotic_Exp_Sum",
        "UCLA3_Loneliness",
    ]
    labels = ["GAD-7", "PHQ-9", "SCI", "PSS", "MDQ", "SBQ-R", "PQ-16", "UCLA"]
    corr = data[cols].corr()
    fig, ax = plt.subplots(figsize=(4.6, 4.1))
    mask = np.triu(np.ones_like(corr, dtype=bool), 1)
    shown = np.ma.array(corr.to_numpy(), mask=mask)
    im = ax.imshow(shown, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1))
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(i + 1):
            ax.text(
                j,
                i,
                f"{corr.iloc[i,j]:.2f}",
                ha="center",
                va="center",
                fontsize=5,
                color="white" if abs(corr.iloc[i, j]) > 0.55 else COLORS["black"],
            )
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Pearson r")
    ax.tick_params(length=0)
    save_figure(
        fig, cfg["outputs"]["figures_dir"] / "supplementary_figure1_scale_correlations"
    )


def supplementary2_subgroups(cfg):
    set_nature_style()
    d = pd.read_csv(cfg["outputs"]["tables_dir"] / "table_subgroup_safety_combined.csv")
    overall = pd.read_csv(
        cfg["outputs"]["tables_dir"] / "table_selected_models.csv"
    ).set_index("outcome")["auroc"]
    outcomes = [
        ("depression_risk", "Depression"),
        ("anxiety_risk", "Anxiety"),
        ("suicidal_behavior_risk", "SBQ-R indicator"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.0), sharex=True)
    for k, (ax, (outcome, title)) in enumerate(zip(axes, outcomes)):
        g = d[d.outcome == outcome].copy()
        g["label"] = g.subgroup.str.replace("_", " ") + "=" + g.level.astype(str)
        g = g.sort_values("auroc")
        y = np.arange(len(g))
        sc = ax.scatter(
            g.auroc,
            y,
            c=g.false_negative_rate,
            cmap="magma",
            vmin=0,
            vmax=0.4,
            s=28,
            edgecolor="white",
            linewidth=0.4,
        )
        ax.axvline(overall.loc[outcome], color=COLORS["grey"], ls="--", lw=0.7)
        ax.set_yticks(y, g.label if k == 0 else [""] * len(g))
        ax.set_xlim(0.72, 0.98)
        ax.set_xlabel("Subgroup AUROC")
        ax.set_title(title)
        clean_axis(ax)
        panel_label(ax, chr(97 + k), x=-0.18)
    cb = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.03)
    cb.set_label("False-negative rate")
    save_figure(
        fig, cfg["outputs"]["figures_dir"] / "supplementary_figure2_subgroup_safety"
    )


def supplementary3_importance(cfg):
    set_nature_style()
    d = pd.read_csv(
        cfg["outputs"]["tables_dir"] / "table_feature_importance_top10_combined.csv"
    )
    outcomes = [
        ("depression_risk", "Depression"),
        ("anxiety_risk", "Anxiety"),
        ("suicidal_behavior_risk", "SBQ-R indicator"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.2), gridspec_kw={"wspace": 0.72})
    for k, (ax, (outcome, title)) in enumerate(zip(axes, outcomes)):
        g = d[d.outcome == outcome].sort_values("mean_abs_importance")
        y = np.arange(len(g))
        ax.errorbar(
            g.mean_abs_importance,
            y,
            xerr=g.importance_sd,
            fmt="o",
            ms=3,
            capsize=2,
            color=[COLORS["navy"], COLORS["orange"], COLORS["red"]][k],
        )
        ax.axvline(0, color=COLORS["grey"], lw=0.7, ls="--")
        labels = [x.replace("_", " ") for x in g.feature]
        ax.set_yticks(y, labels, fontsize=5.5)
        ax.set_xlabel("Permutation ΔAUROC")
        ax.set_title(title)
        clean_axis(ax)
        panel_label(ax, chr(97 + k), x=-0.18)
    save_figure(
        fig,
        cfg["outputs"]["figures_dir"] / "supplementary_figure3_permutation_importance",
    )


def make_all(results, joint, cfg, data=None):
    figure1_framework(cfg)
    figure2_upset(results["patterns8"], cfg)
    grad = pd.read_csv(cfg["outputs"]["tables_dir"] / "table_risk_burden_gradients.csv")
    figure3_burden_gradients(grad, cfg)
    figure4_forest(results["effects"], cfg)
    figure5_profiles(results["profile_markers"], results["profiles6"], cfg)
    figure6_joint_prediction(joint, cfg)
    figure7_model_validation(cfg)
    figure8_safety_utility(cfg)
    if data is not None:
        supplementary1_correlations(data, cfg)
    supplementary2_subgroups(cfg)
    supplementary3_importance(cfg)
