"""
Regenerates every figure and table in the README from results/runs_all.csv.
Figures -> figures/ (PNG 600dpi + PDF + EPS). Tables -> results/Table*.csv.

Fig. 1 and Fig. 6 need per-point raw arrays produced by generate_figure_data.py (run
that first, once -- it retrains one EnergyPhasePINN model, ~1 min on GPU). Figs. 2-5
and 7-9, and all 9 tables, only need results/runs_all.csv (already included).
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import pub_style as S

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
TAB_DIR = os.path.join(HERE, "results")
DATA_DIR = os.path.join(HERE, "results", "figure_data")  # from generate_figure_data.py
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

S.apply_style(font_size=7)
import matplotlib.pyplot as plt

df = pd.read_csv(os.path.join(HERE, "results", "runs_all.csv"))
gate = json.load(open(os.path.join(HERE, "results", "gate_decomposition.json"), encoding="utf-8"))

MODEL_LABELS = {"EnergyPhasePINN": "EnergyPhasePINN", "PhasePINN": "PhasePINN", "GPR": "GPR"}
MODEL_ORDER = ["EnergyPhasePINN", "PhasePINN", "GPR"]


def agg(sub, by="model"):
    return sub.groupby(by).agg(R2_mean=("R2", "mean"), R2_std=("R2", "std"),
                               MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
                               RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
                               n=("R2", "size"))


def fig1():
    d = np.load(os.path.join(DATA_DIR, "fig1_loading_branch.npz"))
    fig, axes = S.new_fig(width="2col", height_mm=65, ncols=2)

    ax = axes[0]
    m = d["contam_mask_old_test"]
    ax.scatter(d["y_true_old"][~m], d["y_pred_old"][~m], s=3, alpha=0.5,
               color=S.OKABE_ITO["blue"], label=f"clean loading (n={(~m).sum()})", rasterized=True)
    ax.scatter(d["y_true_old"][m], d["y_pred_old"][m], s=5, alpha=0.8,
               color=S.OKABE_ITO["vermillion"], label=f"unloading contaminant (n={m.sum()})",
               rasterized=True)
    lims = [0, max(d["y_true_old"].max(), d["y_pred_old"].max())]
    ax.plot(lims, lims, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Actual force (nN)")
    ax.set_ylabel("Predicted force (nN)")
    ax.set_title(f"(a) Original (depth-sorted) filter\n$R^2$={float(d['R2_old']):.3f}")
    ax.legend(loc="upper left", frameon=False, markerscale=2)

    ax = axes[1]
    ax.scatter(d["y_true_new"], d["y_pred_new"], s=3, alpha=0.5,
               color=S.OKABE_ITO["bluish_green"], label=f"n={len(d['y_true_new'])}", rasterized=True)
    lims2 = [0, max(d["y_true_new"].max(), d["y_pred_new"].max())]
    ax.plot(lims2, lims2, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Actual force (nN)")
    ax.set_ylabel("Predicted force (nN)")
    ax.set_title(f"(b) Corrected (time-ordered) filter\n$R^2$={float(d['R2_new']):.3f}")
    ax.legend(loc="upper left", frameon=False, markerscale=2)

    S.save(fig, FIG_DIR, "Fig1_loading_branch_correction")
    plt.close(fig)


def fig6():
    d = np.load(os.path.join(DATA_DIR, "fig6_force_decomposition.npz"))
    fig, axes = S.new_fig(width="1col", height_mm=110, nrows=2, sharex=True,
                          gridspec_kw={"height_ratios": [2, 1]})

    ax = axes[0]
    ax.plot(d["delta"], d["F_total"], color=S.OKABE_ITO["black"], lw=1.1, label="$F$ (total, exact)")
    ax.plot(d["delta"], d["F_weighted"], color=S.OKABE_ITO["blue"], lw=1.0, ls="--",
           label="weighted branch forces")
    ax.fill_between(d["delta"], d["F_weighted"], d["F_total"], color=S.OKABE_ITO["vermillion"],
                    alpha=0.35, label="gate-derivative term")
    ax.set_ylabel("Force (nN)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Force decomposition: $F=dU/d\\delta$")

    ax = axes[1]
    ax.plot(d["delta"], d["pi1"], color=S.OKABE_ITO["bluish_green"], label="$\\pi_1$ (plastic-on)")
    ax.plot(d["delta"], d["pi2"], color=S.OKABE_ITO["reddish_purple"], label="$\\pi_2$ (hardening-on)")
    ax.set_xlabel("Depth $\\delta$")
    ax.set_ylabel("Gate value")
    ax.legend(frameon=False, loc="center right")

    S.save(fig, FIG_DIR, "Fig6_gate_derivative_decomposition")


def fig2():
    sub = df[df.exp == "E3"]
    a = agg(sub).reindex(MODEL_ORDER)
    fig, ax = S.new_fig(width="1col", height_mm=70)
    x = np.arange(len(a))
    ax.bar(x, a.R2_mean, yerr=a.R2_std, capsize=2.5, color=S.PALETTE[:len(a)],
          edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in a.index], rotation=35, ha="right")
    ax.set_ylabel("$R^2$ (leave-one-condition-out)")
    ax.set_ylim(0, 1.05)
    for xi, (r2, n) in enumerate(zip(a.R2_mean, a.n)):
        ax.text(xi, 0.04, f"n={int(n)}", ha="center", va="bottom", fontsize=5, color="white")
    ax.set_title("Model comparison (mean $\\pm$ SD across folds/seeds)")
    S.save(fig, FIG_DIR, "Fig2_model_comparison_R2")


def fig3():
    e1 = agg(df[df.exp == "E1"]).reindex(MODEL_ORDER).dropna()
    e3 = agg(df[df.exp == "E3"]).reindex(e1.index)
    fig, ax = S.new_fig(width="1col", height_mm=70)
    x = np.arange(len(e1))
    w = 0.35
    ax.bar(x - w / 2, e1.R2_mean, w, yerr=e1.R2_std, capsize=2, label="pooled-random (leaky)",
          color=S.OKABE_ITO["vermillion"], edgecolor="black", linewidth=0.4)
    ax.bar(x + w / 2, e3.R2_mean, w, yerr=e3.R2_std, capsize=2, label="LOCO (corrected)",
          color=S.OKABE_ITO["blue"], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in e1.index], rotation=35, ha="right")
    ax.set_ylabel("$R^2$")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("Effect of point-wise data leakage")
    S.save(fig, FIG_DIR, "Fig3_leakage_effect")


def fig4():
    e2 = agg(df[df.exp == "E2"])
    e3e = agg(df[(df.exp == "E3") & (df.model == "EnergyPhasePINN")])
    labels = ["Original filter\n(depth-sorted)", "Corrected filter\n(time-ordered)"]
    means = [e2.R2_mean.iloc[0], e3e.R2_mean.iloc[0]]
    stds = [e2.R2_std.iloc[0], e3e.R2_std.iloc[0]]
    ns = [int(e2.n.iloc[0]), int(e3e.n.iloc[0])]
    fig, ax = S.new_fig(width="1col", height_mm=65)
    x = np.arange(2)
    ax.bar(x, means, yerr=stds, capsize=3, color=[S.OKABE_ITO["vermillion"], S.OKABE_ITO["blue"]],
          edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("$R^2$ (EnergyPhasePINN, LOCO)")
    ax.set_ylim(0, 1.05)
    for xi, (m_, n_) in enumerate(zip(means, ns)):
        ax.text(xi, m_ + 0.03, f"n={n_}", ha="center", fontsize=5.5)
    ax.set_title("Loading-branch extraction: submitted vs corrected")
    S.save(fig, FIG_DIR, "Fig4_loading_filter_comparison")


def fig5():
    full = agg(df[df.exp == "E3"]).reindex(["EnergyPhasePINN", "PhasePINN"])
    nopow_map = {"EnergyPhasePINN": "EnergyPhasePINN_nopow", "PhasePINN": "PhasePINN_nopow"}
    e5 = agg(df[df.exp == "E5"])
    nopow = e5.reindex([nopow_map[m] for m in full.index])
    nopow.index = full.index

    fig, ax = S.new_fig(width="1col", height_mm=70)
    x = np.arange(len(full))
    w = 0.35
    ax.bar(x - w / 2, full.R2_mean, w, yerr=full.R2_std, capsize=2, label="with $\\delta$ powers",
          color=S.OKABE_ITO["blue"], edgecolor="black", linewidth=0.4)
    ax.bar(x + w / 2, nopow.R2_mean, w, yerr=nopow.R2_std, capsize=2, label="independent vars only",
          color=S.OKABE_ITO["orange"], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in full.index], rotation=35, ha="right")
    ax.set_ylabel("$R^2$ (LOCO)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("Depth-power feature ablation")
    S.save(fig, FIG_DIR, "Fig5_depth_power_ablation")


def fig7():
    e7 = df[df.exp == "E7"]
    fig, axes = S.new_fig(width="2col", height_mm=65, ncols=2)

    ax = axes[0]
    levels = np.linspace(0.05, 0.95, 19)
    # reconstruct mean empirical coverage per level from the 1/2-sigma PICPs isn't exact;
    # instead show the two reported coverage points against the diagonal, matching Table.
    ax.plot([0, 1], [0, 1], color="black", lw=0.8, ls="--", label="ideal")
    x_levels = [0.6827, 0.9545]  # +/-1 sigma, +/-2 sigma nominal Gaussian coverage
    y_emp = [e7.PICP_1sigma.mean(), e7.PICP_2sigma.mean()]
    y_err = [e7.PICP_1sigma.std(), e7.PICP_2sigma.std()]
    ax.errorbar(x_levels, y_emp, yerr=y_err, fmt="o", color=S.OKABE_ITO["vermillion"],
               capsize=3, label=f"observed (n={len(e7)})", markersize=4)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage (PICP)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("(a) Reliability")

    ax = axes[1]
    metrics = ["PICP_1sigma", "PICP_2sigma", "ECE", "MCE"]
    mlabels = ["PICP\n1$\\sigma$", "PICP\n2$\\sigma$", "ECE", "MCE"]
    means = [e7[m].mean() for m in metrics]
    stds = [e7[m].std() for m in metrics]
    xpos = np.arange(len(metrics))
    ax.bar(xpos, means, yerr=stds, capsize=2.5, color=S.PALETTE[:4], edgecolor="black", linewidth=0.5)
    ax.axhline(0.6827, color="black", lw=0.6, ls=":")
    ax.axhline(0.9545, color="black", lw=0.6, ls=":")
    ax.set_xticks(xpos); ax.set_xticklabels(mlabels)
    ax.set_title(f"(b) Calibration metrics (n={len(e7)})")

    S.save(fig, FIG_DIR, "Fig7_calibration")


def fig8():
    e3 = df[df.exp == "E3"]
    ep = e3[e3.model == "EnergyPhasePINN"].groupby("fold")["R2"].mean()
    pp = e3[e3.model == "PhasePINN"].groupby("fold")["R2"].mean()
    t, p = stats.ttest_rel(ep, pp)

    fig, ax = S.new_fig(width="1col", height_mm=75)
    for f in ep.index:
        ax.plot([0, 1], [pp[f], ep[f]], color="gray", lw=0.5, alpha=0.6, zorder=1)
    ax.scatter(np.zeros(len(pp)), pp, s=10, color=S.OKABE_ITO["blue"], zorder=2, label="PhasePINN")
    ax.scatter(np.ones(len(ep)), ep, s=10, color=S.OKABE_ITO["vermillion"], zorder=2,
              label="EnergyPhasePINN")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["PhasePINN", "EnergyPhasePINN"])
    ax.set_ylabel("$R^2$ (per held-out condition, n=18)")
    wins = int((ep > pp).sum())
    ax.set_title(f"Paired per-fold comparison\npaired $t$-test $p$={p:.3f}, "
                f"EnergyPhasePINN wins {wins}/18 folds")
    ax.legend(frameon=False, loc="lower right")
    S.save(fig, FIG_DIR, "Fig8_paired_fold_comparison")


def fig9():
    e3 = df[df.exp == "E3"]
    conds = pd.read_csv(os.path.join(HERE, "results", "fold_conditions.csv"), index_col="fold")
    fig, ax = S.new_fig(width="2col", height_mm=70)
    for name, color, marker in [("EnergyPhasePINN", S.OKABE_ITO["vermillion"], "o"),
                                ("PhasePINN", S.OKABE_ITO["blue"], "s"),
                                ("GPR", S.OKABE_ITO["black"], "^")]:
        fm = e3[e3.model == name].groupby("fold")["R2"].mean().sort_index()
        ax.plot(fm.index, fm.values, marker=marker, ms=3.5, lw=0.8, color=color, label=name)
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("Held-out condition (fold index)")
    ax.set_ylabel("$R^2$ (mean over seeds)")
    ax.set_xticks(conds.index)
    ax.set_xticklabels([f"{r.grain_size:g}/{r.temperature:g}/{r.velocity:g}"
                        for _, r in conds.iterrows()], rotation=90, fontsize=4.5)
    ax.set_xlabel("Held-out condition: grain size (nm) / T (K) / v (nm/s)")
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("Per-condition reliability: GPR degrades at design extremes; both PINNs stay robust")
    S.save(fig, FIG_DIR, "Fig9_per_fold_reliability")


def tables():
    t1 = agg(df[df.exp == "E3"]).reindex(MODEL_ORDER).round(4)
    t1.to_csv(os.path.join(TAB_DIR, "Table1_model_comparison_full_budget.csv"))

    # Fair-budget comparison: EnergyPhasePINN/PhasePINN capped at 3,000 training points
    # (E4), alongside GPR's own native run, which already uses 3,000 points (E3) -- so
    # all three rows in this table share the same training-set size.
    t2a = agg(df[df.exp == "E4"]).reindex(["EnergyPhasePINN", "PhasePINN"])
    t2b = agg(df[(df.exp == "E3") & (df.model == "GPR")])
    t2 = pd.concat([t2a, t2b]).round(4)
    t2.to_csv(os.path.join(TAB_DIR, "Table2_model_comparison_budget_matched.csv"))

    full = agg(df[df.exp == "E3"]).reindex(["EnergyPhasePINN", "PhasePINN"])
    nopow_map = {"EnergyPhasePINN": "EnergyPhasePINN_nopow", "PhasePINN": "PhasePINN_nopow"}
    e5 = agg(df[df.exp == "E5"])
    t3 = pd.DataFrame({
        "R2_with_powers": full.R2_mean, "R2_with_powers_std": full.R2_std,
        "R2_independent_only": [e5.loc[nopow_map[m], "R2_mean"] for m in full.index],
        "R2_independent_only_std": [e5.loc[nopow_map[m], "R2_std"] for m in full.index],
    }, index=full.index).round(4)
    t3.to_csv(os.path.join(TAB_DIR, "Table3_depth_power_ablation.csv"))

    e7 = df[df.exp == "E7"]
    t4 = e7[["PICP_1sigma", "PICP_2sigma", "ECE", "MCE", "NLL", "sharpness_mean_sigma"]].agg(
        ["mean", "std"]).T.round(4)
    t4.to_csv(os.path.join(TAB_DIR, "Table4_calibration.csv"))

    e2 = agg(df[df.exp == "E2"])
    e3e = agg(df[(df.exp == "E3") & (df.model == "EnergyPhasePINN")])
    t5 = pd.concat([e2, e3e]).round(4)
    t5.index = ["Original filter (depth-sorted)", "Corrected filter (time-ordered)"]
    t5.to_csv(os.path.join(TAB_DIR, "Table5_loading_filter_comparison.csv"))

    t6 = pd.DataFrame(gate).round(4)
    t6.to_csv(os.path.join(TAB_DIR, "Table6_gate_derivative_share.csv"), index=False)

    e6 = agg(df[df.exp == "E6"])
    e3e2 = agg(df[(df.exp == "E3") & (df.model == "EnergyPhasePINN")])
    t7 = pd.concat([e3e2, e6]).round(4)
    t7.index = ["EnergyPhasePINN (with residual)", "EnergyPhasePINN (no residual)"]
    t7.to_csv(os.path.join(TAB_DIR, "Table7_residual_ablation.csv"))

    sig = json.load(open(os.path.join(HERE, "results", "gpr_significance.json"), encoding="utf-8"))
    t9 = pd.DataFrame(sig).T.round(4)
    t9.to_csv(os.path.join(TAB_DIR, "Table9_significance_vs_GPR.csv"))

    print("Tables written to", TAB_DIR)
    for f in sorted(os.listdir(TAB_DIR)):
        print(" ", f)




if __name__ == "__main__":
    figs_needing_npz = [fig1, fig6]
    figs_from_summary = [fig2, fig3, fig4, fig5, fig7, fig8, fig9]
    for fn in figs_needing_npz:
        if os.path.exists(DATA_DIR):
            print(f"-- {fn.__name__} --")
            fn()
        else:
            print(f"-- skipping {fn.__name__} (run generate_figure_data.py first) --")
    for fn in figs_from_summary:
        print(f"-- {fn.__name__} --")
        fn()
    tables()
    print("\nDone. Figures in", FIG_DIR, "/ Tables in", TAB_DIR)
