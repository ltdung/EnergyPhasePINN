"""
Generates the per-point raw arrays needed for Fig. 1 (loading-branch contamination) and
Fig. 6 (gate-derivative force decomposition) that are not already summarized in
results/runs.jsonl. Saves to results/figure_data/*.npz. Deterministic (seed=0).
"""

import os

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import data as D
import models as M
import train as R

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "figure_data")  # per-point arrays for Fig 1 / Fig 6
os.makedirs(OUT, exist_ok=True)


def old_keep_loading(df):
    keys = D.KEYS
    d2 = df.copy()
    d2["gk"] = d2[keys].astype(str).agg("|".join, axis=1)
    keep = []
    for _, g in d2.groupby("gk"):
        g = g.sort_values("Depth")
        im = int(np.argmax(g["Force"].values))
        keep.extend(g.iloc[:im + 1].index.tolist())
    return df.loc[sorted(set(keep))].copy()


def main():
    df_raw = D.load()
    df_new = D.keep_loading(df_raw)
    df_old = old_keep_loading(df_raw)

    # ---------------- Fig 1 data: old-filter vs corrected-filter contamination ----------
    delta_o, ctx_o, y_o, gid_o = R.build_arrays(df_old)
    delta_n, ctx_n, y_n, gid_n = R.build_arrays(df_new)

    old_idx = set(df_old.index)
    new_idx = set(df_new.index)
    contam_idx = sorted(old_idx - new_idx)  # rows old filter kept but corrected filter dropped

    res_o, model_o, (t_o, p_o, ite_o) = R.run_once(df_old, M.EnergyPhasePINN, "loco", 0, 0,
                                                    epochs=800)
    res_n, model_n, (t_n, p_n, ite_n) = R.run_once(df_new, M.EnergyPhasePINN, "loco", 0, 0,
                                                    epochs=800)

    contam_in_test = df_old.index[ite_o].isin(contam_idx)
    np.savez(os.path.join(OUT, "fig1_loading_branch.npz"),
             y_true_old=t_o, y_pred_old=p_o, contam_mask_old_test=contam_in_test,
             y_true_new=t_n, y_pred_new=p_n,
             R2_old=res_o["R2"], R2_new=res_n["R2"],
             n_contam_total=len(contam_idx),
             n_contam_low_force=int((df_old.loc[contam_idx, "Force"] < 10).sum()))
    torch.save(model_o.state_dict(), os.path.join(HERE, "results", "checkpoints",
                                                   "EnergyPhasePINN_oldfilter_loco_fold0_seed0.pt"))
    print("Fig1 data saved. R2_old=%.3f R2_new=%.3f contaminants=%d" %
          (res_o["R2"], res_n["R2"], len(contam_idx)))

    # ---------------- Fig 6 data: force decomposition vs depth ----------------------------
    torch.manual_seed(0)
    itr, iva, ite = R.make_split(gid_n, "loco", 0, 0)
    sc = StandardScaler().fit(ctx_n[itr])
    cs = sc.transform(ctx_n).astype(np.float32)

    # sweep depth at the modal (median) context of the held-out condition
    ctx_fixed = cs[ite].mean(axis=0, keepdims=True)
    d_grid = np.linspace(0.01, float(delta_n[ite].max()), 400).reshape(-1, 1).astype(np.float32)
    ctx_grid = np.repeat(ctx_fixed, len(d_grid), axis=0)

    d_t = R.to_t(d_grid, grad=True)
    c_t = R.to_t(ctx_grid)
    Ft, Fw, Fg, pi1, pi2 = model_n.force_decomposition(d_t, c_t)

    np.savez(os.path.join(OUT, "fig6_force_decomposition.npz"),
             delta=d_grid.ravel(),
             F_total=Ft.cpu().numpy().ravel(),
             F_weighted=Fw.cpu().numpy().ravel(),
             F_gate=Fg.cpu().numpy().ravel(),
             pi1=pi1.cpu().numpy().ravel(),
             pi2=pi2.cpu().numpy().ravel())
    print("Fig6 data saved.")


if __name__ == "__main__":
    main()
