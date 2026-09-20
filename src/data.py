"""
Corrected data preparation for the JMSC revision.

Addresses Reviewer 6 Q5 (loading-branch extraction) and Q4 (group-wise splitting).

Key facts established from Indent_AlCuNiTiZr.csv:
  - 9,966 rows, 18 distinct (Grain size, Temperature, Velocity) conditions.
  - Row order within each condition IS the simulation time sequence: depth rises
    monotonically to 25 A then retracts monotonically. Verified for all 18/18
    conditions. No explicit time column is needed.
"""

import os

import numpy as np
import pandas as pd

KEYS = ["Grain size", "Temperature", "Velocity"]
TARGET = "Force"
CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "Indent_AlCuNiTiZr.csv")


def load(path=CSV):
    return pd.read_csv(path)


def check_time_ordering(df):
    """Confirm row order is the time sequence: depth up to a single apex, then down."""
    bad = []
    for key, g in df.groupby(KEYS, sort=False):
        d = g["Depth"].values
        i = int(np.argmax(d))
        up = np.all(np.diff(d[: i + 1]) >= -1e-12)
        down = np.all(np.diff(d[i:]) <= 1e-12)
        if not (up and down):
            bad.append(key)
    return bad


def keep_loading(df):
    """Loading branch = rows up to the depth apex, in ORIGINAL (time) order.

    The submitted version sorted each condition by Depth and cut at argmax(Force).
    Sorting by depth interleaves loading and unloading points that share a depth,
    so 533 unloading rows (307 of them below 10 nN) survived into the training set
    -- the near-zero-force / ~100 nN-prediction cluster Reviewer 6 flags in Fig. 11.
    """
    keep = []
    for _, g in df.groupby(KEYS, sort=False):
        apex = int(np.argmax(g["Depth"].values))
        keep.extend(g.index[: apex + 1].tolist())
    return df.loc[sorted(keep)].copy()


def add_features(df, powers=("sqrt", "32", "52")):
    """Depth powers. Pass powers=() for the Reviewer 2 Q6 independent-variables-only ablation."""
    out = df.copy()
    d = np.clip(out["Depth"].values.astype(np.float32), 0, None)
    if "sqrt" in powers:
        out["Depth_sqrt"] = np.sqrt(d)
    if "32" in powers:
        out["Depth_32"] = d ** 1.5
    if "52" in powers:
        out["Depth_52"] = d ** 2.5
    return out


def group_id(df):
    """Integer condition id, one per (g, T, v) curve -- the grouping unit for CV."""
    gk = df[KEYS].astype(str).agg("|".join, axis=1)
    return gk.astype("category").cat.codes.values


def loco_folds(df):
    """Leave-one-condition-out folds: 18 folds, one held-out curve each."""
    gid = group_id(df)
    return [(np.where(gid != u)[0], np.where(gid == u)[0]) for u in np.unique(gid)]


def describe_design(df):
    """The (g,T,v) design, which determines how hard each LOCO fold is."""
    rows = []
    base = None
    for key, g in df.groupby(KEYS, sort=False):
        rows.append(dict(zip(KEYS, key), n=len(g)))
    d = pd.DataFrame(rows)
    for c in KEYS:
        mode = d[c].mode()[0]
        d[c + "_varied"] = d[c] != mode
    d["n_varied"] = d[[c + "_varied" for c in KEYS]].sum(axis=1)
    return d


if __name__ == "__main__":
    df = load()
    bad = check_time_ordering(df)
    print(f"rows={len(df)}  conditions={df.groupby(KEYS).ngroups}")
    print(f"conditions violating monotone up/down depth: {len(bad)}")

    old_n = 9299  # what the submitted keep_loading() retained
    dl = keep_loading(df)
    print(f"loading rows: submitted filter={old_n}, time-ordered={len(dl)}")

    folds = loco_folds(dl)
    print(f"LOCO folds: {len(folds)}")
    sizes = [len(te) for _, te in folds]
    print(f"  held-out size: min={min(sizes)} max={max(sizes)} mean={np.mean(sizes):.0f}")

    des = describe_design(df)
    print("\nDesign (one-factor-at-a-time star about the modal condition):")
    print(des.to_string(index=False))
