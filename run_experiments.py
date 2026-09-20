"""
Full experiment grid: EnergyPhasePINN vs. PhasePINN vs. GPR only.

Resumable: appends to results/runs.jsonl, skips already-completed `key`s on restart.

    python run_experiments.py            # run everything
    python run_experiments.py --quick    # 1 seed / 3 folds, for a fast smoke test

Experiments
-----------
E1  leakage        : pooled-random split vs. LOCO, same models              (data leakage)
E2  loading_branch : depth-sorted (buggy) vs. time-ordered loading-branch extraction
E3  main            : EnergyPhasePINN / PhasePINN / GPR, LOCO, full training budget
E4  budget_matched  : EnergyPhasePINN / PhasePINN capped at GPR's native 3,000 points
E5  ablation_power  : depth-power features vs. independent variables only (PINNs only)
E6  ablation_residual : EnergyPhasePINN with/without its learned residual-energy term
E7  calibration     : uncertainty (sigma-head) coverage / calibration, EnergyPhasePINN
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import data as D
import models as M
import train as R
import gpr_baseline as GPR

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
RUNS = os.path.join(OUT, "runs.jsonl")

MODEL_FNS = {
    "EnergyPhasePINN": lambda: M.EnergyPhasePINN(),
    "PhasePINN": lambda: M.PhasePINN(),
    "EnergyPhasePINN_nopow": lambda: M.EnergyPhasePINN(res_powers=()),
    "PhasePINN_nopow": lambda: M.PhasePINN(powers=()),
    "EnergyPhasePINN_nores": lambda: M.EnergyPhasePINN(use_residual=False),
}


def done_keys():
    if not os.path.exists(RUNS):
        return set()
    keys = set()
    with open(RUNS, encoding="utf-8") as f:
        for line in f:
            try:
                keys.add(json.loads(line)["key"])
            except Exception:
                pass
    return keys


def append(rec):
    with open(RUNS, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def run_torch(df_load, model_fn, protocol, fold, seed, budget=None, with_sigma=False,
             epochs=800):
    if budget is None:
        res, model, _ = R.run_once(df_load, model_fn, protocol, fold, seed, epochs=epochs,
                                   with_sigma=with_sigma)
        return res
    torch.manual_seed(seed)
    np.random.seed(seed)
    delta, ctx, y, gid = R.build_arrays(df_load)
    itr, iva, ite = R.make_split(gid, protocol, fold, seed)
    rng = np.random.RandomState(seed)
    if len(itr) > budget:
        itr = rng.choice(itr, budget, replace=False)
    sc = StandardScaler().fit(ctx[itr])
    cs = sc.transform(ctx).astype(np.float32)
    pack = lambda i: (R.to_t(delta[i], grad=True), R.to_t(cs[i]), R.to_t(y[i]))
    tr, va, te = pack(itr), pack(iva), pack(ite)
    model = model_fn().to(R.DEVICE)
    model, best = R.fit(model, tr, va, epochs=epochs)
    model.eval()
    mu, _ = model(te[0], te[1], create_graph=False)
    res = R.regression_metrics(y[ite].ravel(), mu.detach().cpu().numpy().ravel())
    res.update(protocol=protocol, fold=int(fold), seed=int(seed),
               n_train=int(len(itr)), n_test=int(len(ite)), best_epoch=best["ep"])
    return res


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


def main(quick=False):
    df_old = D.load()
    df_new = D.keep_loading(df_old)
    df_oldfilter = old_keep_loading(df_old)

    n_folds = 18
    seeds_main = [0] if quick else [0, 1, 2, 3, 4]
    seeds_abl = [0] if quick else [0, 1, 2]
    folds = range(3) if quick else range(n_folds)

    have = done_keys()
    jobs = []

    def add(key, fn):
        if key not in have:
            jobs.append((key, fn))

    for name in ["EnergyPhasePINN", "PhasePINN"]:
        for f in folds:
            for s in seeds_main:
                add(f"E3|{name}|loco|{f}|{s}",
                    lambda n=name, f=f, s=s: ("E3", n, run_torch(df_new, MODEL_FNS[n], "loco", f, s)))
    for f in folds:
        for s in seeds_abl:
            add(f"E3|GPR|loco|{f}|{s}",
                lambda f=f, s=s: ("E3", "GPR", GPR.run_gpr(df_new, "loco", f, s)[0]))

    for name in ["EnergyPhasePINN", "PhasePINN"]:
        for s in seeds_main:
            add(f"E1|{name}|random|0|{s}",
                lambda n=name, s=s: ("E1", n, run_torch(df_new, MODEL_FNS[n], "random", 0, s)))
    for s in seeds_abl:
        add(f"E1|GPR|random|0|{s}",
            lambda s=s: ("E1", "GPR", GPR.run_gpr(df_new, "random", 0, s)[0]))

    for f in folds:
        for s in seeds_abl:
            add(f"E2|EnergyPhasePINN_oldfilter|loco|{f}|{s}",
                lambda f=f, s=s: ("E2", "EnergyPhasePINN_oldfilter",
                                  run_torch(df_oldfilter, MODEL_FNS["EnergyPhasePINN"], "loco", f, s)))

    for name in ["EnergyPhasePINN", "PhasePINN"]:
        for f in folds:
            for s in seeds_abl:
                add(f"E4|{name}|loco|{f}|{s}",
                    lambda n=name, f=f, s=s: ("E4", n,
                                              run_torch(df_new, MODEL_FNS[n], "loco", f, s, budget=3000)))

    for name in ["EnergyPhasePINN_nopow", "PhasePINN_nopow"]:
        for f in folds:
            for s in seeds_abl:
                add(f"E5|{name}|loco|{f}|{s}",
                    lambda n=name, f=f, s=s: ("E5", n, run_torch(df_new, MODEL_FNS[n], "loco", f, s)))

    for f in folds:
        for s in seeds_abl:
            add(f"E6|EnergyPhasePINN_nores|loco|{f}|{s}",
                lambda f=f, s=s: ("E6", "EnergyPhasePINN_nores",
                                  run_torch(df_new, MODEL_FNS["EnergyPhasePINN_nores"], "loco", f, s)))

    for f in folds:
        for s in seeds_main:
            add(f"E7|EnergyPhasePINN_sigma|loco|{f}|{s}",
                lambda f=f, s=s: ("E7", "EnergyPhasePINN_sigma",
                                  run_torch(df_new, MODEL_FNS["EnergyPhasePINN"], "loco", f, s,
                                            with_sigma=True)))

    print(f"{len(jobs)} jobs to run ({len(have)} already done)")
    t0 = time.time()
    for i, (key, fn) in enumerate(jobs, 1):
        try:
            exp, name, res = fn()
            res.update(key=key, exp=exp, model=name)
            append(res)
        except Exception as e:
            append({"key": key, "error": repr(e)})
            print(f"  [{i}] FAILED {key}: {e!r}")
        if i % 10 == 0 or i == len(jobs):
            el = time.time() - t0
            print(f"  [{i}/{len(jobs)}] {el/60:.1f} min elapsed", flush=True)
    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 seed, 3 folds, for a smoke test")
    main(quick=ap.parse_args().quick)
