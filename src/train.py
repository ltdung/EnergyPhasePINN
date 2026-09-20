"""
Training / evaluation harness for the JMSC revision.

Protocols
---------
'random'  : pooled point-wise 80/20 split -- the SUBMITTED protocol, retained only
            to quantify how much the leakage inflated the reported metrics.
'loco'    : leave-one-condition-out over the 18 (g,T,v) curves (Reviewer 6 Q4).

In both cases the context scaler is fitted on TRAINING data only.
"""

import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import data as D
import models as M

DEVICE = M.DEVICE
CTX = D.KEYS


# ----------------------------------------------------------------- data -----
def build_arrays(df_load):
    delta = df_load["Depth"].values.astype(np.float32).reshape(-1, 1)
    ctx = df_load[CTX].values.astype(np.float32)
    y = df_load["Force"].values.astype(np.float32).reshape(-1, 1)
    gid = D.group_id(df_load)
    return delta, ctx, y, gid


def to_t(a, grad=False):
    t = torch.tensor(a, dtype=torch.float32, device=DEVICE)
    if grad:
        t.requires_grad_(True)
    return t


def make_split(gid, protocol, fold, seed, n_val_groups=3):
    """Return (train_idx, val_idx, test_idx). Val is always group-disjoint from train."""
    rng = np.random.RandomState(seed)
    groups = np.unique(gid)

    if protocol == "random":
        n = len(gid)
        perm = rng.permutation(n)
        n_te = int(0.2 * n)
        test = perm[:n_te]
        rest = perm[n_te:]
        n_va = int(0.1 * len(rest))
        return rest[n_va:], rest[:n_va], test

    if protocol == "loco":
        test_g = groups[fold]
        train_g = np.setdiff1d(groups, [test_g])
        val_g = rng.choice(train_g, size=n_val_groups, replace=False)
        fit_g = np.setdiff1d(train_g, val_g)
        return (np.where(np.isin(gid, fit_g))[0],
                np.where(np.isin(gid, val_g))[0],
                np.where(gid == test_g)[0])

    raise ValueError(protocol)


# ----------------------------------------------------------------- loss -----
def physics_loss(mu, delta, y, P, model,
                 w_elastic=0.12, w_mono=0.08, w_force_pos=0.02,
                 w_gate1=0.08, w_gate2=0.12, w_order=0.05, w_res_l2=1e-5):
    """Data + physics penalties, matching the submitted formulation."""
    L_data = torch.mean((mu - y) ** 2)
    total = L_data
    comps = {"data": float(L_data)}

    dmax = delta.max().detach()

    if "k" in P:
        mask_el = (delta < 0.12 * dmax).float()
        L_el = torch.mean(mask_el * (mu - P["k"] * torch.clamp(delta, min=M.EPS) ** 1.5) ** 2)
        total = total + w_elastic * L_el
        comps["elastic"] = float(L_el)

    dmu = torch.autograd.grad(mu.sum(), delta, create_graph=True, retain_graph=True)[0]
    L_mono = torch.mean(torch.relu(-dmu))
    L_pos = torch.mean(torch.relu(-mu))
    total = total + w_mono * L_mono + w_force_pos * L_pos
    comps.update(mono=float(L_mono), pos=float(L_pos))

    if "pi1" in P:
        L_g1 = torch.mean((delta < 0.06 * dmax).float() * P["pi1"] ** 2)
        L_g2 = torch.mean((delta < 0.10 * dmax).float() * P["pi2"] ** 2)
        L_ord = torch.mean(torch.relu(0.07 * dmax - (P["dh"] - P["dy"])))
        total = total + w_gate1 * L_g1 + w_gate2 * L_g2 + w_order * L_ord
        comps.update(gate1=float(L_g1), gate2=float(L_g2), order=float(L_ord))

    if hasattr(model, "res_energy"):
        l2 = sum(torch.sum(p ** 2) for p in model.res_energy.parameters())
        total = total + w_res_l2 * l2

    return total, comps


# -------------------------------------------------------------- training ----
def fit(model, tr, va, epochs=1200, lr=2e-3, use_physics=True, patience=250, verbose=False):
    """Full-batch Adam with early stopping on a GROUP-DISJOINT validation set."""
    d_tr, c_tr, y_tr = tr
    d_va, c_va, y_va = va
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=40)

    best = {"val": np.inf, "state": None, "ep": 0}
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        mu, P = model(d_tr, c_tr)
        if use_physics:
            L, _ = physics_loss(mu, d_tr, y_tr, P, model)
        else:
            L = torch.mean((mu - y_tr) ** 2)
        L.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        model.eval()
        mu_v, _ = model(d_va, c_va, create_graph=False)
        val = float(torch.mean((mu_v - y_va) ** 2))
        sched.step(val)

        if val < best["val"] - 1e-9:
            best = {"val": val,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
                    "ep": ep}
        if ep - best["ep"] > patience:
            break
        if verbose and ep % 200 == 0:
            print(f"    ep {ep:4d} train {float(L):10.3f} val {val:10.3f}")

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, best


# --------------------------------------------------------------- metrics ----
def regression_metrics(y, p):
    return {"R2": float(r2_score(y, p)),
            "MAE": float(mean_absolute_error(y, p)),
            "RMSE": float(np.sqrt(mean_squared_error(y, p)))}


def calibration_metrics(y, mu, sigma):
    """Reviewer 6 Q12: coverage at 1/2 sigma, ECE over Gaussian quantiles, sharpness."""
    z = np.abs(y - mu) / np.maximum(sigma, 1e-9)
    out = {"PICP_1sigma": float(np.mean(z <= 1.0)),
           "PICP_2sigma": float(np.mean(z <= 2.0)),
           "sharpness_mean_sigma": float(np.mean(sigma))}

    from scipy.stats import norm
    levels = np.linspace(0.05, 0.95, 19)
    emp = np.array([np.mean(z <= norm.ppf(0.5 + l / 2)) for l in levels])
    out["ECE"] = float(np.mean(np.abs(emp - levels)))
    out["MCE"] = float(np.max(np.abs(emp - levels)))
    # Negative log-likelihood (lower is better)
    out["NLL"] = float(np.mean(0.5 * np.log(2 * np.pi * sigma ** 2)
                               + 0.5 * ((y - mu) / np.maximum(sigma, 1e-9)) ** 2))
    return out


# ------------------------------------------------------------ single run ----
def run_once(df_load, model_fn, protocol, fold, seed,
             use_physics=True, epochs=1200, with_sigma=False):
    torch.manual_seed(seed)
    np.random.seed(seed)

    delta, ctx, y, gid = build_arrays(df_load)
    itr, iva, ite = make_split(gid, protocol, fold, seed)

    sc = StandardScaler().fit(ctx[itr])          # fitted on TRAIN only
    ctx_s = sc.transform(ctx).astype(np.float32)

    pack = lambda idx: (to_t(delta[idx], grad=True), to_t(ctx_s[idx]), to_t(y[idx]))
    tr, va, te = pack(itr), pack(iva), pack(ite)

    model = model_fn().to(DEVICE)
    model, best = fit(model, tr, va, epochs=epochs, use_physics=use_physics)

    model.eval()
    mu_te, P_te = model(te[0], te[1], create_graph=False)
    p = mu_te.detach().cpu().numpy().ravel()
    t = y[ite].ravel()
    res = regression_metrics(t, p)
    res.update(protocol=protocol, fold=int(fold), seed=int(seed),
               n_test=int(len(ite)), best_epoch=best["ep"])

    if with_sigma:
        for q in model.parameters():
            q.requires_grad = False
        sig = M.SigmaHead().to(DEVICE)
        optS = optim.Adam(sig.parameters(), lr=2e-3)
        mu_tr, _ = model(tr[0], tr[1], create_graph=False)
        r_tr = (tr[2] - mu_tr).detach()
        mu_va, _ = model(va[0], va[1], create_graph=False)
        r_va = (va[2] - mu_va).detach()
        bestS = {"val": np.inf, "state": None, "ep": 0}
        for ep in range(1, 601):
            optS.train() if hasattr(optS, "train") else None
            sig.train(); optS.zero_grad()
            s = sig(tr[0], tr[1])
            nll = torch.mean(0.5 * torch.log(2 * np.pi * s ** 2) + 0.5 * (r_tr / s) ** 2)
            (nll + 5e-4 * torch.mean(s)).backward()
            optS.step()
            sig.eval()
            with torch.no_grad():
                sv = sig(va[0], va[1])
                v = float(torch.mean(0.5 * torch.log(2 * np.pi * sv ** 2) + 0.5 * (r_va / sv) ** 2))
            if v < bestS["val"]:
                bestS = {"val": v, "state": {k: q.detach().clone() for k, q in sig.state_dict().items()}, "ep": ep}
            if ep - bestS["ep"] > 120:
                break
        sig.load_state_dict(bestS["state"])
        sig.eval()
        with torch.no_grad():
            s_te = sig(te[0], te[1]).cpu().numpy().ravel()
        res.update(calibration_metrics(t, p, s_te))

    return res, model, (t, p, ite)


def aggregate(rows, keys=("R2", "MAE", "RMSE")):
    """Mean +/- std across seeds/folds."""
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows if k in r], dtype=float)
        if len(v):
            out[k + "_mean"] = float(v.mean())
            out[k + "_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    out["n_runs"] = len(rows)
    return out


def save(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(f"  wrote {path}")


if __name__ == "__main__":
    df = D.keep_loading(D.load())
    t0 = time.time()
    r, _, _ = run_once(df, M.EnergyPhasePINN, "loco", 0, 0, epochs=400, with_sigma=True)
    print(f"timing check ({time.time()-t0:.1f}s):")
    print(json.dumps(r, indent=2))
