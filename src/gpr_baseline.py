"""
Gaussian Process Regression baseline, evaluated under the identical leave-one-condition-out
(LOCO) protocol, feature set, and scaling as EnergyPhasePINN / PhasePINN (see train.py),
so the comparison in the root README is apples-to-apples.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

import data as D
import train as R

GPR_BUDGET = 3000  # matches the training-set size GPR used in the original manuscript


def run_gpr(df_load, protocol, fold, seed, budget=GPR_BUDGET, powers=("sqrt", "32")):
    """Train/evaluate a GPR (constant * RBF + white noise kernel) on one LOCO fold.

    Uses the same depth-power feature set as PhasePINN/EnergyPhasePINN's context branch
    (delta, sqrt(delta), delta^1.5, grain size, temperature, velocity) and the same
    train/val/test split machinery as train.py, so results are directly comparable to
    Table 1 / Table 2 in the README.
    """
    delta, ctx, y, gid = R.build_arrays(df_load)
    itr, iva, ite = R.make_split(gid, protocol, fold, seed)
    itr = np.concatenate([itr, iva])  # GPR needs no separate validation set

    rng = np.random.RandomState(seed)
    if budget is not None and len(itr) > budget:
        itr = rng.choice(itr, budget, replace=False)

    def feats(idx):
        d = np.clip(delta[idx], 0, None)
        cols = [d]
        if "sqrt" in powers:
            cols.append(np.sqrt(d))
        if "32" in powers:
            cols.append(d ** 1.5)
        if "52" in powers:
            cols.append(d ** 2.5)
        return np.hstack(cols + [ctx[idx]])

    Xtr, Xte = feats(itr), feats(ite)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    ytr, yte = y[itr].ravel(), y[ite].ravel()

    kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(Xtr.shape[1])) \
        + WhiteKernel(noise_level=1.0)
    model = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                     n_restarts_optimizer=0, random_state=seed)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    res = R.regression_metrics(yte, pred)
    res.update(protocol=protocol, fold=int(fold), seed=int(seed),
               n_train=int(len(itr)), n_test=int(len(ite)))
    return res, model
