"""
EnergyPhasePINN and PhasePINN model definitions.

EnergyPhasePINN is the proposed model: a single differentiable free-energy potential
U(delta; g, T, v), gated between elastic / plastic / hardening regimes, from which the
predicted indentation force is obtained as F = dU/ddelta by automatic differentiation
(so force can never contradict the model's own energy landscape).

PhasePINN is EnergyPhasePINN's direct-force ablation: the same three-regime gating
structure, but force is predicted directly by each branch rather than derived from a
shared energy potential. It is reported alongside EnergyPhasePINN to show that the two
give statistically indistinguishable raw accuracy (see the root README and the
notebooks/ comparison) -- EnergyPhasePINN's contribution is physical interpretability
and force-energy consistency, not higher accuracy.

The GPR baseline is implemented separately in gpr_baseline.py (scikit-learn); no MLP or
tree-ensemble baseline is included here -- see the README for why.
"""

"""
Models for the JMSC revision.

Three corrections relative to the submitted code:

1. ELASTIC-GRADIENT BUG. The submitted EnergyPhasePINN built the elastic energy
   from a separate input column (delta^5/2) but read the force as dU/dX[:,0].
   Since Ue had no functional dependence on column 0, dUe/dX[:,0] was identically
   zero and the elastic branch contributed NOTHING to the predicted force.
   Here every energy term is written as a function of a single leaf tensor
   `delta`, so the chain rule is complete by construction.

2. GATE DERIVATIVES (Reviewer 6 Q7). Force is the full autograd derivative of U,
   which already includes the dpi/ddelta terms. `force_decomposition` splits the
   total into the weighted-force part and the gate-derivative part so the latter
   can be quantified.

3. SCALER LEAKAGE. The context scaler is fitted on training conditions only
   (handled in revision_run.py).
"""

import torch
import torch.nn as nn
from torch.autograd import grad

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-6


def _powers(delta, powers):
    """Depth features built from the leaf `delta` so gradients flow through them."""
    # EPS floor: fractional powers of an exact zero produce NaN in the
    # double-backward pass (d/dd of d^0.5 ~ d^-0.5). delta=0 occurs at every
    # curve start, so this floor is required, not cosmetic.
    d = torch.clamp(delta, min=EPS)
    cols = [d]
    if "sqrt" in powers:
        cols.append(torch.sqrt(d))
    if "32" in powers:
        cols.append(d ** 1.5)
    if "52" in powers:
        cols.append(d ** 2.5)
    return torch.cat(cols, dim=1)


class EnergyPhasePINN(nn.Module):
    """Gated free-energy model. Force = dU/ddelta by full autograd."""

    def __init__(self, ctx_dim=3, hidden=128, use_residual=True, res_powers=("32",)):
        super().__init__()
        self.use_residual = use_residual
        self.res_powers = res_powers
        self.softplus = nn.Softplus(beta=10.0)
        self.sigmoid = nn.Sigmoid()

        self.ctx_net = nn.Sequential(
            nn.Linear(ctx_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 12),
        )
        if use_residual:
            n_res = 1 + len(res_powers)
            self.res_energy = nn.Sequential(
                nn.Linear(ctx_dim + n_res, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, 1),
            )

    def params(self, ctx):
        raw = self.ctx_net(ctx)
        (k_r, dy_r, w1_r, Fy_r, a_r, b_r,
         dgap_r, w2_r, c_r, m_r, b1_r, b2_r) = torch.chunk(raw, 12, dim=1)
        sp = self.softplus
        dy = sp(dy_r)
        return {
            "k": sp(k_r), "dy": dy, "w1": sp(w1_r) + 1e-3,
            "Fy": sp(Fy_r), "a": sp(a_r), "b": sp(b_r),
            "dh": dy + sp(dgap_r), "w2": sp(w2_r) + 1e-3,
            "c": sp(c_r), "m": 1.0 + 3.0 * self.sigmoid(m_r),
            "bias1": sp(b1_r), "bias2": sp(b2_r),
        }

    def energy(self, delta, ctx, P, detach_gates=False):
        """Total energy U. Branch energies and gates are all functions of `delta`."""
        Delta = self.softplus(delta - P["dy"]) + EPS
        d = torch.clamp(delta, min=EPS)

        Ue = (2.0 / 5.0) * P["k"] * d ** 2.5
        Up = P["Fy"] * delta + 0.5 * P["a"] * Delta ** 2 + (1.0 / 3.0) * P["b"] * Delta ** 3
        Uh = P["Fy"] * delta + (P["c"] / (P["m"] + 1.0)) * Delta ** (P["m"] + 1.0)

        pi1 = self.sigmoid((delta - P["dy"]) / P["w1"] - P["bias1"])
        pi2 = self.sigmoid((delta - P["dh"]) / P["w2"] - P["bias2"])
        if detach_gates:
            pi1, pi2 = pi1.detach(), pi2.detach()

        U = (1.0 - pi1) * Ue + pi1 * (1.0 - pi2) * Up + pi1 * pi2 * Uh
        if self.use_residual:
            U = U + self.res_energy(torch.cat([_powers(delta, self.res_powers), ctx], dim=1))
        return U, pi1, pi2, (Ue, Up, Uh)

    def forward(self, delta, ctx, create_graph=True):
        P = self.params(ctx)
        U, pi1, pi2, _ = self.energy(delta, ctx, P)
        mu = grad(U.sum(), delta, create_graph=create_graph, retain_graph=True)[0]
        P.update({"U": U, "pi1": pi1, "pi2": pi2})
        return mu, P

    def force_decomposition(self, delta, ctx):
        """Reviewer 6 Q7: split dU/ddelta into weighted-force and gate-derivative parts.

        F_total = [ (1-p1)Ue' + p1(1-p2)Up' + p1 p2 Uh' + Ures' ]   <- weighted forces
                + [ dp1/dd (-Ue + (1-p2)Up + p2 Uh) + p1 dp2/dd (Uh - Up) ]  <- gate terms

        The weighted part is obtained by detaching the gates before differentiating;
        the gate part is the exact remainder.
        """
        P = self.params(ctx)
        U_full, pi1, pi2, _ = self.energy(delta, ctx, P, detach_gates=False)
        F_total = grad(U_full.sum(), delta, create_graph=False, retain_graph=True)[0]

        U_det, _, _, _ = self.energy(delta, ctx, P, detach_gates=True)
        F_weighted = grad(U_det.sum(), delta, create_graph=False, retain_graph=True)[0]

        return (F_total.detach(), F_weighted.detach(), (F_total - F_weighted).detach(),
                pi1.detach(), pi2.detach())


class PhasePINN(nn.Module):
    """Direct-force 3-phase PINN (the submitted PhasePINN): force predicted directly."""

    def __init__(self, ctx_dim=3, hidden=128, powers=("sqrt", "32")):
        super().__init__()
        self.powers = powers
        self.softplus = nn.Softplus(beta=10.0)
        self.sigmoid = nn.Sigmoid()
        self.ctx_net = nn.Sequential(
            nn.Linear(ctx_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 10),
        )
        n_in = 1 + len(powers)
        self.res = nn.Sequential(
            nn.Linear(ctx_dim + n_in, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, delta, ctx, create_graph=True):
        raw = self.ctx_net(ctx)
        k_r, dy_r, w1_r, Fy_r, a_r, dgap_r, w2_r, c_r, m_r, b_r = torch.chunk(raw, 10, dim=1)
        sp = self.softplus
        k, dy, w1 = sp(k_r), sp(dy_r), sp(w1_r) + 1e-3
        Fy, a = sp(Fy_r), sp(a_r)
        dh, w2 = dy + sp(dgap_r), sp(w2_r) + 1e-3
        c, m = sp(c_r), 1.0 + 3.0 * self.sigmoid(m_r)

        d = torch.clamp(delta, min=EPS)
        Delta = sp(delta - dy) + EPS
        Fe = k * d ** 1.5
        Fp = Fy + a * Delta
        Fh = Fy + c * Delta ** m

        pi1 = self.sigmoid((delta - dy) / w1 - sp(b_r))
        pi2 = self.sigmoid((delta - dh) / w2)

        mu = (1 - pi1) * Fe + pi1 * (1 - pi2) * Fp + pi1 * pi2 * Fh
        mu = mu + self.res(torch.cat([_powers(delta, self.powers), ctx], dim=1))
        return mu, {"k": k, "dy": dy, "dh": dh, "m": m, "pi1": pi1, "pi2": pi2}



class SigmaHead(nn.Module):
    """Heteroscedastic sigma on the residuals of a frozen mean model."""

    def __init__(self, ctx_dim=3, hidden=96, powers=("32",)):
        super().__init__()
        self.powers = powers
        self.softplus = nn.Softplus(beta=10.0)
        self.net = nn.Sequential(
            nn.Linear(ctx_dim + 1 + len(powers), hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, delta, ctx):
        raw = self.net(torch.cat([_powers(delta, self.powers), ctx], dim=1))
        return self.softplus(raw) + 1e-4
