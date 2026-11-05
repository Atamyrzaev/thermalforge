"""
ai/calibration.py
ThermalForge :: fit a package family's unknowns to measured parts, with
uncertainty (roadmap S4, step 4.6).

Every real package has numbers nobody published: the cooling coefficient the
rig actually achieved, the interface resistance of the grease that was
actually applied, how far the silicon's conductivity sits from the handbook.
This learns them from measurements -- per package FAMILY, because parts from
one rig share them -- and says how well the data pins each one down.

    parameters   log h, log k_si (a scale on silicon's conductivity) and, when
                 the cooling path has one, log k_tim (a scale on the first
                 TIM -- the interface resistance). Log space: they are
                 positive and their errors are multiplicative.
    forward      the verified engine itself, solved for every measured case
                 at every trial parameter set. Nothing here is a surrogate:
                 the thing being calibrated is the model users run.
    fit          Levenberg-Marquardt on every sensor of every case.
    uncertainty  Laplace: covariance sigma^2 (J^T J)^-1 at the optimum, with
                 sigma^2 from the residual. Reported as a 1-sigma factor per
                 parameter and a correlation matrix.
    identifiable a pair correlated beyond 0.95 is NOT separately determined
                 by this data, and is reported that way instead of as two
                 confident numbers. (A uniform series resistance and h, for
                 example, move the level the same way.)
    check        k-fold by case: fit on the others, predict the held-out
                 cases, report their RMS -- a fit is judged on prediction.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

PARAMS = ("h", "k_si", "k_tim")


@dataclass
class MeasuredCase:
    name: str
    package: dict                     # geometry and cooling; h is the starting value
    chiplets: List[dict]
    sensors: List[Tuple[float, float]]   # (x_mm, y_mm), sampled in the device plane
    measured_c: List[float]


@contextlib.contextmanager
def _silicon_scaled(scale: float):
    from src.grid_builder import MATERIALS
    saved = dict(MATERIALS["silicon"])
    MATERIALS["silicon"]["k"] = saved["k"] * scale
    try:
        yield
    finally:
        MATERIALS["silicon"].clear()
        MATERIALS["silicon"].update(saved)


def _package_at(case: MeasuredCase, theta: Dict[str, float]) -> dict:
    import studio.engine as E
    pkg = dict(case.package)
    pkg["h_w_m2k"] = float(case.package["h_w_m2k"]) * math.exp(theta.get("h", 0.0))
    if "k_tim" in theta:
        layers = [list(l) for l in E.COOLING_STACKS[pkg.get("cooling", "direct")]]
        if not layers:
            raise ValueError("k_tim needs a cooling path with a TIM")
        layers[0][2] = float(layers[0][2]) * math.exp(theta["k_tim"])
        pkg["cooling_layers"] = layers
    return pkg


def forward(case: MeasuredCase, theta: Dict[str, float]) -> np.ndarray:
    """Modelled temperature at each sensor, from a fresh engine model (the
    model cache does not key on material values, so none is shared)."""
    import base64
    import studio.engine as E
    with _silicon_scaled(math.exp(theta.get("k_si", 0.0))):
        spec = E.PackageSpec.parse(_package_at(case, theta))
        out = E.solve(E.ModelCache(capacity=1), spec, E.parse_chiplets(case.chiplets, spec))
    g = out["grid"]
    fieldc = np.frombuffer(base64.b64decode(out["field_b64"]), dtype="<f4").reshape(
        g["nx"], g["ny"], g["nz"]).astype(np.float64)
    k = g["die_k0"]
    cell = spec.cell_mm
    return np.array([fieldc[min(g["nx"] - 1, int(x / cell)), min(g["ny"] - 1, int(y / cell)), k]
                     for x, y in case.sensors])


def _residuals(cases: Sequence[MeasuredCase], theta: Dict[str, float]) -> np.ndarray:
    return np.concatenate([forward(c, theta) - np.asarray(c.measured_c) for c in cases])


@dataclass
class Fit:
    params: Tuple[str, ...]
    theta: Dict[str, float]
    values: Dict[str, float]          # h in W/m2K, k_si and k_tim as factors on the handbook value
    sigma_factor: Dict[str, float]    # 1-sigma multiplicative uncertainty
    correlation: List[List[float]]
    not_identifiable: List[Tuple[str, str, float]]
    rms_k: float
    residual_sigma_k: float
    iterations: int
    evaluations: int
    holdout: Optional[Dict] = None

    def as_dict(self) -> dict:
        return {"params": list(self.params), "values": self.values, "sigma_factor": self.sigma_factor,
                "correlation": self.correlation,
                "not_identifiable": [list(t) for t in self.not_identifiable],
                "rms_k": self.rms_k, "residual_sigma_k": self.residual_sigma_k,
                "iterations": self.iterations, "evaluations": self.evaluations,
                "holdout": self.holdout}


def fit(cases: Sequence[MeasuredCase], params: Sequence[str] = ("h", "k_si"),
        step: float = 0.02, max_iter: int = 12, tol: float = 1e-4) -> Fit:
    params = tuple(params)
    for p in params:
        if p not in PARAMS:
            raise ValueError(f"unknown parameter {p}")
    x = np.zeros(len(params))
    lam = 1e-2
    evals = 0

    def theta_of(v):
        return {p: float(v[i]) for i, p in enumerate(params)}

    r = _residuals(cases, theta_of(x)); evals += 1
    cost = float(r @ r)
    it = 0
    J = None
    for it in range(1, max_iter + 1):
        J = np.empty((r.size, len(params)))
        for i in range(len(params)):
            xp = x.copy(); xp[i] += step
            J[:, i] = (_residuals(cases, theta_of(xp)) - r) / step
            evals += 1
        g = J.T @ r
        H = J.T @ J
        improved = False
        for _ in range(6):
            dx = np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-12), -g)
            rn = _residuals(cases, theta_of(x + dx)); evals += 1
            cn = float(rn @ rn)
            if cn < cost:
                x, r, cost = x + dx, rn, cn
                lam = max(lam / 3, 1e-6)
                improved = True
                break
            lam *= 4
        if not improved or float(np.abs(dx).max()) < tol:
            break
    # Laplace uncertainty at the optimum (Jacobian refreshed there)
    J = np.empty((r.size, len(params)))
    for i in range(len(params)):
        xp = x.copy(); xp[i] += step
        J[:, i] = (_residuals(cases, theta_of(xp)) - r) / step
        evals += 1
    dof = max(1, r.size - len(params))
    s2 = cost / dof
    cov = s2 * np.linalg.pinv(J.T @ J)
    sd = np.sqrt(np.maximum(np.diag(cov), 0.0))
    corr = cov / np.outer(sd, sd) if np.all(sd > 0) else np.eye(len(params))
    flagged = [(params[i], params[j], float(corr[i, j])) for i in range(len(params))
               for j in range(i + 1, len(params)) if abs(corr[i, j]) > 0.95]
    th = theta_of(x)
    base_h = float(cases[0].package["h_w_m2k"])
    values = {"h": base_h * math.exp(th["h"])} if "h" in th else {}
    for p in ("k_si", "k_tim"):
        if p in th:
            values[p] = math.exp(th[p])
    return Fit(params=params, theta=th, values=values,
               sigma_factor={p: math.exp(float(sd[i])) for i, p in enumerate(params)},
               correlation=corr.round(4).tolist(), not_identifiable=flagged,
               rms_k=math.sqrt(cost / r.size), residual_sigma_k=math.sqrt(s2),
               iterations=it, evaluations=evals)


def kfold(cases: Sequence[MeasuredCase], params: Sequence[str] = ("h", "k_si"),
          folds: int = 3, seed: int = 0) -> Dict:
    """Fit on the other folds, predict each held-out case; report its RMS."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(cases))
    held = []
    for f in range(folds):
        test = [cases[i] for i in order[f::folds]]
        train = [cases[i] for i in order if i not in set(order[f::folds].tolist())]
        ft = fit(train, params)
        for c in test:
            err = forward(c, ft.theta) - np.asarray(c.measured_c)
            held.append({"case": c.name, "rms_k": float(np.sqrt(np.mean(err ** 2))),
                         "bias_k": float(err.mean()), "fold": f, "fit": ft.values})
    return {"folds": folds, "cases": held,
            "rms_k": float(np.sqrt(np.mean([h["rms_k"] ** 2 for h in held])))}


def ttc_family(condition: str = "heatsink_with_fan", cooling: str = "lid_heatsink",
               t_si_um: float = 100.0, h0: float = 20000.0) -> List[MeasuredCase]:
    """The SAMOS 2019 test chip as a family: 13 measured patterns, one rig."""
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    data = json.loads((root / "validation" / "measured" / "ttc_samos2019_steady.json")
                      .read_text(encoding="utf-8"))["cooling_conditions"][condition]
    out = []
    for name, case in sorted(data.items()):
        chips = [{"id": f"h{n}", "name": f"H{n}", "w_mm": 2.0, "h_mm": 2.0,
                  "power_w": max(p, 1e-6), "x_mm": (n % 4) * 2.0, "y_mm": (n // 4) * 2.0}
                 for n, p in enumerate(case["power_w"])]
        sensors = [((n % 4) * 2.0 + 1.0, (n // 4) * 2.0 + 1.0) for n in range(16)]
        pkg = {"size_mm": 8.0, "h_w_m2k": h0, "ambient_c": case["ambient_c"], "pitch_mm": 1.0,
               "cooling": cooling, "keepout_um": 0.0, "stack_um": [20.0, 20.0, t_si_um]}
        out.append(MeasuredCase(name, pkg, chips, sensors, list(case["temp_c"])))
    return out
