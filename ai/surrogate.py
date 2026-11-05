"""
ai/surrogate.py
ThermalForge :: the learned surrogate -- physics first, then a learned correction.

    prediction = ROM (ai/rom.py)  x  exp(learned correction per part)

The reduced-order model carries what changes from package to package -- how
far each cooling stack spreads heat and how well it sinks it -- because that
is physics and physics transfers. At a coarse lateral cell it misses what is
finer than the cell: the hotspot inside a patterned part. The correction is
trained to put that back, from features that describe it (how concentrated
the part's power is, how small its smallest block is against the cell, how
the part compares with the spreading length). It is a small ensemble of
two-layer networks in numpy -- no new dependencies, parameters stored as
plain JSON.

Uncertainty is split-conformal: the 95th percentile of |error| / predicted
rise on CALIBRATION packages the model never trained on. Under
exchangeability that interval covers the truth on 95 % of new designs from
the same distribution, and the test split measures whether it did.

It never produces a reported number. It is for instant what-if (where would
this part run cooler?), for screening in inverse design, and for guiding the
exact search -- and every figure a user sees still comes from the solver.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ai import rom as R

PATTERNS = ("uniform", "mac_array", "two_engines", "quad_cores", "imported")
COOLINGS = ("direct", "lid", "lid_heatsink", "coldplate")
FEATURES = (["log_rom_rise", "self_frac", "log_density", "concentration", "log_block_over_cell",
             "log_w_over_cell", "log_h_over_cell", "log_lambda_mm", "log_side_over_lambda",
             "log_h", "log_die_um", "log_bump_um", "log_interposer_um", "tsv", "beol_um",
             "edge_over_lambda", "n_parts", "log_cell_mm"]
            + [f"pattern_{p}" for p in PATTERNS] + [f"cooling_{c}" for c in COOLINGS])


def _concentration(part) -> Tuple[float, float]:
    """(peak density / mean density, smallest block side in mm) of a part."""
    grid = getattr(part, "power_grid", None)
    if grid is not None and grid.cols and grid.rows:
        w = np.asarray(grid.weights, dtype=float)
        mean = float(w.mean()) or 1.0
        return float(w.max() / mean), min(part.w_mm / grid.cols, part.h_mm / grid.rows)
    if not part.blocks:
        return 1.0, min(part.w_mm, part.h_mm)
    conc = max(b.share / max(b.w_frac * b.h_frac, 1e-9) for b in part.blocks)
    uniform = 1.0 - sum(b.share for b in part.blocks)
    conc = max(conc, uniform)          # density of the uniform remainder, per unit footprint
    side = min(min(b.w_frac * part.w_mm, b.h_frac * part.h_mm) for b in part.blocks)
    return float(conc), float(side)


def part_features(spec, chiplets: Sequence, sol: "R.RomSolve") -> np.ndarray:
    lam_mm = R.spreading_length_m(spec) * 1e3
    cell = sol.stack.cell_mm
    rows = []
    for i, c in enumerate(chiplets):
        rise = max(sol.part_rise[i], 1e-9)
        conc, side = _concentration(c)
        edge = min(c.x_mm, c.y_mm, spec.size_mm - c.x_mm - c.w_mm, spec.size_y - c.y_mm - c.h_mm)
        pattern = c.power_pattern if c.power_pattern in PATTERNS else (
            "imported" if getattr(c, "power_grid", None) is not None else "uniform")
        row = [math.log(rise), sol.self_rise[i] / rise,
               math.log(max(c.power_w, 1e-9) / (c.w_mm * c.h_mm)), math.log(conc),
               math.log(max(side, 1e-6) / cell), math.log(c.w_mm / cell), math.log(c.h_mm / cell),
               math.log(lam_mm), math.log(min(c.w_mm, c.h_mm) / lam_mm), math.log(spec.h_w_m2k),
               math.log(spec.stack_um[2]), math.log(spec.stack_um[1]), math.log(spec.stack_um[0]),
               1.0 if spec.interposer_material != "silicon" else 0.0,
               float(getattr(spec, "beol_um", 0.0)), max(edge, 0.0) / lam_mm,
               float(len(chiplets)), math.log(cell)]
        row += [1.0 if pattern == p else 0.0 for p in PATTERNS]
        row += [1.0 if spec.cooling == k else 0.0 for k in COOLINGS]
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# A small MLP, in numpy
# ---------------------------------------------------------------------------

def _init(rng, sizes):
    params = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        params.append(rng.normal(0.0, math.sqrt(1.0 / a), (a, b)))
        params.append(np.zeros(b))
    return params


def _forward(params, x):
    h = x
    acts = [h]
    for n in range(0, len(params) - 2, 2):
        h = np.tanh(h @ params[n] + params[n + 1])
        acts.append(h)
    out = h @ params[-2] + params[-1]
    return out[:, 0], acts


def _train_one(x, y, xv, yv, seed, hidden=48, epochs=400, lr=3e-3, batch=256, l2=1e-5):
    rng = np.random.default_rng(seed)
    params = _init(rng, [x.shape[1], hidden, hidden, 1])
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    best, best_params, stale, t = float("inf"), None, 0, 0
    for epoch in range(epochs):
        order = rng.permutation(len(x))
        for s in range(0, len(x), batch):
            b = order[s:s + batch]
            pred, acts = _forward(params, x[b])
            err = (pred - y[b])[:, None] * (2.0 / len(b))
            grads = [None] * len(params)
            delta = err
            for layer in range(len(params) // 2 - 1, -1, -1):
                a = acts[layer]
                grads[2 * layer] = a.T @ delta + l2 * params[2 * layer]
                grads[2 * layer + 1] = delta.sum(axis=0)
                if layer:
                    delta = (delta @ params[2 * layer].T) * (1.0 - acts[layer] ** 2)
            t += 1
            for n, g in enumerate(grads):
                m[n] = 0.9 * m[n] + 0.1 * g
                v[n] = 0.999 * v[n] + 0.001 * g * g
                mh = m[n] / (1 - 0.9 ** t)
                vh = v[n] / (1 - 0.999 ** t)
                params[n] = params[n] - lr * mh / (np.sqrt(vh) + 1e-8)
        val = float(np.mean((_forward(params, xv)[0] - yv) ** 2))
        if val < best - 1e-7:
            best, best_params, stale = val, [p.copy() for p in params], 0
        else:
            stale += 1
            if stale > 40:
                break
    return best_params, best


@dataclass
class Surrogate:
    mean: np.ndarray
    std: np.ndarray
    members: List[List[np.ndarray]]
    rom_factor: float
    q95_rel: Optional[float] = None       # conformal: |error| / predicted rise, 95 %

    def correction(self, feats: np.ndarray) -> np.ndarray:
        x = (feats - self.mean) / self.std
        return np.mean([_forward(p, x)[0] for p in self.members], axis=0)

    def predict(self, rom_cache: "R.RomCache", spec, chiplets) -> Dict:
        t0 = time.perf_counter()
        sol = R.solve_layout(rom_cache, spec, chiplets)
        feats = part_features(spec, chiplets, sol)
        corr = self.correction(feats)
        rises = np.asarray(sol.part_rise) * np.exp(corr)
        i = int(np.argmax(rises))
        out = {"peak_c": float(spec.ambient_c + rises[i]), "rise_k": float(rises[i]),
               "hottest": chiplets[i].id,
               "chiplet_peaks_c": {c.id: float(spec.ambient_c + r) for c, r in zip(chiplets, rises)},
               "rom_peak_c": float(spec.ambient_c + max(sol.part_rise)),
               "ms": (time.perf_counter() - t0) * 1e3, "label": "surrogate estimate"}
        if self.q95_rel is not None:
            half = self.q95_rel * float(rises[i])
            out["interval_c"] = [out["peak_c"] - half, out["peak_c"] + half]
        return out

    def to_json(self) -> Dict:
        return {"features": list(FEATURES), "mean": self.mean.tolist(), "std": self.std.tolist(),
                "members": [[p.tolist() for p in m] for m in self.members],
                "rom_factor": self.rom_factor, "q95_rel": self.q95_rel}

    @staticmethod
    def from_json(d: Dict) -> "Surrogate":
        if list(d["features"]) != list(FEATURES):
            raise ValueError("this model was trained on a different feature set")
        return Surrogate(mean=np.asarray(d["mean"]), std=np.asarray(d["std"]),
                         members=[[np.asarray(p) for p in m] for m in d["members"]],
                         rom_factor=float(d["rom_factor"]), q95_rel=d.get("q95_rel"))


# ---------------------------------------------------------------------------
# From labels to rows
# ---------------------------------------------------------------------------

def rows_from_labels(labels: Sequence[Dict], rom_cache: "R.RomCache"):
    """Per-part features and targets, plus what is needed to rebuild each
    design's peak: (X, y, design index per row, per-design info)."""
    import studio.engine as E
    xs, ys, owner, designs = [], [], [], []
    for d, lab in enumerate(labels):
        spec = E.PackageSpec.parse(dict(lab["package"], mesh="standard"))
        chips = E.parse_chiplets(lab["chiplets"], spec)
        sol = R.solve_layout(rom_cache, spec, chips)
        feats = part_features(spec, chips, sol)
        amb = lab["package"]["ambient_c"]
        for i, c in enumerate(chips):
            true_rise = lab["chiplet_peaks_c"][c.id] - amb
            xs.append(feats[i])
            ys.append(math.log(max(true_rise, 1e-9) / max(sol.part_rise[i], 1e-9)))
            owner.append(d)
        designs.append({"rom_rise": np.asarray(sol.part_rise), "ambient": amb,
                        "true_peak": lab["peak_c"], "true_rise": lab["peak_c"] - amb,
                        "band_k": lab["band_k"], "package_id": lab["package_id"]})
    return np.asarray(xs), np.asarray(ys), np.asarray(owner), designs


def design_predictions(model: Optional[Surrogate], X: np.ndarray, owner: np.ndarray,
                       designs: List[Dict]) -> np.ndarray:
    """Predicted package peak per design. model=None gives the bare ROM."""
    corr = np.zeros(len(X)) if model is None else model.correction(X)
    peaks = np.empty(len(designs))
    for d, info in enumerate(designs):
        rows = owner == d
        rises = info["rom_rise"] * np.exp(corr[rows])
        peaks[d] = info["ambient"] + rises.max()
    return peaks


def metrics(peaks: np.ndarray, designs: List[Dict]) -> Dict:
    true = np.array([d["true_peak"] for d in designs])
    rise = np.array([d["true_rise"] for d in designs])
    band = np.array([d["band_k"] for d in designs])
    err = peaks - true
    ratio = np.abs(err) / band
    rel = np.abs(err) / rise
    return {"designs": int(len(designs)),
            "abs_err_k": {"p50": float(np.median(np.abs(err))), "p95": float(np.percentile(np.abs(err), 95)),
                          "max": float(np.abs(err).max())},
            "rel_err_pct": {"p50": float(np.median(rel) * 100), "p95": float(np.percentile(rel, 95) * 100)},
            "err_over_band": {"p50": float(np.median(ratio)), "p95": float(np.percentile(ratio, 95)),
                              "max": float(ratio.max())},
            "bias_k": float(err.mean()),
            "inside_band_pct": float((ratio <= 1.0).mean() * 100)}


def fit(train: Sequence[Dict], calib: Sequence[Dict], rom_factor: float = 2.0,
        members: int = 5, seed: int = 0, hidden: int = 48) -> Tuple[Surrogate, Dict]:
    cache = R.RomCache(capacity=8, factor=rom_factor)
    X, y, owner, designs = rows_from_labels(train, cache)
    # hold out 15 % of TRAINING packages for early stopping, by package
    pkgs = np.array(sorted({d["package_id"] for d in designs}))
    rng = np.random.default_rng(seed)
    stop_pkgs = set(rng.choice(pkgs, size=max(1, int(0.15 * len(pkgs))), replace=False).tolist())
    row_pkg = np.array([designs[o]["package_id"] for o in owner])
    va = np.isin(row_pkg, list(stop_pkgs))
    mean, std = X[~va].mean(axis=0), X[~va].std(axis=0)
    std[std < 1e-9] = 1.0
    xs, xv = (X[~va] - mean) / std, (X[va] - mean) / std
    ms = []
    for m in range(members):
        params, _ = _train_one(xs, y[~va], xv, y[va], seed=seed * 100 + m, hidden=hidden)
        ms.append(params)
    model = Surrogate(mean=mean, std=std, members=ms, rom_factor=rom_factor)
    # Split-conformal on calibration PACKAGES -- and clustered by package,
    # because designs on one package are not exchangeable with designs on
    # another: they share its cooling, its stack and its mesh. Each package
    # contributes its own 95th percentile, and the interval is the 95th
    # percentile across packages. Design-level quantiles undercover (measured:
    # 92 % against a nominal 95 %); this is the correction for that.
    Xc, _yc, oc, dc = rows_from_labels(calib, cache)
    pc = design_predictions(model, Xc, oc, dc)
    true = np.array([d["true_peak"] for d in dc])
    pred_rise = pc - np.array([d["ambient"] for d in dc])
    scores = np.abs(pc - true) / np.maximum(pred_rise, 1e-9)
    pkg_of = np.array([d["package_id"] for d in dc])
    per_package = np.array([np.quantile(scores[pkg_of == p], 0.95, method="higher")
                            for p in np.unique(pkg_of)])
    n = len(per_package)
    level = min(1.0, math.ceil((n + 1) * 0.95) / n)
    model.q95_rel = float(np.quantile(per_package, level, method="higher"))
    return model, {"train_rows": int((~va).sum()), "stop_rows": int(va.sum()),
                   "calib_designs": len(scores), "calib_packages": int(n),
                   "q95_rel": model.q95_rel,
                   "q95_rel_by_design": float(np.quantile(scores, min(1.0, math.ceil(
                       (len(scores) + 1) * 0.95) / len(scores)), method="higher"))}


def coverage(model: Surrogate, peaks: np.ndarray, designs: List[Dict]) -> float:
    true = np.array([d["true_peak"] for d in designs])
    rise = peaks - np.array([d["ambient"] for d in designs])
    half = model.q95_rel * rise
    return float(np.mean(np.abs(true - peaks) <= half))


def what_if(rom_cache: "R.RomCache", spec, chiplets: Sequence, index: int,
            positions: Sequence[Tuple[float, float]], model: Optional[Surrogate] = None) -> Dict:
    """Predicted package peak with part `index` at each of `positions`.

    For instant what-if: the other parts' fields do not change, so they are
    solved once and reused, and each position costs one back-substitution on
    the reduced-order model plus the correction. Estimates, never reported
    numbers -- the solver is what the user's temperature comes from.
    """
    from dataclasses import replace as _replace
    t0 = time.perf_counter()
    stack = rom_cache.get(spec)
    others = [R.part_power_map(c, stack.nx, stack.ny, stack.cell_mm)
              for i, c in enumerate(chiplets) if i != index]
    other_self = [stack.device_rise(q) for q in others]
    base = np.sum(other_self, axis=0) if other_self else np.zeros((stack.nx, stack.ny))
    other_masks = [R.footprint_mask(c, stack.nx, stack.ny, stack.cell_mm)
                   for i, c in enumerate(chiplets) if i != index]
    out = []
    for x, y in positions:
        moved = _replace(chiplets[index], x_mm=float(x), y_mm=float(y))
        q = R.part_power_map(moved, stack.nx, stack.ny, stack.cell_mm)
        mine = stack.device_rise(q)
        total = base + mine
        mask = R.footprint_mask(moved, stack.nx, stack.ny, stack.cell_mm)
        order = [moved] + [c for i, c in enumerate(chiplets) if i != index]
        sol = R.RomSolve(stack=stack, device=total,
                         part_rise=[float(total[m].max()) if m.any() else 0.0
                                    for m in [mask] + other_masks],
                         self_rise=[float(f[m].max()) if m.any() else 0.0
                                    for f, m in zip([mine] + other_self, [mask] + other_masks)],
                         part_q=[q] + others)
        rises = np.asarray(sol.part_rise)
        if model is not None:
            rises = rises * np.exp(model.correction(part_features(spec, order, sol)))
        out.append({"x_mm": float(x), "y_mm": float(y),
                    "peak_c": float(spec.ambient_c + rises.max())})
    best = min(out, key=lambda r: r["peak_c"]) if out else None
    return {"positions": out, "best": best, "ms": (time.perf_counter() - t0) * 1e3,
            "label": "surrogate estimate -- not a reported temperature",
            "model": "rom+correction" if model is not None else "rom only"}


def load_registered(name: str = "thermal-surrogate", version: Optional[int] = None) -> Tuple[Surrogate, Dict]:
    from ai import registry
    params, card = registry.load(name, version)
    return Surrogate.from_json(params), card
