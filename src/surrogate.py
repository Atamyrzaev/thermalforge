"""
surrogate.py
ThermalForge :: a screening surrogate for placement, with its error bars.

WHAT THIS IS
A linear least-squares fit over five physically-motivated features per
chiplet, used to RANK candidate placements cheaply. It is not a neural
network, it is not "AI acceleration", and it is not a solver: it never
produces a reported temperature.

WHAT IT IS FOR
Ranking. The exact search (src/exhaustive.py) evaluates every buildable
lattice layout by superposition, which costs one linear solve per
(chiplet, position) pair up front. This surrogate needs no linear solve at
all to score a layout, so it can rank a candidate set first; the top of that
ranking is then solved for real, and src/experiment.py re-solves the winner
and refuses to report it unless the real solver agrees.

So a bad surrogate costs ACCURACY, never correctness: the worst it can do is
rank a good layout below the cut, in which case the pipeline reports a real
temperature for a worse placement. That property is pinned by a test that
inverts the ranking on purpose.

THE FEATURES (per chiplet i, all in kelvin of rise)
    floor         P_total / (h A_package)   -- the part placement cannot change
    self_1d       P_i / (h A_i)             -- local rise with no spreading
    self_spread   P_i / (h A_i_eff)         -- spread over lambda, where
                                               A_i_eff = (w + 2L)(d + 2L)
    neighbours    sum_j P_j/(h A_j_eff) exp(-gap_ij / lambda)
    edges         P_i/(h A_i_eff) sum_walls exp(-2 s_iw / lambda)

lambda = sqrt(sum(k_xy t) / h) is the lateral spreading length already
printed in the run banner. The neighbour and edge terms are the two ways a
placement changes a peak in this model: heat from a neighbour arriving
within a spreading length, and the mirror image an adiabatic wall creates.
The fit decides how much each matters; the features decide what can matter.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.experiment import (
    ChipletDef, Experiment, Package, Placement, STACK_MATERIALS, solve_placement,
)
from src.grid_builder import MATERIALS

FEATURE_NAMES: Tuple[str, ...] = ("floor", "self_1d", "self_spread", "neighbours", "edges")

# Enumerating more layouts than this costs more memory than the screening
# saves, so the search samples instead and says so in its result.
MAX_ENUMERATED_LAYOUTS = 4_000_000


class SurrogateError(Exception):
    """Raised when a surrogate cannot be calibrated or applied."""


# ---------------------------------------------------------------------------
# Geometry the model needs
# ---------------------------------------------------------------------------

def spreading_length_m(package: Package) -> float:
    """lambda = sqrt(sum(k_xy * t) / h): how far heat spreads laterally before
    the sink takes it out. The same quantity the run banner prints."""
    sheet_w_per_k = 0.0
    for material, thickness_um in zip(STACK_MATERIALS, package.stack_um):
        props = MATERIALS[material]
        kx = float(props.get("kx", props.get("k")))
        ky = float(props.get("ky", props.get("k")))
        sheet_w_per_k += math.sqrt(kx * ky) * thickness_um * 1e-6
    if package.h_convective <= 0:
        raise SurrogateError("a spreading length needs a positive h_convective")
    return math.sqrt(sheet_w_per_k / package.h_convective)


def layout_features(package: Package, catalog: Sequence[ChipletDef], lam: float,
                    xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Feature tensor of shape (candidates, chiplets, features).

    `xs` and `ys` are lower-left origins in metres, shape (candidates,
    chiplets). Vectorised over candidates: the whole point of a screening
    model is that scoring a million layouts is array arithmetic.
    """
    xs = np.atleast_2d(np.asarray(xs, dtype=np.float64))
    ys = np.atleast_2d(np.asarray(ys, dtype=np.float64))
    if xs.shape != ys.shape or xs.shape[1] != len(catalog):
        raise SurrogateError(f"origins {xs.shape} do not match a catalog of {len(catalog)}")

    h = package.h_convective
    widths = np.array([c.width_m for c in catalog])
    heights = np.array([c.height_m for c in catalog])
    powers = np.array([c.power_w for c in catalog])

    area = widths * heights
    area_eff = (widths + 2.0 * lam) * (heights + 2.0 * lam)
    self_1d = powers / (h * area)                       # (n,)
    self_spread = powers / (h * area_eff)               # (n,)
    floor = float(np.sum(powers)) / (h * package.sink_area_m2)

    x1, y1 = xs + widths, ys + heights                  # (C, n)

    # Edge-to-edge gaps between every pair, per candidate.
    # gap[c, i, j]: how far block i is from block j along each axis, zero when
    # they overlap on that axis. np.maximum broadcasts; np.maximum.reduce over
    # a list would try to stack the differently-shaped operands first.
    gap_x = np.maximum(np.maximum(xs[:, :, None] - x1[:, None, :],
                                  xs[:, None, :] - x1[:, :, None]), 0.0)
    gap_y = np.maximum(np.maximum(ys[:, :, None] - y1[:, None, :],
                                  ys[:, None, :] - y1[:, :, None]), 0.0)
    gap = np.hypot(gap_x, gap_y)                        # (C, n, n)
    coupling = np.exp(-gap / lam) * self_spread[None, None, :]
    idx = np.arange(len(catalog))
    coupling[:, idx, idx] = 0.0                         # a chiplet is not its own neighbour
    neighbours = coupling.sum(axis=2)                   # (C, n)

    # Adiabatic side walls mirror a block back onto itself: the image sits
    # twice its distance from the wall away.
    wall = (np.exp(-2.0 * np.maximum(xs, 0.0) / lam)
            + np.exp(-2.0 * np.maximum(package.lx_m - x1, 0.0) / lam)
            + np.exp(-2.0 * np.maximum(ys, 0.0) / lam)
            + np.exp(-2.0 * np.maximum(package.ly_m - y1, 0.0) / lam))
    edges = wall * self_spread[None, :]

    candidates = xs.shape[0]
    features = np.empty((candidates, len(catalog), len(FEATURE_NAMES)))
    features[:, :, 0] = floor
    features[:, :, 1] = self_1d[None, :]
    features[:, :, 2] = self_spread[None, :]
    features[:, :, 3] = neighbours
    features[:, :, 4] = edges
    return features


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SurrogateModel:
    """Coefficients, and what they were fitted on. Carried with the numbers so
    a prediction can never be read without its provenance."""
    coefficients: Tuple[float, ...]
    lambda_m: float
    training_layouts: int
    training_mae_k: float
    feature_names: Tuple[str, ...] = FEATURE_NAMES

    def predict_chiplet_rise(self, features: np.ndarray) -> np.ndarray:
        return features @ np.asarray(self.coefficients)

    def predict_peak_rise(self, features: np.ndarray) -> np.ndarray:
        """Predicted peak rise per candidate: the hottest chiplet's rise."""
        return self.predict_chiplet_rise(features).max(axis=1)


@dataclass(frozen=True)
class SurrogateAccuracy:
    """Held-out error, measured, never assumed. The predictions and truths are
    carried so a report can plot them rather than assert them."""
    layouts: int
    mae_k: float
    max_error_k: float
    rank_correlation: float
    top1_is_truly_best: bool
    top10_recall: float
    predicted_rise_k: Tuple[float, ...] = ()
    true_rise_k: Tuple[float, ...] = ()


def fit_surrogate(package: Package, catalog: Sequence[ChipletDef],
                  training: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                  lam: Optional[float] = None) -> SurrogateModel:
    """Least squares over the per-chiplet rises of solved layouts.

    `training` is a sequence of (xs, ys, per_chiplet_rise_k) for one layout
    each. Fitting per CHIPLET rather than per layout gives n equations per
    solve instead of one, so a few dozen real solves calibrate five
    coefficients comfortably.
    """
    if not training:
        raise SurrogateError("no training layouts: a surrogate must be calibrated on "
                             "real solves before it can rank anything")
    lam = spreading_length_m(package) if lam is None else lam
    rows, targets = [], []
    for xs, ys, rise in training:
        features = layout_features(package, catalog, lam, xs, ys)[0]
        rows.append(features)
        targets.append(np.asarray(rise, dtype=np.float64))
    design = np.concatenate(rows, axis=0)
    target = np.concatenate(targets, axis=0)
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = design @ coefficients - target
    return SurrogateModel(coefficients=tuple(float(v) for v in coefficients),
                          lambda_m=lam, training_layouts=len(training),
                          training_mae_k=float(np.mean(np.abs(residual))))


def measure_accuracy(model: SurrogateModel, package: Package,
                     catalog: Sequence[ChipletDef],
                     held_out: Sequence[Tuple[np.ndarray, np.ndarray, float]]
                     ) -> SurrogateAccuracy:
    """Error and ranking quality on layouts the fit never saw."""
    if len(held_out) < 3:
        raise SurrogateError("need at least three held-out layouts to measure accuracy")
    xs = np.array([sample[0] for sample in held_out])
    ys = np.array([sample[1] for sample in held_out])
    truth = np.array([sample[2] for sample in held_out], dtype=np.float64)
    predicted = model.predict_peak_rise(layout_features(package, catalog,
                                                        model.lambda_m, xs, ys))
    error = predicted - truth

    order_pred = np.argsort(predicted)
    order_true = np.argsort(truth)
    rank_pred = np.empty_like(order_pred)
    rank_true = np.empty_like(order_true)
    rank_pred[order_pred] = np.arange(len(truth))
    rank_true[order_true] = np.arange(len(truth))
    correlation = float(np.corrcoef(rank_pred, rank_true)[0, 1])

    best_true = set(np.argsort(truth)[:10].tolist())
    best_pred = set(np.argsort(predicted)[:10].tolist())
    return SurrogateAccuracy(
        layouts=len(truth), mae_k=float(np.mean(np.abs(error))),
        max_error_k=float(np.max(np.abs(error))), rank_correlation=correlation,
        top1_is_truly_best=bool(order_pred[0] == order_true[0]),
        top10_recall=len(best_true & best_pred) / min(10, len(truth)),
        predicted_rise_k=tuple(float(v) for v in predicted),
        true_rise_k=tuple(float(v) for v in truth),
    )


# ---------------------------------------------------------------------------
# Candidate layouts
# ---------------------------------------------------------------------------

def lattice_positions(package: Package, catalog: Sequence[ChipletDef],
                      pitch_m: float) -> List[List[Tuple[int, int]]]:
    """Every in-bounds lattice origin, in cells, for each chiplet."""
    pitch_x = max(1, int(round(pitch_m / package.dx)))
    pitch_y = max(1, int(round(pitch_m / package.dy)))
    positions: List[List[Tuple[int, int]]] = []
    for chiplet in catalog:
        nvx = int(round(chiplet.width_m / package.dx))
        nvy = int(round(chiplet.height_m / package.dy))
        xs = list(range(0, package.nx - nvx + 1, pitch_x))
        ys = list(range(0, package.ny - nvy + 1, pitch_y))
        if not xs or not ys:
            raise SurrogateError(
                f"'{chiplet.id}' has no lattice position on this die at a "
                f"{pitch_m * 1e6:.0f} um pitch")
        positions.append([(x, y) for x in xs for y in ys])
    return positions


def enumerate_buildable(package: Package, catalog: Sequence[ChipletDef],
                        pitch_m: float, limit: int = MAX_ENUMERATED_LAYOUTS
                        ) -> Tuple[np.ndarray, np.ndarray, int, bool]:
    """Every non-overlapping lattice layout, as (xs, ys) cell arrays.

    Depth-first with the largest block first and overlap pruning, which is
    the same enumeration the exact search performs -- a test pins the counts
    against each other. Returns (xs, ys, layouts_in_full_product, complete).
    """
    positions = lattice_positions(package, catalog, pitch_m)
    extents = [(int(round(c.width_m / package.dx)), int(round(c.height_m / package.dy)))
               for c in catalog]
    order = sorted(range(len(catalog)), key=lambda i: extents[i][0] * extents[i][1],
                   reverse=True)
    total = 1
    for block in positions:
        total *= len(block)

    found_x: List[Tuple[int, ...]] = []
    found_y: List[Tuple[int, ...]] = []
    assignment: Dict[int, Tuple[int, int]] = {}
    complete = True

    def overlaps(idx: int, pos: Tuple[int, int]) -> bool:
        ax0, ay0 = pos
        ax1, ay1 = ax0 + extents[idx][0], ay0 + extents[idx][1]
        for other, (bx0, by0) in assignment.items():
            bx1, by1 = bx0 + extents[other][0], by0 + extents[other][1]
            if min(ax1, bx1) > max(ax0, bx0) and min(ay1, by1) > max(ay0, by0):
                return True
        return False

    def descend(depth: int) -> None:
        nonlocal complete
        if not complete:
            return
        if depth == len(catalog):
            found_x.append(tuple(assignment[i][0] for i in range(len(catalog))))
            found_y.append(tuple(assignment[i][1] for i in range(len(catalog))))
            if len(found_x) >= limit:
                complete = False
            return
        idx = order[depth]
        for pos in positions[idx]:
            if overlaps(idx, pos):
                continue
            assignment[idx] = pos
            descend(depth + 1)
            del assignment[idx]
            if not complete:
                return

    descend(0)
    if not found_x:
        raise SurrogateError("no buildable layout exists on this lattice")
    return (np.array(found_x, dtype=np.int32), np.array(found_y, dtype=np.int32),
            total, complete)


# ---------------------------------------------------------------------------
# The screening search
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SurrogateSearchResult:
    """What the screening did, and what it cost. Every temperature here came
    from the real solver; the surrogate only chose which layouts to solve."""
    best_layout: Tuple[Tuple[int, int], ...]
    best_peak_c: float
    model: SurrogateModel
    accuracy: SurrogateAccuracy
    candidates_screened: int
    candidates_in_full_space: int
    complete_enumeration: bool
    top_k_solved: int
    calibration_layouts: int
    calibration_s: float
    enumeration_s: float
    screening_s: float
    solve_s: float

    @property
    def total_s(self) -> float:
        return self.calibration_s + self.enumeration_s + self.screening_s + self.solve_s

    @property
    def screening_us_per_candidate(self) -> float:
        return 1e6 * self.screening_s / max(1, self.candidates_screened)


class SurrogateScreeningSearch:
    """Calibrate on a few real solves, rank every candidate, solve the top of
    the ranking for real, and hand the winner back to be re-solved."""

    def __init__(self, experiment: Experiment, top_k: int = 25,
                 calibration_layouts: int = 40, seed: int = 0):
        if experiment.initial is None:
            raise SurrogateError("solve_initial() must run before screening: the "
                                 "surrogate is calibrated with the real solver")
        if top_k < 1:
            raise SurrogateError(f"top_k must be >= 1, got {top_k}")
        self.exp = experiment
        self.top_k = int(top_k)
        self.calibration_layouts = int(calibration_layouts)
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    # -- helpers ---------------------------------------------------------
    def _placement(self, xs_cells, ys_cells) -> Placement:
        package = self.exp.package
        return Placement(
            pitch_m=self.exp.initial_placement.pitch_m,
            origins_m=tuple((c.id, float(x) * package.dx, float(y) * package.dy)
                            for c, x, y in zip(self.exp.catalog, xs_cells, ys_cells)))

    def _solve(self, xs_cells, ys_cells):
        """One real solve. Returns (per-chiplet rise, peak degC)."""
        result = solve_placement(self.exp.solver, self.exp.package, self.exp.catalog,
                                 self._placement(xs_cells, ys_cells), "surrogate sample")
        rise = np.array([row.max_c - self.exp.package.t_ambient_c
                         for row in result.per_chiplet])
        return rise, result.peak_c

    def _sample_valid(self, xs_all: np.ndarray, ys_all: np.ndarray, count: int):
        picks = self.rng.choice(xs_all.shape[0], size=min(count, xs_all.shape[0]),
                                replace=False)
        return xs_all[picks], ys_all[picks]

    # -- the search ------------------------------------------------------
    def run(self, verbose: bool = True) -> SurrogateSearchResult:
        package, catalog = self.exp.package, self.exp.catalog
        pitch_m = self.exp.initial_placement.pitch_m

        t0 = time.perf_counter()
        xs_all, ys_all, total_space, complete = enumerate_buildable(
            package, catalog, pitch_m)
        enumeration_s = time.perf_counter() - t0
        if verbose:
            print(f"[surrogate] enumerated {xs_all.shape[0]:,} buildable layouts of "
                  f"{total_space:,} in {enumeration_s:.2f}s"
                  + ("" if complete else " (capped)"))

        # -- calibrate on real solves ----------------------------------
        wanted = min(self.calibration_layouts + 20, xs_all.shape[0])
        cal_x, cal_y = self._sample_valid(xs_all, ys_all, wanted)
        split = min(self.calibration_layouts, max(1, len(cal_x) - 3))
        t0 = time.perf_counter()
        training, held_out = [], []
        for n, (xs, ys) in enumerate(zip(cal_x, cal_y)):
            rise, peak_c = self._solve(xs, ys)
            if n < split:
                training.append((xs * package.dx, ys * package.dy, rise))
            else:
                held_out.append((xs * package.dx, ys * package.dy,
                                 peak_c - package.t_ambient_c))
        calibration_s = time.perf_counter() - t0

        model = fit_surrogate(package, catalog, training)
        accuracy = measure_accuracy(model, package, catalog, held_out)
        if verbose:
            print(f"[surrogate] fitted on {model.training_layouts} real solves in "
                  f"{calibration_s:.2f}s; held-out MAE {accuracy.mae_k:.2f} K, "
                  f"rank correlation {accuracy.rank_correlation:.3f}")

        # -- rank everything -------------------------------------------
        t0 = time.perf_counter()
        predicted = model.predict_peak_rise(
            layout_features(package, catalog, model.lambda_m,
                            xs_all * package.dx, ys_all * package.dy))
        ranking = np.argsort(predicted)[:self.top_k]
        screening_s = time.perf_counter() - t0
        if verbose:
            print(f"[surrogate] screened {xs_all.shape[0]:,} layouts in "
                  f"{screening_s:.2f}s "
                  f"({1e6 * screening_s / max(1, xs_all.shape[0]):.2f} us each)")

        # -- the real solver decides among the shortlist ---------------
        t0 = time.perf_counter()
        # The initial placement is always a candidate. Screening can fail to
        # shortlist anything cooler, and the product must never report an
        # "optimized" placement hotter than the one it started from. Its peak
        # is already a real, verified solve, so it costs nothing here.
        best_peak = float(self.exp.initial.peak_c)
        best_layout = self.exp.layout_of(self.exp.initial_placement)
        for index in ranking:
            _rise, peak_c = self._solve(xs_all[index], ys_all[index])
            if peak_c < best_peak:
                best_peak = peak_c
                best_layout = tuple((int(x), int(y))
                                    for x, y in zip(xs_all[index], ys_all[index]))
        solve_s = time.perf_counter() - t0
        if verbose:
            print(f"[surrogate] solved the top {len(ranking)} for real in {solve_s:.2f}s; "
                  f"best {best_peak:.4f} degC")

        return SurrogateSearchResult(
            best_layout=best_layout, best_peak_c=float(best_peak), model=model,
            accuracy=accuracy, candidates_screened=int(xs_all.shape[0]),
            candidates_in_full_space=int(total_space), complete_enumeration=complete,
            top_k_solved=int(len(ranking)), calibration_layouts=model.training_layouts,
            calibration_s=calibration_s, enumeration_s=enumeration_s,
            screening_s=screening_s, solve_s=solve_s)
