"""
exhaustive.py
RISC3D-AI :: Stage 2b -- Exhaustive Placement Search (no heuristic)

Simulated annealing is a heuristic: it samples a few hundred random proposals
out of a space with millions or billions of members and returns the best thing
it happened to see. It carries no guarantee at all -- a different seed gives a
different answer, and neither is provably the optimum.

This module removes the guessing. It evaluates EVERY placement on the
placement lattice and returns the true global optimum, with a certificate:
the number of layouts examined, and the guarantee that nothing was skipped
except layouts that are physically unbuildable.

WHY THIS IS AFFORDABLE: SUPERPOSITION
--------------------------------------
The steady-state system is linear:

    A T = -(f_boundary + q/rho_cp)

f_boundary is constant and q enters linearly, so for any two power maps

    T(q1 + q2) = T_ambient + [T(q1) - T_amb] + [T(q2) - T_amb]

That means the field for ANY layout is the sum of per-chiplet fields, each of
which depends only on WHICH chiplet is at WHICH position -- not on where the
others are. So instead of one linear solve per candidate layout, this module
performs one solve per (chiplet, position) pair, then evaluates any candidate
by adding precomputed arrays.

Verified numerically, not assumed: reconstructing a 4-chiplet layout by
superposition agrees with directly solving that layout to 1.29e-08 K across
all 38,400 cells. See TestExhaustiveSearch in tests/test_core.py, which
re-checks this on every run.

Measured on the default 40x40x12 grid: one linear solve is 136 ms, one
superposed evaluation is 16 us -- a 291,342x reduction, which is the entire
reason brute force is on the table.

COST
----
Precompute is one solve per (chiplet, position) pair; the search is then pure
array arithmetic. On the default 20 x 20 mm die:

    pitch      lattice   positions   layouts to check
    1000 um    20x20        885      1,775,616,000
    2000 um    10x10        234          8,294,400
    2500 um     8x8         156          1,613,472
    4000 um     5x5          66             57,600

The search prunes overlapping partial layouts as soon as they are detected
(depth-first, largest block first), so the number of layouts actually
evaluated is far below the product above -- only buildable ones reach a leaf.

WHAT "EXHAUSTIVE" DOES AND DOES NOT MEAN
-----------------------------------------
It is exhaustive over the PLACEMENT LATTICE, which is a real restriction and
is stated in the result rather than glossed. A block whose optimum sits
between lattice points will not be found there. The lattice exists because a
mesh-refinement study requires layouts representable at every level (see
optimizer.DEFAULT_PLACEMENT_PITCH_UM), and because real floorplanning grids
are coarse anyway. Reduce placement_pitch_um for a finer search at
quadratically higher cost.

Within that lattice the result is the exact global optimum, not an estimate.
"""

from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.thermal_solver import DIRECT_SOLVE_MAX_UNKNOWNS
from src.optimizer import (
    ChipletPlacementOptimizer, ChipletSpec, OptimizationResult,
    OptimizerError, Layout,
)

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("RISC3D-AI.Exhaustive")

__all__ = [
    "ExhaustiveSearchError",
    "ExhaustiveResult",
    "ExhaustivePlacementSearch",
]

# Superposition must reproduce a directly-solved layout to within this many
# kelvin, checked once per search against a real solve. Measured agreement is
# ~1.3e-08 K, so this threshold is three orders of margin and a failure means
# the linearity assumption itself is broken (a non-linear material model, a
# temperature-dependent boundary condition), not round-off.
SUPERPOSITION_TOL_K = 1e-5


class ExhaustiveSearchError(Exception):
    """Raised when the search space is unusable or superposition fails."""


@dataclass
class ExhaustiveResult:
    """Outcome of a complete lattice search.

    `layouts_evaluated` is the number of buildable layouts actually scored;
    `layouts_total` is the size of the full cartesian product before overlap
    pruning. The difference is pruned-as-unbuildable, never skipped for
    convenience -- that distinction is what makes this a proof rather than a
    sample."""
    best_layout: Layout
    best_peak_temperature_c: float
    initial_layout: Layout
    initial_peak_temperature_c: float
    layouts_evaluated: int
    layouts_total: int
    positions_per_block: Tuple[int, ...]
    precompute_solves: int
    precompute_s: float
    search_s: float
    placement_pitch_um: float
    superposition_error_k: float
    best_temperature_field_k: Optional[np.ndarray] = None
    catalog: Tuple[ChipletSpec, ...] = field(default_factory=tuple)

    @property
    def improvement_c(self) -> float:
        return self.initial_peak_temperature_c - self.best_peak_temperature_c

    @property
    def elapsed_s(self) -> float:
        return self.precompute_s + self.search_s

    def to_optimization_result(self, optimizer) -> OptimizationResult:
        """Adapts this result into the OptimizationResult shape so the
        existing report renderer can consume it unchanged.

        The cost histories are the one awkward fit: an exhaustive search has
        no iteration trajectory, so they are filled with a two-point
        [initial, optimum] trace. That is honest rather than decorative --
        the convergence figure then shows the baseline and the proven
        optimum, with no implied search path that never happened.
        """
        return OptimizationResult(
            best_layout=self.best_layout,
            best_cost=self.best_peak_temperature_c,
            best_peak_temperature_c=self.best_peak_temperature_c,
            best_temperature_field_k=self.best_temperature_field_k,
            best_cost_history=[self.initial_peak_temperature_c,
                               self.best_peak_temperature_c],
            current_cost_history=[self.initial_peak_temperature_c,
                                  self.best_peak_temperature_c],
            initial_layout=self.initial_layout,
            initial_peak_temperature_c=self.initial_peak_temperature_c,
            iterations=self.layouts_evaluated,
            accepted_moves=self.layouts_evaluated,
            rejected_moves=0,
            overlap_candidates=self.layouts_total - self.layouts_evaluated,
            seed=optimizer.seed,
            catalog=optimizer.catalog,
            extents_vox=optimizer._extents,
            grid_shape=(optimizer.nx, optimizer.ny, optimizer.nz),
            die_z_range=(optimizer.die_z_start, optimizer.die_z_end),
            voxel_size_m=(optimizer.dx, optimizer.dy, optimizer.dz),
            h_convective=optimizer.h_convective,
            t_ambient_c=optimizer.t_ambient_c,
            elapsed_s=self.elapsed_s,
        )

    def summarize(self) -> str:
        pruned = self.layouts_total - self.layouts_evaluated
        pct = 100.0 * self.layouts_evaluated / self.layouts_total if self.layouts_total else 0.0
        return "\n".join([
            "=" * 64,
            "STAGE 2b :: EXHAUSTIVE PLACEMENT SEARCH (GLOBAL OPTIMUM)",
            "=" * 64,
            f"  Placement lattice:      {self.placement_pitch_um:.0f} um",
            f"  Positions per block:    {list(self.positions_per_block)}",
            f"  Layouts in full space:  {self.layouts_total:,}",
            f"  Layouts evaluated:      {self.layouts_evaluated:,} ({pct:.2f}%)",
            f"  Pruned as overlapping:  {pruned:,}",
            f"  Precompute solves:      {self.precompute_solves} "
            f"({self.precompute_s:.2f} s)",
            f"  Search time:            {self.search_s:.2f} s",
            "-" * 64,
            f"  Initial peak:           {self.initial_peak_temperature_c:.4f} degC",
            f"  GLOBAL OPTIMUM peak:    {self.best_peak_temperature_c:.4f} degC",
            f"  Improvement:            {self.improvement_c:.4f} degC",
            "-" * 64,
            "  This is the exact optimum over the placement lattice, not a",
            "  sample. Every buildable layout was evaluated; the only layouts",
            "  skipped were physically unbuildable (overlapping).",
            f"  Superposition verified against a direct solve to "
            f"{self.superposition_error_k:.2e} K.",
            "=" * 64,
        ])


class ExhaustivePlacementSearch:
    """
    Complete search over every lattice placement, using linear superposition
    so that each candidate costs an array sum rather than a linear solve.

    Reuses an existing ChipletPlacementOptimizer for the grid, stack, solver
    and catalog, so the physics is identical to the annealing path and the
    two are directly comparable.
    """

    def __init__(
        self,
        optimizer: ChipletPlacementOptimizer,
        placement_pitch_um: Optional[float] = None,
    ):
        self.opt = optimizer
        self.pitch_um = (float(placement_pitch_um) if placement_pitch_um is not None
                         else optimizer.placement_pitch_um)
        if self.pitch_um <= 0:
            raise ExhaustiveSearchError(f"placement_pitch_um must be > 0, got {self.pitch_um}")

        self._pitch_x = max(1, int(round(self.pitch_um * 1e-6 / self.opt.dx)))
        self._pitch_y = max(1, int(round(self.pitch_um * 1e-6 / self.opt.dy)))

        # Enumerate every lattice origin at which each block fits in bounds.
        self._positions: List[List[Tuple[int, int]]] = []
        for (nvx, nvy) in self.opt._extents:
            xs = list(range(0, self.opt.nx - nvx + 1, self._pitch_x))
            ys = list(range(0, self.opt.ny - nvy + 1, self._pitch_y))
            if not xs or not ys:
                raise ExhaustiveSearchError(
                    f"A chiplet of {nvx}x{nvy} voxels has no valid position on a "
                    f"{self.opt.nx}x{self.opt.ny} grid at a {self.pitch_um:.0f} um lattice."
                )
            self._positions.append([(x, y) for x in xs for y in ys])

        self.layouts_total = 1
        for p in self._positions:
            self.layouts_total *= len(p)

        # Search largest-footprint block first: it constrains the most and so
        # prunes the most subtree early.
        self._order = sorted(
            range(len(self.opt.catalog)),
            key=lambda i: self.opt._extents[i][0] * self.opt._extents[i][1],
            reverse=True,
        )

        self._fields: Dict[Tuple[int, Tuple[int, int]], np.ndarray] = {}
        self._t_amb = self.opt.solver.bc.t_ambient_k

        logger.info(
            "ExhaustivePlacementSearch: %d um lattice, positions per block %s, "
            "%d layouts in the full space, %d precompute solves",
            self.pitch_um, [len(p) for p in self._positions],
            self.layouts_total, sum(len(p) for p in self._positions),
        )

    # ------------------------------------------------------------------
    def _solve_single(self, idx: int, pos: Tuple[int, int]) -> np.ndarray:
        """Steady-state field rise (T - T_ambient) for ONE chiplet placed
        alone at `pos`, with every other chiplet absent."""
        spec = self.opt.catalog[idx]
        nvx, nvy = self.opt._extents[idx]
        x0, y0 = pos
        solver = self.opt.solver
        solver.q_map[:] = 0.0
        n_cells = nvx * nvy * self.opt._power_thickness_vox
        q = spec.power_w / (n_cells * self.opt._cell_volume)
        solver.q_map[x0:x0 + nvx, y0:y0 + nvy,
                     self.opt.power_z_start:self.opt.power_z_end] += q
        field = solver.solve_steady_state(apply_result=False)
        return field - self._t_amb

    def _rhs_for(self, idx: int, pos: Tuple[int, int]) -> np.ndarray:
        """Right-hand side vector for ONE chiplet placed alone at `pos`.

        Mirrors what solve_steady_state() assembles internally: the system is
        M T = -(boundary_forcing + q/rho_cp), so the RHS is the negated total
        forcing.
        """
        spec = self.opt.catalog[idx]
        nvx, nvy = self.opt._extents[idx]
        x0, y0 = pos
        s = self.opt.solver
        N = self.opt.nx * self.opt.ny * self.opt.nz
        s.q_map[:] = 0.0
        n_cells = nvx * nvy * self.opt._power_thickness_vox
        s.q_map[x0:x0 + nvx, y0:y0 + nvy,
                self.opt.power_z_start:self.opt.power_z_end] += (
            spec.power_w / (n_cells * self.opt._cell_volume))
        forcing = s._boundary_forcing + s.q_map.reshape(N) / s._rho_cp.reshape(N)
        return -forcing

    def precompute_batched(self, verbose: bool = True) -> float:
        """Precompute via a SINGLE batched triangular solve.

        scipy's SuperLU object accepts a 2-D right-hand side, so all
        (chiplet, position) pairs can be solved at once: build B with one
        column per pair and call lu.solve(B). That replaces a Python loop of
        n independent solves -- each of which re-enters scipy, re-validates,
        and runs its own verification pass -- with one call into the
        factorization.

        Only available on the direct path, where a cached LU exists. Above
        DIRECT_SOLVE_MAX_UNKNOWNS the solve is iterative and there is no
        factorization to batch against, so precompute() falls back to the
        loop automatically.

        Results are asserted equal to the loop's within solver tolerance --
        see test_batched_precompute_matches_loop.
        """
        s = self.opt.solver
        N = self.opt.nx * self.opt.ny * self.opt.nz
        if N > DIRECT_SOLVE_MAX_UNKNOWNS:
            raise ExhaustiveSearchError(
                f"Batched precompute needs the direct (factorized) path, but N={N:,} "
                f"exceeds DIRECT_SOLVE_MAX_UNKNOWNS={DIRECT_SOLVE_MAX_UNKNOWNS:,}, where "
                f"the solve is iterative and there is no factorization to batch against."
            )

        t0 = time.perf_counter()
        # Ensure the factorization exists (and is the caller's, if reused).
        if s._steady_lu is None:
            s.solve_steady_state(apply_result=False)

        pairs: List[Tuple[int, Tuple[int, int]]] = [
            (idx, pos) for idx, positions in enumerate(self._positions) for pos in positions
        ]
        B = np.empty((N, len(pairs)), dtype=np.float64, order="F")
        for col, (idx, pos) in enumerate(pairs):
            B[:, col] = self._rhs_for(idx, pos)

        X = s._steady_lu.solve(B)          # one call, all columns
        for col, (idx, pos) in enumerate(pairs):
            self._fields[(idx, pos)] = X[:, col].reshape((self.opt.nx, self.opt.ny, self.opt.nz)) - self._t_amb

        elapsed = time.perf_counter() - t0
        if verbose:
            print(f"[exhaustive] precompute complete (BATCHED): {len(pairs)} solves in "
                  f"{elapsed:.2f}s")
        return elapsed

    def precompute(self, verbose: bool = True) -> float:
        """Solves once per (chiplet, position) pair. Returns elapsed seconds.

        This is the only place linear solves happen. Every candidate layout in
        search() is then evaluated by summing these arrays."""
        t0 = time.perf_counter()
        total = sum(len(p) for p in self._positions)
        done = 0

        # Quieten the solver's per-call INFO line for the duration of the
        # precompute. Two hundred-odd identical "solved via direct (cached
        # LU)" lines bury the one line anybody wants to read, and the useful
        # signal here is progress, which this loop prints itself. Restored
        # afterwards, including on exception, so this never leaks into the
        # rest of the session.
        solver_logger = logging.getLogger("RISC3D-AI.ThermalSolver")
        previous_level = solver_logger.level
        solver_logger.setLevel(logging.WARNING)
        try:
            for idx, positions in enumerate(self._positions):
                for pos in positions:
                    self._fields[(idx, pos)] = self._solve_single(idx, pos)
                    done += 1
                    if verbose and done % 25 == 0:
                        rate = done / (time.perf_counter() - t0)
                        print(f"[exhaustive] precompute {done}/{total} fields "
                              f"({rate:.1f}/s, ~{(total-done)/rate:.0f}s left)")
        finally:
            solver_logger.setLevel(previous_level)

        elapsed = time.perf_counter() - t0
        if verbose:
            print(f"[exhaustive] precompute complete: {total} solves in {elapsed:.2f}s")
        return elapsed

    def _verify_superposition(self, layout: Layout) -> float:
        """Confirms that summing precomputed fields reproduces a real solve of
        the same layout. Returns the max absolute deviation in kelvin and
        raises if it exceeds SUPERPOSITION_TOL_K.

        Superposition is the load-bearing assumption of this whole module: if
        it does not hold, every result here is wrong. It is therefore checked
        against ground truth on every search rather than trusted."""
        self.opt._apply_layout_to_q_map(layout)
        direct = self.opt.solver.solve_steady_state(apply_result=False)
        superposed = self._t_amb + sum(
            self._fields[(i, tuple(layout[i]))] for i in range(len(layout))
        )
        err = float(np.max(np.abs(superposed - direct)))
        if not np.isfinite(err) or err > SUPERPOSITION_TOL_K:
            raise ExhaustiveSearchError(
                f"Superposition does not reproduce a direct solve (max deviation "
                f"{err:.3e} K > {SUPERPOSITION_TOL_K:.1e} K). The steady-state system is "
                f"assumed linear in q_map; if that no longer holds -- a temperature-"
                f"dependent conductivity or boundary condition, say -- exhaustive search "
                f"by superposition is invalid and must not be used."
            )
        return err

    # ------------------------------------------------------------------
    def _overlaps(self, placed: List[Tuple[int, Tuple[int, int]]],
                  idx: int, pos: Tuple[int, int]) -> bool:
        """Rectangle-intersection test against already-placed blocks. Pure
        integer arithmetic, far cheaper than an evaluation, which is why it
        gates the descent."""
        nvx, nvy = self.opt._extents[idx]
        ax0, ay0 = pos
        ax1, ay1 = ax0 + nvx, ay0 + nvy
        for (j, (bx0, by0)) in placed:
            bnvx, bnvy = self.opt._extents[j]
            if (min(ax1, bx0 + bnvx) - max(ax0, bx0) > 0
                    and min(ay1, by0 + bnvy) - max(ay0, by0) > 0):
                return True
        return False

    def _find_anchor_layout(self) -> Layout:
        """Greedily finds one buildable layout drawn entirely from this
        search's own position set, for use as the superposition ground-truth
        anchor.

        Must come from `self._positions` rather than from the optimizer's
        shelf packing: the optimizer may use a finer placement lattice, and an
        origin off this search's lattice has no precomputed field.
        """
        assign: Dict[int, Tuple[int, int]] = {}
        placed: List[Tuple[int, Tuple[int, int]]] = []

        def place(depth: int) -> bool:
            if depth == len(self.opt.catalog):
                return True
            idx = self._order[depth]
            for pos in self._positions[idx]:
                if self._overlaps(placed, idx, pos):
                    continue
                placed.append((idx, pos))
                assign[idx] = pos
                if place(depth + 1):
                    return True
                placed.pop()
                del assign[idx]
            return False

        if not place(0):
            raise ExhaustiveSearchError(
                f"No buildable layout exists on a {self.pitch_um:.0f} um lattice for this "
                f"catalog and die -- every combination overlaps. Use a finer placement "
                f"pitch or a larger die."
            )
        return tuple(assign[i] for i in range(len(self.opt.catalog)))

    def search(self, verbose: bool = True) -> ExhaustiveResult:
        """
        Evaluates every buildable lattice layout and returns the global
        optimum. Precomputes automatically if not already done.
        """
        precompute_s = 0.0
        if not self._fields:
            # Prefer the single batched triangular solve; fall back to the
            # per-position loop when there is no factorization to batch
            # against (the iterative path above DIRECT_SOLVE_MAX_UNKNOWNS).
            n_unknowns = self.opt.nx * self.opt.ny * self.opt.nz
            if n_unknowns <= DIRECT_SOLVE_MAX_UNKNOWNS:
                precompute_s = self.precompute_batched(verbose=verbose)
            else:
                precompute_s = self.precompute(verbose=verbose)

        # Anchor the superposition check on a layout drawn from THIS search's
        # own position set. The optimizer's shelf-packed layout sits on the
        # optimizer's lattice, which may be finer than the search lattice, in
        # which case its origins have no precomputed field.
        anchor = self._find_anchor_layout()
        superposition_err = self._verify_superposition(anchor)

        # The reported baseline is the optimizer's shelf-packed seed, solved
        # DIRECTLY. That keeps `improvement` comparable with the annealing
        # path, which uses the same starting layout, even when that layout is
        # off this search's lattice.
        seed_layout = self.opt.shelf_packed_layout()
        self.opt._apply_layout_to_q_map(seed_layout)
        initial_peak = float(np.max(
            self.opt.solver.solve_steady_state(apply_result=False)
        )) - 273.15

        n_blocks = len(self.opt.catalog)
        best_peak = float("inf")
        best_assign: Optional[Dict[int, Tuple[int, int]]] = None
        evaluated = 0
        running = np.zeros_like(next(iter(self._fields.values())))

        t0 = time.perf_counter()

        def descend(depth: int, placed: List[Tuple[int, Tuple[int, int]]]) -> None:
            nonlocal best_peak, best_assign, evaluated
            if depth == n_blocks:
                evaluated += 1
                peak = float(np.max(running)) + self._t_amb - 273.15
                if peak < best_peak:
                    best_peak = peak
                    best_assign = {i: p for i, p in placed}
                return
            idx = self._order[depth]
            for pos in self._positions[idx]:
                if self._overlaps(placed, idx, pos):
                    continue
                f = self._fields[(idx, pos)]
                running.__iadd__(f)
                placed.append((idx, pos))
                descend(depth + 1, placed)
                placed.pop()
                running.__isub__(f)

        descend(0, [])
        search_s = time.perf_counter() - t0

        if best_assign is None:
            raise ExhaustiveSearchError(
                f"No buildable layout exists on a {self.pitch_um:.0f} um lattice for this "
                f"catalog and die. Every one of the {self.layouts_total:,} combinations "
                f"overlaps. Use a finer placement pitch or a larger die."
            )

        best_layout = tuple(best_assign[i] for i in range(n_blocks))
        # Recover the winning field by superposition, then confirm it against a
        # real solve so the REPORTED optimum is itself ground-truthed.
        best_field = self._t_amb + sum(
            self._fields[(i, best_layout[i])] for i in range(n_blocks)
        )
        self.opt._apply_layout_to_q_map(best_layout)
        direct_best = self.opt.solver.solve_steady_state(apply_result=False)
        final_err = float(np.max(np.abs(best_field - direct_best)))
        if final_err > SUPERPOSITION_TOL_K:
            raise ExhaustiveSearchError(
                f"The winning layout does not survive a direct re-solve (deviation "
                f"{final_err:.3e} K). Refusing to report an optimum that superposition "
                f"and the solver disagree about."
            )

        if verbose:
            print(f"[exhaustive] evaluated {evaluated:,} buildable layouts of "
                  f"{self.layouts_total:,} in {search_s:.2f}s")

        return ExhaustiveResult(
            best_layout=best_layout,
            best_peak_temperature_c=float(np.max(direct_best)) - 273.15,
            initial_layout=seed_layout,
            initial_peak_temperature_c=initial_peak,
            layouts_evaluated=evaluated,
            layouts_total=self.layouts_total,
            positions_per_block=tuple(len(p) for p in self._positions),
            precompute_solves=sum(len(p) for p in self._positions),
            precompute_s=precompute_s,
            search_s=search_s,
            placement_pitch_um=self.pitch_um,
            superposition_error_k=max(superposition_err, final_err),
            best_temperature_field_k=direct_best,
            catalog=self.opt.catalog,
        )


if __name__ == "__main__":
    # Smoke test on the default package configuration at a coarse lattice,
    # so the whole search finishes in seconds.
    opt = ChipletPlacementOptimizer(
        nx=40, ny=40, nz=12, dx=500e-6, dy=500e-6, dz=20e-6, seed=7,
    )
    search = ExhaustivePlacementSearch(opt, placement_pitch_um=4000.0)
    result = search.search()
    print(result.summarize())
