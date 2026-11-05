"""
ai/guided.py
ThermalForge :: exact placement search with a branching policy chosen per node.

The shipped search (studio.engine.branch_and_bound) fixes one order of parts
before it starts -- hottest first -- and at every depth branches on the next
part in that order. The order matters: the bound at a node is the placed
parts' exact rise plus every UNPLACED part's per-cell minimum, so which parts
are still unplaced decides how tight it is. Measured on the regression set,
no fixed alternative order beats hottest-first, and even handing the search
the true optimum as its starting incumbent saves at most 4.7x. The gain that
is left is in choosing the part to branch on from the state of each node.

This search does that. Everything that makes the answer exact is unchanged:

    the bound     placed exact rise + unplaced per-cell minima, a proven lower
                  bound whatever the order, so no layout that could beat the
                  incumbent is ever cut;
    the leaves    complete layouts are scored on every cell;
    symmetry      identical parts are always placed in member order, each at a
                  higher candidate index than the previous member -- a valid
                  canonical form under ANY branching order;
    ties          broken by the layout, not by arrival order.

A policy only chooses WHICH part to branch on next. The learned policy
(ai.guided_policy) imitates an expensive exact rule cheaply; a wrong choice
costs time, never the answer, and tests hold every policy to the shipped
search's optimum.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

TIE = 1e-12


class Stopped(Exception):
    """The caller asked the search to stop (should_stop returned True)."""


@dataclass
class GuidedStats:
    nodes: int = 0
    leaves: int = 0                 # complete layouts scored -- the certificate's layouts_scored
    prescreened: int = 0            # complete layouts excluded by the prior-cell bound, unscored
    prior_hits: int = 0             # of those, excluded by a LEARNED prior cell, not an analytic one
    bound_rows: int = 0             # partial layouts whose bound was computed, any depth
    policy_evals: int = 0           # extra bound rows a policy spent deciding
    pruned_overlap: int = 0
    pruned_symmetry: int = 0
    pruned_bound: int = 0
    pruned_constraint: int = 0      # partial layouts cut by a pairwise-cost cap (exact)
    choices: List[Tuple[int, int]] = field(default_factory=list)   # (depth, part) when recorded


@dataclass
class NodeView:
    """What a policy sees at a node. Everything is exact and cheap."""
    depth: int
    unplaced: List[int]                  # parts eligible to branch on now
    counts: Dict[int, int]               # allowed candidates per eligible part
    self_peak: Sequence[float]
    spread: Sequence[float]
    floor: Sequence[float]
    best: float
    bound_now: float                     # this node's own bound
    hot_cell: int                        # argmax cell of the node's bound row
    min_at_hot: Sequence[float]          # each part's minimum rise at hot_cell
    survivors: Callable[[int], Tuple[int, float]]   # exact: (children under best, min child bound)


Policy = Callable[[NodeView], int]


def shipped_policy(view: NodeView) -> int:
    """Hottest first, as the shipped search does: reproduces its tree."""
    return max(view.unplaced, key=lambda i: (view.self_peak[i], -i))


def fail_first_policy(view: NodeView) -> int:
    """The exact rule the learned policy imitates: branch on the part with
    the fewest children that survive the bound (ties: tightest bound, then
    hottest). Costs one bound row per allowed candidate of every eligible part
    -- counted in stats.policy_evals, because it is not free."""
    best_key, best_i = None, None
    for i in view.unplaced:
        n, lo = view.survivors(i)
        key = (n, -lo, -view.self_peak[i], i)
        if best_key is None or key < best_key:
            best_key, best_i = key, i
    return best_i


def guided_branch_and_bound(fields: List[np.ndarray], positions: List[np.ndarray],
                            extents: List[np.ndarray], signatures: List[tuple],
                            incumbent: Tuple[float, Optional[List[int]]],
                            policy: Policy = shipped_policy,
                            bound_cells: Optional[int] = None,
                            clearance_cells: int = 0,
                            improve_incumbent: bool = True,
                            record_choices: bool = False,
                            record_views: Optional[list] = None,
                            deadline_s: Optional[float] = None,
                            prescreen: bool = False,
                            prior_cells: int = 4,
                            prior_seed: Optional[Sequence[int]] = None,
                            pair_cost: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
                            cost_cap: float = float("inf"),
                            should_stop: Optional[Callable[[], bool]] = None,
                            on_progress: Optional[Callable[[GuidedStats, float], None]] = None,
                            ) -> Tuple[float, Optional[List[int]], GuidedStats]:
    """Exact minimum of the peak rise over every legal layout, branching on the
    part `policy` names at each node. Same contract as the engine's search.

    `prescreen`: before a complete layout is scored on every bound cell, check
    it on a few PRIOR cells -- the running layout's hot cell, the new part's
    own hottest cell, and the `prior_cells` cells that have bound the peak most
    often so far in this search (seeded by `prior_seed`, e.g. from a learned
    model of where peaks sit). A layout's peak is at least its value at any
    cell, so one that already reaches the incumbent there cannot win and is
    excluded unscored. Exact: it removes only layouts the full score would
    also reject, and it is counted (stats.prescreened), not hidden.

    `pair_cost` / `cost_cap`: an exact side constraint -- e.g. interconnect
    length. pair_cost[(a, b)][i, j] is the cost of a at candidate i with b at
    candidate j (a < b); a layout is legal only if its total is <= cost_cap.
    Costs are non-negative, so the placed pairs' sum is a lower bound on any
    completion and a partial layout already over the cap is cut exactly. This
    is what makes the inverse-design Pareto front a set of PROVEN optima."""
    k = len(fields)
    n_cells = fields[0].shape[1]
    nb = n_cells if bound_cells is None else max(1, min(int(bound_cells), n_cells))
    full = fields
    part = [np.ascontiguousarray(f[:, :nb]) for f in fields]
    self_peak = [float(f.max()) if f.size else 0.0 for f in fields]
    spread = [float(f.max(axis=1).max() - f.max(axis=1).min()) if f.size else 0.0 for f in fields]
    minima = [f.min(axis=0) for f in part]
    floor = [float(m.max()) for m in minima]
    stats = GuidedStats()
    t_end = None if deadline_s is None else time.perf_counter() + deadline_s
    # Stopping and reporting happen on a clock, not on a node count: a node is
    # cheap in one instance and expensive in another, and a user waiting to
    # pause should not have to wait for the next multiple of anything.
    last_tick = [time.perf_counter()]

    # identical parts: a group, placed in member order with increasing index
    groups: Dict[tuple, List[int]] = {}
    for i in range(k):
        groups.setdefault(signatures[i], []).append(i)
    group_of = {i: tuple(g) for g in groups.values() for i in g}

    compat: Dict[Tuple[int, int], np.ndarray] = {}
    for a in range(k):
        for b in range(k):
            if a == b:
                continue
            pa, pb = positions[a], positions[b]
            ea, eb = extents[a], extents[b]
            wa, ha = ea[:, 0][:, None], ea[:, 1][:, None]
            wb, hb = eb[:, 0][None, :], eb[:, 1][None, :]
            ax, ay = pa[:, 0][:, None], pa[:, 1][:, None]
            bx, by = pb[:, 0][None, :], pb[:, 1][None, :]
            overlap = ((np.minimum(ax + wa, bx + wb) + clearance_cells > np.maximum(ax, bx))
                       & (np.minimum(ay + ha, by + hb) + clearance_cells > np.maximum(ay, by)))
            compat[(a, b)] = ~overlap

    chosen: List[Optional[int]] = [None] * k
    linked: Dict[int, List[int]] = {i: [] for i in range(k)}
    if pair_cost:
        for (a, b), mat in pair_cost.items():
            if not a < b or mat.shape != (len(positions[a]), len(positions[b])):
                raise ValueError(f"pair_cost[{(a, b)}] must be keyed a < b and shaped by candidates")
            if (mat < 0).any():
                raise ValueError("pair costs must be non-negative for the cap to be a bound")
            linked[a].append(b)
            linked[b].append(a)
    constrained = bool(pair_cost) and np.isfinite(cost_cap)

    def added_cost(ci: int, idx: np.ndarray, placed: Sequence[Optional[int]]) -> np.ndarray:
        """Cost ci at each candidate in idx adds against the placed parts."""
        out = np.zeros(idx.size)
        for cj in linked[ci]:
            j = placed[cj]
            if j is None:
                continue
            out += pair_cost[(ci, cj)][idx, j] if ci < cj else pair_cost[(cj, ci)][j, idx]
        return out

    # per candidate: its own hottest bound cell, for the prior-cell screen
    self_arg = [np.argmax(pp, axis=1) if pp.size else np.zeros(0, dtype=int) for pp in part]
    self_max = [pp[np.arange(len(pp)), a] if pp.size else np.zeros(0) for pp, a in zip(part, self_arg)]
    binding = np.zeros(nb, dtype=np.int64)          # how often each cell set a scored peak
    prior_top = {"cells": np.zeros(0, dtype=np.int64), "age": 0}   # refreshed every 32 leaf nodes
    if prior_seed is not None:
        for rank, c in enumerate(prior_seed):
            if 0 <= int(c) < nb:
                binding[int(c)] += max(1, len(prior_seed) - rank)

    def allowed_for(ci: int) -> np.ndarray:
        mask = np.ones(len(positions[ci]), dtype=bool)
        for cj in range(k):
            if cj != ci and chosen[cj] is not None:
                mask &= compat[(cj, ci)][chosen[cj]]
        return mask

    def symmetry_floor(ci: int) -> int:
        """Lowest candidate index ci may take: above the previous member."""
        g = group_of[ci]
        pos = g.index(ci)
        if pos == 0:
            return 0
        prev = chosen[g[pos - 1]]
        return 0 if prev is None else prev + 1

    def eligible() -> List[int]:
        """Unplaced parts that may be branched on: for a group, only its next
        unplaced member, so members are always placed in order."""
        out = []
        for g in groups.values():
            for i in g:
                if chosen[i] is None:
                    out.append(i)
                    break
        return out

    def total_cost(layout: Sequence[Optional[int]]) -> float:
        tot = 0.0
        for (a, b), mat in (pair_cost or {}).items():
            if layout[a] is not None and layout[b] is not None:
                tot += float(mat[layout[a], layout[b]])
        return tot

    if constrained and incumbent[1] is not None and total_cost(incumbent[1]) > cost_cap + 1e-9:
        incumbent = (float("inf"), None)       # a starting layout over the cap is not a layout

    # -- incumbent: the same greedy + one-move improvement the engine uses ----
    best = {"value": incumbent[0], "choice": incumbent[1]}
    if improve_incumbent:
        order0 = [i for g in sorted(groups.values(), key=lambda g: -max(self_peak[i] for i in g))
                  for i in g]
        greedy: List[Optional[int]] = [None] * k
        running = np.zeros(n_cells)
        ok = True
        for ci in order0:
            mask = np.ones(len(positions[ci]), dtype=bool)
            for cj in range(k):
                if cj != ci and greedy[cj] is not None:
                    mask &= compat[(cj, ci)][greedy[cj]]
            idx = np.flatnonzero(mask)
            if constrained and idx.size:
                idx = idx[total_cost(greedy) + added_cost(ci, idx, greedy) <= cost_cap + 1e-9]
            if idx.size == 0:
                ok = False
                break
            j = int(idx[int(np.argmin((full[ci][idx] + running).max(axis=1)))])
            greedy[ci] = j
            running += full[ci][j]
        if ok:
            value = float(running.max())
            for _ in range(50):
                improved = False
                for ci in range(k):
                    others = running - full[ci][greedy[ci]]
                    mask = np.ones(len(positions[ci]), dtype=bool)
                    for cj in range(k):
                        if cj != ci:
                            mask &= compat[(cj, ci)][greedy[cj]]
                    idx = np.flatnonzero(mask)
                    if constrained:
                        others_cost = total_cost(greedy) - float(added_cost(
                            ci, np.array([greedy[ci]]), greedy)[0])
                        idx = idx[others_cost + added_cost(ci, idx, greedy) <= cost_cap + 1e-9]
                        if idx.size == 0:
                            continue
                    peaks = (full[ci][idx] + others).max(axis=1)
                    j = int(np.argmin(peaks))
                    if peaks[j] < value - TIE:
                        greedy[ci] = int(idx[j])
                        running = others + full[ci][greedy[ci]]
                        value = float(peaks[j])
                        improved = True
                if not improved:
                    break
            if value < best["value"]:
                best["value"], best["choice"] = value, list(greedy)

    def offer(value: float, choice: List[int]) -> None:
        if value < best["value"] - TIE or (
                abs(value - best["value"]) <= TIE
                and (best["choice"] is None or choice < best["choice"])):
            best["value"], best["choice"] = value, list(choice)

    class Deadline(Exception):
        pass

    def children(ci: int):
        """Allowed candidate indices of ci, and how many overlap / symmetry cut."""
        mask = allowed_for(ci)
        count = mask.size
        n_allowed = int(mask.sum())
        lo = symmetry_floor(ci)
        sym = int(mask[:lo].sum()) if lo else 0
        if lo:
            mask[:lo] = False
        return np.flatnonzero(mask), count - n_allowed, sym

    def descend(depth: int, run_full: np.ndarray, run_part: np.ndarray, rest: np.ndarray,
                spent: float = 0.0) -> None:
        now = time.perf_counter()
        if t_end is not None and now > t_end:
            raise Deadline()
        if now - last_tick[0] > 0.2:
            last_tick[0] = now
            # is_set() BLOCKS while the job is paused (studio.server.Control),
            # so this is where a pause takes effect as well as a cancel.
            if should_stop is not None and should_stop():
                raise Stopped()
            if on_progress is not None:
                on_progress(stats, best["value"])
        cands = eligible()
        if len(cands) == 1:
            ci = cands[0]
        else:
            bound_row = run_part + rest
            hot = int(np.argmax(bound_row))

            def survivors(i: int) -> Tuple[int, float]:
                idx, _, _ = children(i)
                stats.policy_evals += int(idx.size)
                if idx.size == 0:
                    return 0, float("inf")
                b = (part[i][idx] + run_part + (rest - minima[i])).max(axis=1)
                return int((b < best["value"]).sum()), float(b.min())

            view = NodeView(depth=depth, unplaced=cands,
                            counts={i: int(allowed_for(i).sum()) for i in cands},
                            self_peak=self_peak, spread=spread, floor=floor,
                            best=best["value"], bound_now=float(bound_row[hot]),
                            hot_cell=hot, min_at_hot=[float(m[hot]) for m in minima],
                            survivors=survivors)
            ci = policy(view)
            if ci not in cands:
                raise ValueError(f"policy chose part {ci}, which is not eligible here")
            if record_views is not None:
                record_views.append((view, ci))
        if record_choices:
            stats.choices.append((depth, ci))

        rest_wo = rest - minima[ci]
        idx, n_over, n_sym = children(ci)
        stats.pruned_overlap += n_over
        stats.pruned_symmetry += n_sym
        costs = None
        if constrained and idx.size:
            costs = spent + added_cost(ci, idx, chosen)
            ok = costs <= cost_cap + 1e-9
            stats.pruned_constraint += int(idx.size - ok.sum())
            idx, costs = idx[ok], costs[ok]
        if idx.size == 0:
            return
        stats.nodes += 1
        last = all(chosen[j] is not None for j in range(k) if j != ci)
        if last:
            ceiling = best["value"]
            if prescreen and idx.size:
                hot = int(np.argmax(run_part))
                cheap = np.maximum(run_part[hot] + part[ci][idx, hot],
                                   self_max[ci][idx] + run_part[self_arg[ci][idx]])
                keep = cheap < ceiling
                if prior_cells > 0 and binding.any():
                    if prior_top["age"] <= 0:
                        top = np.argpartition(binding, -prior_cells)[-prior_cells:]                             if binding.size > prior_cells else np.arange(binding.size)
                        prior_top["cells"] = top[binding[top] > 0]
                        prior_top["age"] = 32
                    prior_top["age"] -= 1
                    cells = prior_top["cells"]
                    alive = np.flatnonzero(keep)
                    if cells.size and alive.size:
                        # only where the analytic cells did not already decide
                        learned = (part[ci][np.ix_(idx[alive], cells)] + run_part[cells]).max(axis=1)
                        dropped = alive[learned >= ceiling]
                        stats.prior_hits += int(dropped.size)
                        keep[dropped] = False
                stats.prescreened += int(idx.size - keep.sum())
                idx = idx[keep]
            stats.leaves += int(idx.size)
            if idx.size == 0:
                chosen[ci] = None
                return
            # only now are the survivors scored on every bound cell
            partial = part[ci][idx] + run_part
            scores = partial.max(axis=1)
            if prescreen:
                np.add.at(binding, np.argmax(partial, axis=1), 1)
            contenders = idx[scores < ceiling]
            if contenders.size:
                peaks = (full[ci][contenders] + run_full).max(axis=1)
                j = int(np.argmin(peaks))
                if peaks[j] < ceiling:
                    chosen[ci] = int(contenders[j])
                    offer(float(peaks[j]), list(chosen))
            chosen[ci] = None
            return
        if prescreen and idx.size:
            # A child whose bound already reaches the incumbent on two cells is
            # cut before its full bound row is built. The traversal is the same:
            # such a child sorts past the break below, and the incumbent only falls.
            base = run_part + rest_wo
            hot = int(np.argmax(base))
            cheap = np.maximum(base[hot] + part[ci][idx, hot],
                               self_max[ci][idx] + base[self_arg[ci][idx]])
            live = cheap < best["value"]
            stats.pruned_bound += int(idx.size - live.sum())
            idx = idx[live]
            if costs is not None:
                costs = costs[live]
            if idx.size == 0:
                return
        partial = part[ci][idx] + run_part
        bounds = (partial + rest_wo).max(axis=1)
        stats.bound_rows += int(idx.size)
        ranked = np.argsort(bounds, kind="stable")
        for rank, j in enumerate(ranked):
            if bounds[j] >= best["value"]:
                stats.pruned_bound += int(ranked.size - rank)
                break
            chosen[ci] = int(idx[j])
            descend(depth + 1, run_full + full[ci][idx[j]], partial[j], rest_wo,
                    spent if costs is None else float(costs[j]))
        chosen[ci] = None

    rest0 = np.sum(minima, axis=0) if k else np.zeros(nb)
    try:
        descend(0, np.zeros(n_cells), np.zeros(nb), rest0)
    except Deadline:
        return float("nan"), None, stats
    if on_progress is not None:
        on_progress(stats, best["value"])
    return best["value"], best["choice"], stats
