"""
thermal_solver.py
RISC3D-AI :: 3D IC / Advanced Packaging Thermal Modeling Framework

ThermalSolver consumes the voxelized arrays produced by GridBuilder
(kx_map, ky_map, kz_map, rho_map, cp_map, q_map, boundary_mask) and solves
the transient anisotropic heat conduction equation:

    rho*Cp*(dT/dt) = d/dx(kx dT/dx) + d/dy(ky dT/dy) + d/dz(kz dT/dz) + Q

Phase 1 (this file): fully vectorized explicit 7-point-stencil FDM (forward
Euler in time, conservative finite-volume in space) with:
  - Harmonic-mean face conductivities across heterogeneous material
    interfaces (correct treatment for the Si/Cu/TIM/bump-underfill stack;
    a naive arithmetic mean or single-node k badly misrepresents flux at
    the low-k underfill / high-k Cu interfaces).
  - Neumann (adiabatic) BCs on top and all 4 side faces.
  - Dirichlet or Robin (convective) BC on the bottom (heat-sink) face.
  - An automated, per-cell stability audit (compute_max_dt) that inspects
    the actual assembled coefficients -- not a single global k_max estimate
    -- and will correctly flag the Cu-dominated worst-case cells.

Phase 2 (future): an implicit / Crank-Nicolson / ADI time-stepper will be
added as ThermalSolver.step_implicit(), reusing the same face-conductivity
and boundary-coefficient assembly (_assemble_face_conductivities,
_assemble_boundary_coefficients) so no interface changes are required
upstream (GridBuilder) or downstream (analytics/hotspot code).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("RISC3D-AI.ThermalSolver")

# Explicit public export list. `SolverBCConfig` in particular must be
# importable as `from src.thermal_solver import SolverBCConfig` -- it is
# the boundary-condition configuration class shared by main.py,
# config_loader.py, and ThermalSolver.from_grid_builder().
__all__ = [
    "ThermalSolver",
    "SolverBCConfig",
    "ThermalSolverError",
    "StabilityWarning",
    "REQUIRED_ARRAY_KEYS",
]

REQUIRED_ARRAY_KEYS = (
    "kx_map", "ky_map", "kz_map", "rho_map", "cp_map", "q_map", "boundary_mask",
)

# Above this unknown count, direct sparse LU factorization (scipy.sparse.linalg.splu)
# suffers severe fill-in on a 3D 7-point-stencil matrix (no nested-dissection
# reordering is applied), so factorization time grows super-linearly. Measured
# directly on this codebase (first-step time, i.e. factorization + first solve):
# 0.062s at N=3,375, climbing to 1.79s at N=15,625 -- a ~4.6x increase in N
# producing a ~29x increase in time, confirming super-linear fill-in growth
# well before the 20,000-unknown cutover chosen here. Above the threshold,
# step_implicit() automatically switches to a Jacobi-preconditioned BiCGSTAB
# iterative solve, independently confirmed fast and convergent at N=27,000
# (0.019s/solve) and N=64,000 (0.045s/solve).
#
# NOTE ON PRECONDITIONER CHOICE: Jacobi (diagonal) was chosen over ILU because
# (I - dt/2*M) is diagonally dominant by construction (the same property
# compute_max_dt() checks for the explicit scheme), so cheap Jacobi-
# preconditioned iterations converge quickly without paying ILU's fill-in
# cost on this stencil.
#
# SCOPE OF THAT NOTE: it is about the CRANK-NICOLSON matrix (I - dt/2*M),
# whose identity term is what makes it strictly diagonally dominant. It does
# NOT transfer to the STEADY-STATE system, which solves M itself. Measured on
# a 40x40x24 package-scale grid:
#
#     matrix                     min |a_ii|/sum|a_ij|   rows at ratio ~1.0
#     M (steady state)                 1.000000               95.83%
#     (I - dt/2*M), dt=1e-6            2.1125                  0.00%
#
# M's only diagonal dominance comes from the Robin boundary cells (1600 of
# 38,400 = 4.17%, exactly the complement of the 95.83% above); every interior
# row is exactly weakly dominant, so Jacobi has essentially nothing to
# precondition and BiCGSTAB stalls. solve_steady_state() previously inherited
# this preconditioner choice and failed to converge on any package-scale grid.
# See STEADY-STATE BACKEND below for what it uses instead.
DIRECT_SOLVE_MAX_UNKNOWNS = 20_000
ITERATIVE_RTOL = 1e-8
ITERATIVE_MAXITER = 2000

# STEADY-STATE BACKEND.
# _build_conduction_operator writes M[a,b] = coeff/rho_cp[a] and
# M[b,a] = coeff/rho_cp[b], i.e. M = D^-1 A with D = diag(rho_cp) and A the
# symmetric conductance matrix. M is therefore asymmetric wherever rho*cp
# differs across a material interface (measured: ||M-M^T||/||M|| = 2.45e-3),
# but S = -D M = -A is EXACTLY symmetric (measured: ||S-S^T||/||S|| = 0.0e+00)
# and positive definite. Solving M T = -f is equivalent to S T = D f, which
# admits conjugate gradients -- the correct Krylov method for an SPD system.
#
# Measured at 40x40x24 (N=38,400, 50:1 mesh aspect), against splu(M) as ground
# truth (peak 200.3900 degC):
#     CG + Jacobi on S            peak 200.3900 degC, err 4.8e-09, 3.65s, 3062 iters
#     CG + ILU on S               FAILED (spilu is not symmetric, so the
#                                 preconditioner is not SPD and CG breaks down;
#                                 scipy has no incomplete Cholesky)
#     BiCGSTAB + Jacobi on S      FAILED (info=5000)
# so it is the Krylov method, not the preconditioner, that matters here.
#
# CG also scales where direct factorization cannot: splu fill grows ~N^0.59
# on this stencil (~44M nonzeros already at N=38,400), which is why the
# direct path keeps its DIRECT_SOLVE_MAX_UNKNOWNS ceiling. Measured CG
# on the refinement ladder: 935 iters/0.07s at N=4,800; 3,062/3.95s at
# N=38,400; 8,425/246.8s at N=307,200.
STEADY_CG_RTOL = 1e-12
STEADY_CG_MAXITER = 200_000

# VERIFICATION, NOT TRUST.
# A solver's own convergence flag says "my iteration stopped", not "the answer
# is right": scipy returns info=0 as soon as its internal residual estimate
# crosses rtol, and that estimate can be wrong (stagnation, loss of
# orthogonality, breakdown near machine precision). A direct factorization can
# likewise return a badly wrong answer on an ill-conditioned operator without
# raising anything at all. So solve_steady_state() MEASURES its answer, with
# three checks, and raises if any of them fails:
#
#   1. ALGEBRAIC -- ||Ax - b|| / ||b||, against the system actually solved.
#      Confirms the linear algebra. On its own it only proves "I solved the
#      system I was handed"; it cannot detect a wrongly assembled operator.
#
#   2. PHYSICAL -- at steady state every generated watt must leave through the
#      Robin/Dirichlet boundary, so
#          sum(boundary_coeff * (T_bottom - T_ambient)) == sum(q_map / rho_cp)
#      exactly, independent of mesh, materials, layout and solver. This is a
#      genuinely independent check: it follows from energy conservation rather
#      than from the discretization, so it catches operator-assembly errors
#      that a residual cannot.
#
#   3. ORDERING -- peak rise >= mean bottom-surface rise. Necessarily true,
#      since the peak is a maximum over a field that contains that surface, so
#      it is free when the answer is right. It exists to catch a REPORTING
#      error rather than a solver one: any quantity that is not actually a
#      peak (a temperature difference, say) fails it immediately. Check 2
#      pins the mean bottom rise to P/(h*A), so this transitively enforces
#      peak_rise >= P/(h*A) -- the convective floor.
#
# All three are cheap beside the solve itself (two sparse mat-vecs and a few
# reductions), so they run on EVERY call rather than as an opt-in debug mode.
STEADY_RESIDUAL_TOL = 1e-8
# The energy imbalance is measured in EQUIVALENT KELVIN, not as a ratio:
#     imbalance_K = |P_in - P_out| / sum(boundary_coeff / rho_cp)
# i.e. the uniform bottom-surface temperature error that would account for the
# mismatch. A ratio is unusable here because the zero-power case makes both
# sides vanish, and dividing float noise by float noise reports 100% error on
# a perfectly correct solve. Equivalent kelvin stays meaningful at any power,
# including none, and is directly interpretable against the temperatures the
# model reports.
STEADY_ENERGY_BALANCE_TOL_K = 1e-6

_EPS = 1e-30  # guards against div-by-zero in harmonic means / coefficient sums


class ThermalSolverError(Exception):
    """Raised on invalid grid input, boundary configuration, or a violated
    numerical-stability condition."""


class StabilityWarning(RuntimeWarning):
    """Raised (as a warning, not exception) when a requested dt is within
    margin of the CFL bound but the caller explicitly allows override."""


@dataclass
class SolverBCConfig:
    """Boundary-condition parameters, mirrored from GridBuilder.GridConfig so
    ThermalSolver can be constructed standalone (e.g. in tests) or directly
    from a GridBuilder instance."""
    dx: float
    dy: float
    dz: float
    t_ambient_k: float = 298.15
    h_convective: float = 2500.0
    bottom_bc: str = "robin"     # "robin" | "dirichlet"
    top_bc: str = "neumann"      # only "neumann" supported (adiabatic)
    side_bc: str = "neumann"     # only "neumann" supported (adiabatic)

    def validate(self) -> None:
        if self.dx <= 0 or self.dy <= 0 or self.dz <= 0:
            raise ThermalSolverError(f"Grid spacing must be positive, got dx={self.dx}, dy={self.dy}, dz={self.dz}")
        if self.bottom_bc not in ("robin", "dirichlet"):
            raise ThermalSolverError(f"bottom_bc must be 'robin' or 'dirichlet', got '{self.bottom_bc}'")
        if self.top_bc != "neumann":
            raise ThermalSolverError("top_bc: only 'neumann' (adiabatic) is currently supported.")
        if self.side_bc != "neumann":
            raise ThermalSolverError("side_bc: only 'neumann' (adiabatic) is currently supported.")
        if self.bottom_bc == "robin" and self.h_convective <= 0:
            raise ThermalSolverError(f"h_convective must be > 0 for Robin BC, got {self.h_convective}")


class ThermalSolver:
    """
    Explicit finite-volume / finite-difference transient thermal solver for
    heterogeneous, anisotropic 3D IC / advanced-packaging stacks.

    Array indexing convention matches GridBuilder: [ix, iy, iz], iz=0 is the
    bottom (heat-sink) face, iz=nz-1 is the top (adiabatic) face.
    """

    def __init__(
        self,
        grid_arrays: Dict[str, np.ndarray],
        bc_config: SolverBCConfig,
        initial_temperature_k: Optional[np.ndarray] = None,
    ):
        self._validate_grid_arrays(grid_arrays)
        bc_config.validate()

        self.kx_map = np.asarray(grid_arrays["kx_map"], dtype=np.float64)
        self.ky_map = np.asarray(grid_arrays["ky_map"], dtype=np.float64)
        self.kz_map = np.asarray(grid_arrays["kz_map"], dtype=np.float64)
        self.rho_map = np.asarray(grid_arrays["rho_map"], dtype=np.float64)
        self.cp_map = np.asarray(grid_arrays["cp_map"], dtype=np.float64)
        self.q_map = np.asarray(grid_arrays["q_map"], dtype=np.float64)
        self.boundary_mask = np.asarray(grid_arrays["boundary_mask"])

        self.shape: Tuple[int, int, int] = self.kx_map.shape
        self.nx, self.ny, self.nz = self.shape
        self.bc = bc_config

        rho_cp = self.rho_map * self.cp_map
        if np.any(rho_cp <= 0):
            raise ThermalSolverError(
                "rho_map * cp_map contains non-positive values; every voxel must have "
                "physical density and specific heat assigned (did you call "
                "GridBuilder.validate_grid()?)."
            )
        self._rho_cp = rho_cp

        if initial_temperature_k is None:
            self.T = np.full(self.shape, self.bc.t_ambient_k, dtype=np.float64)
        else:
            init = np.asarray(initial_temperature_k, dtype=np.float64)
            if init.shape != self.shape:
                raise ThermalSolverError(
                    f"initial_temperature_k shape {init.shape} does not match grid shape {self.shape}"
                )
            if not np.all(np.isfinite(init)):
                raise ThermalSolverError("initial_temperature_k contains NaN/Inf values.")
            self.T = init.copy()

        self.time_s: float = 0.0
        self.step_count: int = 0

        # Pre-assemble static (time-invariant) face conductivities and boundary
        # coefficients once at construction -- these depend only on the material
        # maps, not on T, so recomputing them every step would be wasted work.
        self._kx_face, self._ky_face, self._kz_face = self._assemble_face_conductivities()
        self._bottom_boundary_coeff = self._assemble_boundary_coefficients()

        # Sparse linear conduction operator for the implicit (Crank-Nicolson)
        # time-stepper. Built once here since it depends only on the static
        # material maps and BC configuration, not on T or dt. step_implicit()
        # reuses this directly; only the (I -+ dt/2 * M) system matrices and
        # their LU factorizations vary with dt, and those are cached
        # separately in _cn_lu_cache since a run() typically reuses one dt
        # across many steps.
        self._conduction_operator, self._boundary_forcing = self._build_conduction_operator()
        self._cn_solver_cache: Dict[float, Dict[str, object]] = {}
        # Steady-state solve caches. Both depend only on the static material
        # maps and BC, never on T or q_map, so they survive every q_map change
        # -- which is what makes a placement sweep (src/optimizer.py moves
        # chiplets by rewriting q_map alone) cheap: one factorization, then a
        # triangular solve per candidate.
        self._steady_lu: Optional[object] = None          # splu(M), direct path
        self._steady_spd: Optional[Tuple[sparse.csr_matrix, np.ndarray]] = None  # (S, diag(S))
        # Reference explicit-CFL dt, computed once, used only to flag (via a
        # warning, never an error) implicit steps taken at a dt so large
        # relative to the diffusive timescale that Crank-Nicolson's known
        # non-monotonic-oscillation risk becomes a real accuracy concern --
        # CN remains unconditionally *stable* at any dt, but "stable" is not
        # the same guarantee as "physically smooth."
        self._explicit_dt_reference = self.compute_max_dt(safety_factor=1.0)

        logger.info(
            "ThermalSolver initialized: shape=%s, dx=%.2eum dy=%.2eum dz=%.2eum, "
            "bottom_bc=%s, t_ambient=%.2fK",
            self.shape, self.bc.dx * 1e6, self.bc.dy * 1e6, self.bc.dz * 1e6,
            self.bc.bottom_bc, self.bc.t_ambient_k,
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_grid_arrays(grid_arrays: Dict[str, np.ndarray]) -> None:
        missing = [k for k in REQUIRED_ARRAY_KEYS if k not in grid_arrays]
        if missing:
            raise ThermalSolverError(f"grid_arrays missing required keys: {missing}")
        shapes = {k: np.asarray(grid_arrays[k]).shape for k in REQUIRED_ARRAY_KEYS}
        unique_shapes = set(shapes.values())
        if len(unique_shapes) != 1:
            raise ThermalSolverError(f"grid_arrays have mismatched shapes: {shapes}")
        for k in ("kx_map", "ky_map", "kz_map", "rho_map", "cp_map", "q_map"):
            arr = np.asarray(grid_arrays[k])
            if not np.all(np.isfinite(arr)):
                raise ThermalSolverError(f"grid_arrays['{k}'] contains NaN/Inf values.")

    @classmethod
    def from_grid_builder(cls, grid_builder, initial_temperature_k: Optional[np.ndarray] = None) -> "ThermalSolver":
        """
        Convenience constructor: builds a ThermalSolver directly from a
        (validated, boundary-tagged) GridBuilder instance, pulling grid
        spacing and BC parameters from grid_builder.config.
        """
        try:
            grid_builder.validate_grid()
        except AttributeError as exc:
            raise ThermalSolverError(f"grid_builder does not expose validate_grid(): {exc}") from exc

        arrays = grid_builder.export_arrays()
        cfg = grid_builder.config
        bc_config = SolverBCConfig(
            dx=cfg.dx, dy=cfg.dy, dz=cfg.dz,
            t_ambient_k=cfg.t_ambient_k, h_convective=cfg.h_convective,
            bottom_bc=cfg.bottom_bc, top_bc=cfg.top_bc, side_bc=cfg.side_bc,
        )
        return cls(arrays, bc_config, initial_temperature_k=initial_temperature_k)

    @staticmethod
    def _harmonic_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Harmonic mean of adjacent-cell conductivities for face flux.
        Correctly weights heat flux across heterogeneous material interfaces
        (e.g. Cu-to-underfill) -- an arithmetic mean overestimates flux
        through a thin low-k layer sandwiched between high-k neighbors."""
        return 2.0 * a * b / (a + b + _EPS)

    def _assemble_face_conductivities(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes harmonic-mean conductivity at every internal face of the
        domain, per axis. Shapes:
          kx_face: (nx-1, ny, nz)  -- faces between (i, i+1) along x
          ky_face: (nx, ny-1, nz)  -- faces between (j, j+1) along y
          kz_face: (nx, ny, nz-1)  -- faces between (k, k+1) along z
        """
        kx_face = self._harmonic_mean(self.kx_map[:-1, :, :], self.kx_map[1:, :, :])
        ky_face = self._harmonic_mean(self.ky_map[:, :-1, :], self.ky_map[:, 1:, :])
        kz_face = self._harmonic_mean(self.kz_map[:, :, :-1], self.kz_map[:, :, 1:])
        return kx_face, ky_face, kz_face

    def _assemble_boundary_coefficients(self) -> np.ndarray:
        """
        Coefficient (W/m^3K per unit T-difference, i.e. units matching
        k/d^2) for the bottom (iz=0) boundary face, per the chosen BC:
          - Dirichlet: half-cell-distance ghost at fixed T_ambient ->
                coeff = 2 * kz[:, :, 0] / dz^2
          - Robin (convective film): coeff = h_convective / dz
        Returned as a (nx, ny) array (broadcastable over the bottom layer).
        """
        if self.bc.bottom_bc == "dirichlet":
            coeff = 2.0 * self.kz_map[:, :, 0] / (self.bc.dz ** 2)
        else:  # robin
            coeff = np.full((self.nx, self.ny), self.bc.h_convective / self.bc.dz, dtype=np.float64)
        return coeff

    def _build_conduction_operator(self) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """
        Assembles the sparse linear conduction operator M (shape N x N,
        N = nx*ny*nz, C-order flattening matching T.flatten()/.reshape())
        and the constant boundary-forcing vector b (shape N,) such that:

            dT/dt = M @ T_flat + b + Q_flat / rho_cp_flat

        i.e. M encodes exactly the same physics as _compute_divergence()
        (harmonic-mean face conductivities, adiabatic top/sides, Robin/
        Dirichlet bottom BC) but as an explicit sparse matrix instead of a
        vectorized stencil pass -- required for the implicit (Crank-
        Nicolson) time-stepper, which needs to solve a linear system rather
        than just evaluate the operator once per step.

        This is assembled ONCE at construction (materials are static) and
        reused by every step_implicit() call regardless of dt.
        """
        nx, ny, nz = self.shape
        N = nx * ny * nz
        rho_cp_flat = self._rho_cp.reshape(N)

        rows: List[np.ndarray] = []
        cols: List[np.ndarray] = []
        data: List[np.ndarray] = []

        def flat_idx(ii: np.ndarray, jj: np.ndarray, kk: np.ndarray) -> np.ndarray:
            return (ii * ny + jj) * nz + kk

        ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")

        def add_axis_couplings(face_k: np.ndarray, d: float, axis: int) -> None:
            # face_k has shape one-shorter along `axis` than self.shape.
            # Build index arrays for the two cells (a, b) sharing each face.
            if axis == 0:
                ia, ja, ka = ii[:-1, :, :], jj[:-1, :, :], kk[:-1, :, :]
                ib, jb, kb = ii[1:, :, :], jj[1:, :, :], kk[1:, :, :]
            elif axis == 1:
                ia, ja, ka = ii[:, :-1, :], jj[:, :-1, :], kk[:, :-1, :]
                ib, jb, kb = ii[:, 1:, :], jj[:, 1:, :], kk[:, 1:, :]
            else:
                ia, ja, ka = ii[:, :, :-1], jj[:, :, :-1], kk[:, :, :-1]
                ib, jb, kb = ii[:, :, 1:], jj[:, :, 1:], kk[:, :, 1:]

            idx_a = flat_idx(ia, ja, ka).ravel()
            idx_b = flat_idx(ib, jb, kb).ravel()
            coeff = (face_k / (d ** 2)).ravel()

            rho_cp_a = rho_cp_flat[idx_a]
            rho_cp_b = rho_cp_flat[idx_b]

            # Off-diagonal: contribution of neighbor b to cell a's rate, and
            # vice versa. Diagonal: negative self-coupling (outflow) at each.
            rows.append(idx_a); cols.append(idx_b); data.append(coeff / rho_cp_a)
            rows.append(idx_a); cols.append(idx_a); data.append(-coeff / rho_cp_a)
            rows.append(idx_b); cols.append(idx_a); data.append(coeff / rho_cp_b)
            rows.append(idx_b); cols.append(idx_b); data.append(-coeff / rho_cp_b)

        add_axis_couplings(self._kx_face, self.bc.dx, axis=0)
        add_axis_couplings(self._ky_face, self.bc.dy, axis=1)
        add_axis_couplings(self._kz_face, self.bc.dz, axis=2)

        # Bottom (z=0) boundary term: self-coupling only (references the
        # external T_ambient, which becomes part of the constant forcing
        # vector `b`, not the linear operator M).
        idx_bottom = flat_idx(ii[:, :, 0], jj[:, :, 0], kk[:, :, 0]).ravel()
        coeff_b_flat = self._bottom_boundary_coeff.ravel()
        rho_cp_bottom = rho_cp_flat[idx_bottom]
        rows.append(idx_bottom); cols.append(idx_bottom); data.append(-coeff_b_flat / rho_cp_bottom)

        row_arr = np.concatenate(rows)
        col_arr = np.concatenate(cols)
        data_arr = np.concatenate(data)

        M = sparse.coo_matrix((data_arr, (row_arr, col_arr)), shape=(N, N)).tocsr()
        M.sum_duplicates()

        boundary_forcing = np.zeros(N, dtype=np.float64)
        boundary_forcing[idx_bottom] = (coeff_b_flat * self.bc.t_ambient_k) / rho_cp_bottom

        return M, boundary_forcing

    # ------------------------------------------------------------------
    # Spatial discretization: divergence (conduction) operator
    # ------------------------------------------------------------------
    def _compute_divergence(self, T: np.ndarray) -> np.ndarray:
        """
        Computes the full conservative divergence term
            d/dx(kx dT/dx) + d/dy(ky dT/dy) + d/dz(kz dT/dz)
        evaluated at the current temperature field T, with adiabatic
        (Neumann) BCs on top/sides and the configured BC on the bottom face.
        Returns an array of shape self.shape with units W/m^3 (pre rho*cp
        division).
        """
        div = np.zeros(self.shape, dtype=np.float64)

        # --- x-direction (adiabatic both ends) ---
        flux_x = self._kx_face * (T[1:, :, :] - T[:-1, :, :]) / self.bc.dx  # (nx-1,ny,nz)
        div[1:-1, :, :] += (flux_x[1:, :, :] - flux_x[:-1, :, :]) / self.bc.dx
        div[0, :, :] += flux_x[0, :, :] / self.bc.dx
        div[-1, :, :] += -flux_x[-1, :, :] / self.bc.dx

        # --- y-direction (adiabatic both ends) ---
        flux_y = self._ky_face * (T[:, 1:, :] - T[:, :-1, :]) / self.bc.dy  # (nx,ny-1,nz)
        div[:, 1:-1, :] += (flux_y[:, 1:, :] - flux_y[:, :-1, :]) / self.bc.dy
        div[:, 0, :] += flux_y[:, 0, :] / self.bc.dy
        div[:, -1, :] += -flux_y[:, -1, :] / self.bc.dy

        # --- z-direction: interior faces (top is adiabatic, handled by the
        # same "no flux beyond domain" logic as x/y; bottom gets the special
        # Robin/Dirichlet boundary term added on top of the interior coupling
        # to the z=1 layer) ---
        flux_z = self._kz_face * (T[:, :, 1:] - T[:, :, :-1]) / self.bc.dz  # (nx,ny,nz-1)
        div[:, :, 1:-1] += (flux_z[:, :, 1:] - flux_z[:, :, :-1]) / self.bc.dz
        div[:, :, -1] += -flux_z[:, :, -1] / self.bc.dz  # top: adiabatic, no term beyond domain

        # z=0 layer: interior coupling to z=1 (from flux_z[...,0]) PLUS the
        # boundary exchange with T_ambient through the configured BC.
        if self.bc.bottom_bc == "dirichlet":
            boundary_flux = self._bottom_boundary_coeff * (T[:, :, 0] - self.bc.t_ambient_k)
        else:  # robin
            boundary_flux = self._bottom_boundary_coeff * (T[:, :, 0] - self.bc.t_ambient_k)
            # NOTE: for Robin the coefficient already equals h/dz (units differ
            # from the Dirichlet 2k/dz^2 case) but the flux expression has the
            # same form: coeff * (T_cell - T_ambient).
        div[:, :, 0] += flux_z[:, :, 0] / self.bc.dz - boundary_flux

        return div

    # ------------------------------------------------------------------
    # Stability analysis (automated CFL / diagonal-dominance check)
    # ------------------------------------------------------------------
    def _assemble_total_coefficient_field(self) -> np.ndarray:
        """
        Per-cell sum of all outgoing coupling coefficients (internal
        conduction faces + boundary Robin/Dirichlet term where present),
        each with units W/m^3K per unit temperature difference (i.e.
        k_face/d^2 or h/dz). This is the diagonal term whose reciprocal
        (scaled by rho*cp) bounds the explicit timestep for guaranteed
        positivity / diagonal dominance of the update -- a stricter and
        more accurate criterion than a single global k_max estimate,
        because it is evaluated per-cell against the ACTUAL assembled
        heterogeneous coefficients (e.g. it will correctly identify Cu
        TSV/interposer cells as the binding constraint).
        """
        total = np.zeros(self.shape, dtype=np.float64)

        cx = self._kx_face / (self.bc.dx ** 2)  # (nx-1,ny,nz)
        left_x = np.zeros(self.shape); left_x[1:, :, :] = cx
        right_x = np.zeros(self.shape); right_x[:-1, :, :] = cx
        total += left_x + right_x

        cy = self._ky_face / (self.bc.dy ** 2)  # (nx,ny-1,nz)
        left_y = np.zeros(self.shape); left_y[:, 1:, :] = cy
        right_y = np.zeros(self.shape); right_y[:, :-1, :] = cy
        total += left_y + right_y

        cz = self._kz_face / (self.bc.dz ** 2)  # (nx,ny,nz-1)
        left_z = np.zeros(self.shape); left_z[:, :, 1:] = cz
        right_z = np.zeros(self.shape); right_z[:, :, :-1] = cz
        total += left_z + right_z

        # Bottom boundary coefficient adds to the diagonal of the z=0 layer.
        if self.bc.bottom_bc == "dirichlet":
            total[:, :, 0] += self._bottom_boundary_coeff
        else:
            total[:, :, 0] += self._bottom_boundary_coeff  # h/dz, same additive role

        return total

    def compute_max_dt(self, safety_factor: float = 0.9) -> float:
        """
        Automated stability audit: dynamically inspects the assembled grid
        (conductivities, spacing, boundary coefficients) and returns the
        largest timestep (seconds) for which the explicit update remains
        diagonally dominant (i.e. numerically stable) at every cell,
        scaled by safety_factor for margin.

        dt_max = safety_factor * min_over_cells[ rho*cp / total_coeff_cell ]
        """
        if not (0.0 < safety_factor <= 1.0):
            raise ThermalSolverError(f"safety_factor must be in (0, 1], got {safety_factor}")

        total_coeff = self._assemble_total_coefficient_field()
        if np.any(total_coeff <= 0):
            raise ThermalSolverError(
                "Encountered cell(s) with zero total coupling coefficient (fully isolated "
                "voxel with no conduction paths and no boundary term); check grid assembly."
            )
        dt_per_cell = self._rho_cp / total_coeff
        dt_max = float(np.min(dt_per_cell)) * safety_factor

        if not np.isfinite(dt_max) or dt_max <= 0:
            raise ThermalSolverError(f"Computed non-physical dt_max={dt_max}; check material/grid assembly.")

        return dt_max

    def estimate_time_constant(self) -> float:
        """
        First-order thermal time constant of the whole package, in seconds:

            tau = C_total / G_boundary
                = [sum(rho*cp * V_cell)] / [sum(boundary_coeff * V_cell)]

        i.e. the lumped R*C of the stack against its heat-sink boundary. This
        is the timescale on which a transient run approaches equilibrium: at
        t = tau the field has covered 1 - 1/e = 63% of its final rise, at 3*tau
        95%, at 5*tau 99.3%.

        It is computed from the ACTUAL assembled grid -- the real rho*cp of
        every voxel and the real boundary coefficients, whether Robin or
        Dirichlet -- not from a hardcoded constant, so it stays correct when
        the stack, mesh or boundary condition changes.

        Sanity check against the closed form for a uniform slab with a Robin
        base: boundary_coeff = h/dz, so G = sum(h/dz * dx*dy*dz) = h*A and
        C = rho*cp*A*t, giving tau = rho*cp*t/h -- independent of area, as it
        must be.

        WHY THIS MATTERS: a transient run shorter than a few tau has not
        reached steady state, and any threshold or safety verdict drawn from
        it is meaningless. Callers should compare their simulated duration
        against this value before making claims about the result. main.py
        does exactly that.
        """
        v_cell = self.bc.dx * self.bc.dy * self.bc.dz
        c_total = float(np.sum(self._rho_cp)) * v_cell            # J/K
        g_boundary = float(np.sum(self._bottom_boundary_coeff)) * v_cell   # W/K
        if g_boundary <= 0:
            raise ThermalSolverError(
                "Total boundary conductance is non-positive, so the package has no path "
                "to ambient and no finite time constant exists. Check the bottom BC."
            )
        return c_total / g_boundary

    def steady_state_fraction(self, elapsed_s: float) -> float:
        """Fraction of the steady-state rise a first-order system has covered
        after `elapsed_s`: 1 - exp(-t/tau).

        A single-pole estimate. The real system has a spectrum of modes (fast
        per-layer diffusion, lateral spreading, the global convective pole),
        so this is indicative rather than exact -- but it is the right order
        of magnitude and it is vastly better than the implicit assumption
        that any finite run has converged.
        """
        if elapsed_s <= 0:
            return 0.0
        tau = self.estimate_time_constant()
        return float(1.0 - np.exp(-elapsed_s / tau))

    def check_stability(self, dt: float, safety_factor: float = 0.9, raise_on_violation: bool = True) -> bool:
        """
        Validates a proposed timestep dt against the automated CFL bound.
        Returns True if stable. If raise_on_violation is True (default),
        raises ThermalSolverError on violation; otherwise logs a warning
        and returns False.
        """
        dt_max = self.compute_max_dt(safety_factor=safety_factor)
        if dt > dt_max:
            msg = (
                f"Requested dt={dt:.6e}s exceeds the stable bound dt_max={dt_max:.6e}s "
                f"(safety_factor={safety_factor}). The explicit FDM update WILL diverge. "
                f"Reduce dt, refine the timestep automatically via compute_max_dt(), or "
                f"switch to an implicit scheme once available."
            )
            if raise_on_violation:
                raise ThermalSolverError(msg)
            logger.warning(msg)
            return False
        return True

    # ------------------------------------------------------------------
    # Time stepping (Phase 1: explicit forward-Euler)
    # ------------------------------------------------------------------
    def step(self, dt: float, enforce_stability: bool = True, safety_factor: float = 0.9) -> np.ndarray:
        """
        Advances the temperature field by one explicit forward-Euler step
        of size dt (seconds). Raises ThermalSolverError if dt violates the
        automated CFL/diagonal-dominance bound and enforce_stability=True.
        """
        try:
            if dt <= 0:
                raise ThermalSolverError(f"dt must be positive, got {dt}")
            if enforce_stability:
                self.check_stability(dt, safety_factor=safety_factor, raise_on_violation=True)

            div = self._compute_divergence(self.T)
            dT = (dt / self._rho_cp) * (div + self.q_map)
            T_new = self.T + dT

            if not np.all(np.isfinite(T_new)):
                raise ThermalSolverError(
                    "Numerical divergence detected (NaN/Inf in updated temperature field). "
                    "This indicates dt exceeded stability limits or a grid assembly error."
                )

            self.T = T_new
            self.time_s += dt
            self.step_count += 1
            return self.T

        except ThermalSolverError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ThermalSolverError(f"step() failed at t={self.time_s:.4e}s: {exc}") from exc

    def run(
        self,
        total_time_s: float,
        dt: Optional[float] = None,
        safety_factor: float = 0.9,
        record_interval_s: Optional[float] = None,
        max_steps: int = 10_000_000,
        method: str = "explicit",
    ) -> Dict[str, object]:
        """
        Runs the solver forward for total_time_s seconds using either the
        explicit (method="explicit") or implicit Crank-Nicolson
        (method="crank_nicolson") time-stepper.

        Parameters
        ----------
        dt : float, optional
            Fixed timestep. If None:
              - explicit: computed once via compute_max_dt() (CFL-bound).
              - crank_nicolson: defaults to total_time_s / 200 (unconditionally
                stable, so this is a resolution/accuracy choice, not a
                stability requirement). Override freely for finer/coarser
                temporal resolution.
        record_interval_s : float, optional
            If set, snapshots of T and time are appended to the returned
            history at approximately this cadence. If None, only the final
            state is returned.
        max_steps : int
            Hard safety cap to prevent runaway loops from a misconfigured
            (extremely small) dt.
        method : str
            "explicit" or "crank_nicolson".

        Returns
        -------
        dict with keys: 'final_T', 'final_time_s', 'steps_taken',
        'history_T' (list, possibly empty), 'history_time_s' (list), 'method'.
        """
        if total_time_s <= 0:
            raise ThermalSolverError(f"total_time_s must be positive, got {total_time_s}")
        if method not in ("explicit", "crank_nicolson"):
            raise ThermalSolverError(f"method must be 'explicit' or 'crank_nicolson', got '{method}'")

        if dt is None:
            if method == "explicit":
                dt = self.compute_max_dt(safety_factor=safety_factor)
                logger.info("run(): auto-selected dt=%.6e s from compute_max_dt() [explicit]", dt)
            else:
                dt = total_time_s / 200.0
                logger.info("run(): auto-selected dt=%.6e s (total_time_s/200) [crank_nicolson]", dt)
        elif method == "explicit":
            self.check_stability(dt, safety_factor=safety_factor, raise_on_violation=True)
        # crank_nicolson: no stability check needed (unconditionally stable);
        # step_implicit() itself logs a soft warning for accuracy-risk dt.

        n_steps = int(np.ceil(total_time_s / dt))
        if n_steps > max_steps:
            raise ThermalSolverError(
                f"Requested run would take {n_steps} steps (dt={dt:.4e}s over "
                f"{total_time_s:.4e}s), exceeding max_steps={max_steps}. Increase "
                f"max_steps, shorten total_time_s, or use a larger dt "
                f"{'/ switch to method=\"crank_nicolson\"' if method == 'explicit' else ''}."
            )

        history_T: List[np.ndarray] = []
        history_time: List[float] = []
        next_record_t = 0.0

        logger.info("run(): starting %s integration for %d steps (dt=%.4e s, total=%.4e s)",
                    method, n_steps, dt, total_time_s)

        for i in range(n_steps):
            remaining = total_time_s - self.time_s
            step_dt = dt if remaining >= dt else remaining
            if step_dt <= 0:
                break

            if method == "explicit":
                self.step(step_dt, enforce_stability=(step_dt >= dt * 0.999), safety_factor=safety_factor)
            else:
                self.step_implicit(step_dt, method="crank_nicolson")

            if record_interval_s is not None and self.time_s + 1e-15 >= next_record_t:
                history_T.append(self.T.copy())
                history_time.append(self.time_s)
                next_record_t += record_interval_s

        logger.info("run(): completed at t=%.6e s after %d steps (peak T=%.2f K)",
                    self.time_s, self.step_count, float(np.max(self.T)))

        return {
            "final_T": self.T,
            "final_time_s": self.time_s,
            "steps_taken": self.step_count,
            "history_T": history_T,
            "history_time_s": history_time,
            "method": method,
        }

    # ------------------------------------------------------------------
    # Phase 2 hook (architecture preparation only -- not implemented here)
    # ------------------------------------------------------------------
    def step_implicit(self, dt: float, method: str = "crank_nicolson") -> np.ndarray:
        """
        Advances the temperature field by one implicit step of size dt
        (seconds), unconditionally stable with respect to dt (no CFL
        restriction) for the linear anisotropic heat-conduction operator.

        Currently implements Crank-Nicolson (theta=0.5):
            (I - dt/2 * M) T^{n+1} = (I + dt/2 * M) T^n + dt * f
        where M is the sparse conduction operator assembled once in
        __init__ (_build_conduction_operator, reusing the identical
        harmonic-mean face conductivities and Robin/Dirichlet boundary
        coefficients as the explicit path), and f = boundary_forcing +
        Q/(rho*Cp).

        ADI (method="adi") is intentionally NOT implemented: it requires
        Strang/Douglas-style operator splitting into 3 per-axis tridiagonal
        sweeps, which introduces its own splitting-error term and does not
        reuse the single sparse operator M this class already assembles --
        it would need a materially different code path, not an extension of
        this one. Crank-Nicolson was prioritized because it is unconditionally
        stable in full 3D without splitting approximation and reuses 100% of
        the existing operator-assembly machinery.

        The system matrix backend is chosen automatically based on problem
        size (see DIRECT_SOLVE_MAX_UNKNOWNS): direct sparse LU (splu) for
        small/medium grids, or a Jacobi-preconditioned BiCGSTAB iterative
        solve for large grids, where direct 3D sparse LU fill-in becomes
        intractable (see the module-level comment on DIRECT_SOLVE_MAX_UNKNOWNS
        for the measured rationale behind choosing Jacobi over ILU here).
        Either way, the factorization/preconditioner is cached and reused
        across calls with the same dt (the common case inside run()), so
        repeated steps at a fixed dt only pay for a fast solve, not a full
        refactorization.
        """
        if dt <= 0:
            raise ThermalSolverError(f"dt must be positive, got {dt}")
        if method != "crank_nicolson":
            raise NotImplementedError(
                f"step_implicit(method='{method}') is not implemented. Only "
                f"'crank_nicolson' is currently supported; ADI is reserved for "
                f"a future iteration (see docstring for rationale)."
            )

        if dt > 500.0 * self._explicit_dt_reference:
            logger.warning(
                "step_implicit: dt=%.4e s is %.1fx the explicit-CFL reference "
                "(%.4e s). Crank-Nicolson remains unconditionally STABLE at this "
                "dt, but such large steps relative to the diffusive timescale "
                "can produce non-physical damped oscillations (a known CN "
                "accuracy artifact, not a divergence). Inspect results carefully "
                "or reduce dt if the field looks non-monotonic near sharp "
                "gradients.",
                dt, dt / self._explicit_dt_reference, self._explicit_dt_reference,
            )

        try:
            N = self.nx * self.ny * self.nz
            M = self._conduction_operator
            forcing = self._boundary_forcing + self.q_map.reshape(N) / self._rho_cp.reshape(N)
            T_flat = self.T.reshape(N)
            rhs = T_flat + (dt / 2.0) * (M @ T_flat + forcing) + (dt / 2.0) * forcing

            if dt not in self._cn_solver_cache:
                identity = sparse.identity(N, format="csr")
                lhs = (identity - (dt / 2.0) * M).tocsc()
                self._cn_solver_cache.clear()  # only ever cache the single most-recent dt

                if N <= DIRECT_SOLVE_MAX_UNKNOWNS:
                    lu = sparse_linalg.splu(lhs)
                    self._cn_solver_cache[dt] = {"backend": "direct", "lu": lu, "lhs": lhs}
                    logger.info(
                        "step_implicit: factorized DIRECT (splu) LHS for dt=%.6e s (N=%d unknowns)",
                        dt, N,
                    )
                else:
                    diag = lhs.diagonal()
                    if np.any(np.abs(diag) < 1e-300):
                        raise ThermalSolverError(
                            "Zero (or near-zero) diagonal entry in the implicit system matrix; "
                            "cannot build a Jacobi preconditioner. Check for isolated/unassigned "
                            "voxels in the grid."
                        )
                    inv_diag = 1.0 / diag
                    preconditioner = sparse_linalg.LinearOperator(lhs.shape, matvec=lambda x: inv_diag * x)
                    self._cn_solver_cache[dt] = {"backend": "iterative", "preconditioner": preconditioner, "lhs": lhs}
                    logger.info(
                        "step_implicit: N=%d exceeds DIRECT_SOLVE_MAX_UNKNOWNS=%d; using "
                        "Jacobi-preconditioned BiCGSTAB for dt=%.6e s",
                        N, DIRECT_SOLVE_MAX_UNKNOWNS, dt,
                    )

            cache_entry = self._cn_solver_cache[dt]
            if cache_entry["backend"] == "direct":
                T_new_flat = cache_entry["lu"].solve(rhs)
            else:
                T_new_flat, info = sparse_linalg.bicgstab(
                    cache_entry["lhs"], rhs, M=cache_entry["preconditioner"],
                    rtol=ITERATIVE_RTOL, maxiter=ITERATIVE_MAXITER,
                )
                if info != 0:
                    raise ThermalSolverError(
                        f"Iterative (BiCGSTAB) solve failed to converge (info={info}) at "
                        f"t={self.time_s:.4e}s. This may indicate an ill-conditioned system "
                        f"(e.g. extreme material contrast); consider a tighter ILU drop_tol "
                        f"or a different dt."
                    )

            T_new = T_new_flat.reshape(self.shape)

            if not np.all(np.isfinite(T_new)):
                raise ThermalSolverError(
                    "Numerical failure detected (NaN/Inf in updated temperature field) "
                    "during implicit step. This indicates a malformed conduction operator "
                    "or singular system matrix -- check grid assembly, not dt (Crank-"
                    "Nicolson has no CFL restriction on dt itself)."
                )

            self.T = T_new
            self.time_s += dt
            self.step_count += 1
            return self.T

        except ThermalSolverError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ThermalSolverError(f"step_implicit() failed at t={self.time_s:.4e}s: {exc}") from exc

    def _steady_state_spd_system(self) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """
        Builds (and caches) the symmetric positive-definite form of the
        steady-state system.

        _build_conduction_operator assembles M = D^-1 A, where D = diag(rho*cp)
        and A is the symmetric conductance matrix: it writes
        M[a,b] = coeff/rho_cp[a] but M[b,a] = coeff/rho_cp[b], so the two
        differ wherever rho*cp differs across a material interface. Left-
        multiplying by D cancels that per-row scaling on both sides, giving

            S = -D M = -A,   exactly symmetric, positive definite

        (measured: ||M - M^T||/||M|| = 2.45e-3 versus ||S - S^T||/||S|| = 0.0).
        The explicit (S + S^T)/2 symmetrization below removes only
        floating-point round-off, not a real asymmetry, so that CG never sees
        a matrix it would reject.

        Depends only on the static material maps and boundary configuration,
        so it is built once and reused across every q_map.
        """
        if self._steady_spd is None:
            N = self.nx * self.ny * self.nz
            D = sparse.diags(self._rho_cp.reshape(N))
            S = -(D @ self._conduction_operator)
            S = ((S + S.T) * 0.5).tocsr()
            diag = S.diagonal()
            if np.any(diag <= 0):
                raise ThermalSolverError(
                    "Symmetrized steady-state operator has a non-positive diagonal entry, "
                    "so it is not positive definite and conjugate gradients does not apply. "
                    "Check for isolated or unassigned voxels in the grid."
                )
            self._steady_spd = (S, diag)
        return self._steady_spd

    def _verify_steady_state(self, T_flat: np.ndarray, forcing: np.ndarray) -> Dict[str, float]:
        """
        Verifies a computed steady-state field against two independent checks
        and raises ThermalSolverError if either fails. Returns the measured
        values so callers can log or assert on them.

        See the VERIFICATION, NOT TRUST note at module level for why this runs
        unconditionally rather than trusting a convergence flag.

        Check 1 (algebraic): the defining equation is M T = -forcing, so
        ||M T + forcing|| / ||forcing|| must be at machine-precision level.
        Evaluated on M directly -- not on the symmetrized S -- so it verifies
        the system that was actually posed, whichever backend produced T.

        Check 2 (physical): at equilibrium the net rate of change is zero
        everywhere, so summing rho_cp * (dT/dt) over the domain gives total
        generation minus total boundary loss = 0. Expressed in the operator's
        own units, total power in must equal total power out through the
        bottom boundary.
        """
        N = self.nx * self.ny * self.nz
        M = self._conduction_operator

        # --- 1. algebraic residual on the posed system ---
        resid_vec = M @ T_flat + forcing
        forcing_norm = float(np.linalg.norm(forcing))
        resid = float(np.linalg.norm(resid_vec)) / (forcing_norm if forcing_norm > 0 else 1.0)
        if not np.isfinite(resid) or resid > STEADY_RESIDUAL_TOL:
            raise ThermalSolverError(
                f"Steady-state solve failed verification: relative residual "
                f"||M T + f|| / ||f|| = {resid:.3e} exceeds STEADY_RESIDUAL_TOL="
                f"{STEADY_RESIDUAL_TOL:.1e} at N={N:,}. The solver reported success but "
                f"the answer does not satisfy the equation, so it must not be used."
            )

        # --- 2. physical energy balance, independent of the discretization ---
        T_grid = T_flat.reshape(self.shape)
        boundary_conductance = self._bottom_boundary_coeff / self._rho_cp[:, :, 0]
        power_in = float(np.sum(self.q_map / self._rho_cp))
        power_out = float(np.sum(boundary_conductance * (T_grid[:, :, 0] - self.bc.t_ambient_k)))

        # Normalize by the total boundary conductance so the result is an
        # equivalent surface-temperature error in kelvin. A ratio would be
        # undefined at zero power, where both terms vanish -- see the note on
        # STEADY_ENERGY_BALANCE_TOL_K.
        total_conductance = float(np.sum(boundary_conductance))
        if total_conductance <= 0:
            raise ThermalSolverError(
                "Total boundary conductance is non-positive, so no heat can leave the "
                "domain and no steady state exists. Check h_convective and the bottom BC."
            )
        imbalance_k = abs(power_in - power_out) / total_conductance
        if not np.isfinite(imbalance_k) or imbalance_k > STEADY_ENERGY_BALANCE_TOL_K:
            raise ThermalSolverError(
                f"Steady-state solve failed the energy-balance check: generation and "
                f"boundary loss differ by {imbalance_k:.3e} K equivalent surface temperature "
                f"(tolerance {STEADY_ENERGY_BALANCE_TOL_K:.1e} K). At equilibrium all "
                f"generated power must leave through the boundary exactly, independent of "
                f"mesh -- a mismatch indicates a wrongly assembled operator or boundary "
                f"term, not discretization error."
            )

        # --- 3. the peak must clear the convective floor ---
        # At steady state the mean bottom-surface rise is pinned to P/(h*A) by
        # check 2, and the peak is a maximum over a field that includes that
        # surface, so peak_rise >= P/(h*A) necessarily. Cheap to assert, and
        # it catches a whole class of reporting error: a number that is not
        # actually a peak (a temperature DIFFERENCE, say) will fail it. That
        # has happened twice in review, where an "improvement" of ~39 K was
        # read as a peak and appeared to violate a 90 K floor.
        peak_rise = float(np.max(T_grid)) - self.bc.t_ambient_k
        mean_bottom_rise = float(np.mean(T_grid[:, :, 0])) - self.bc.t_ambient_k
        if peak_rise < mean_bottom_rise - STEADY_ENERGY_BALANCE_TOL_K:
            raise ThermalSolverError(
                f"Peak rise {peak_rise:.6f} K is below the mean bottom-surface rise "
                f"{mean_bottom_rise:.6f} K, which is impossible: the peak is a maximum over "
                f"a field that contains that surface. This indicates a corrupted field or a "
                f"mis-assembled operator."
            )

        return {
            "residual": resid,
            "energy_imbalance_k": imbalance_k,
            "peak_rise_k": peak_rise,
            "mean_bottom_rise_k": mean_bottom_rise,
        }

    def solve_steady_state(self, apply_result: bool = True) -> np.ndarray:
        """
        Directly solves for the steady-state (dT/dt = 0) temperature field:

            0 = M @ T + f   =>   M @ T = -f

        reusing the exact same sparse conduction operator M and boundary-
        forcing vector f already assembled in __init__ for the Crank-
        Nicolson stepper. This is a single linear solve, NOT a time
        integration -- it gives the equilibrium field a sufficiently long
        transient run would eventually approach, directly and exactly (up
        to linear-solve tolerance), which is normally the quantity actually
        wanted for hotspot analysis (peak steady-state die temperature)
        rather than any particular elapsed-time snapshot.

        Only well-posed because bottom_bc is always Robin or Dirichlet
        (never pure Neumann everywhere) -- that boundary term is what keeps
        M non-singular. A hypothetical fully-adiabatic domain would leave
        the steady state defined only up to an additive constant and this
        solve would fail or blow up; SolverBCConfig.validate() already
        prevents an all-Neumann configuration from reaching this point.

        BACKENDS (see STEADY-STATE BACKEND at module level for the measured
        rationale):
          - N <= DIRECT_SOLVE_MAX_UNKNOWNS: sparse LU of M, factorized once
            and cached. Repeated calls that change only q_map -- the entire
            inner loop of src/optimizer.py's placement search -- then cost
            one triangular solve each (measured 0.136 s at N=38,400 against
            a 23 s factorization).
          - larger: conjugate gradients on S = -diag(rho_cp) @ M, which is
            exactly symmetric positive definite even though M is not. This
            replaces a Jacobi-preconditioned BiCGSTAB on M that did not
            converge at any package-scale grid size.

        CACHE VALIDITY: both caches depend only on the static material maps
        (kx/ky/kz, rho, cp) and the boundary configuration -- never on T or
        q_map. reset() and load_checkpoint() touch only T/time_s/step_count,
        so they cannot stale them. Assigning directly to solver.kx_map (or
        any other material map) after construction WOULD stale them and is
        not supported; rebuild the solver instead.

        A useful independent check on any result this returns: at steady
        state all generated power must leave through the Robin boundary, so

            mean(T[:, :, 0]) - T_ambient == P / (h * A)

        exactly, independent of the mesh. A deviation indicates a failed
        solve rather than discretization error.

        Parameters
        ----------
        apply_result : bool
            If True (default), sets self.T to the computed steady-state
            field. Does not touch self.time_s/self.step_count -- a steady-
            state solve is not a timestep, and mutating elapsed time to
            represent "infinity" would corrupt any subsequent transient
            stepping from this state.
        """
        try:
            N = self.nx * self.ny * self.nz
            M = self._conduction_operator
            rho_cp_flat = self._rho_cp.reshape(N)
            forcing = self._boundary_forcing + self.q_map.reshape(N) / rho_cp_flat

            if N <= DIRECT_SOLVE_MAX_UNKNOWNS:
                # Direct path. M depends only on the static material maps, so
                # its factorization is computed once and reused for every
                # subsequent q_map (a placement sweep pays one factorization,
                # then a triangular solve per candidate).
                if self._steady_lu is None:
                    self._steady_lu = sparse_linalg.splu(M.tocsc())
                T_steady_flat = self._steady_lu.solve(-forcing)
                backend = "direct (cached LU)"
            else:
                # Iterative path on the SYMMETRIZED system (see STEADY-STATE
                # BACKEND at module level). S = -diag(rho_cp) @ M is exactly
                # symmetric positive definite, so:
                #     M T = -forcing   <=>   S T = diag(rho_cp) @ forcing
                # and conjugate gradients applies. Jacobi-preconditioned
                # BiCGSTAB on the unsymmetrized M does NOT converge here.
                S, diag = self._steady_state_spd_system()
                b = rho_cp_flat * forcing
                inv_diag = 1.0 / diag
                preconditioner = sparse_linalg.LinearOperator(
                    S.shape, matvec=lambda x: inv_diag * x
                )
                T_steady_flat, info = sparse_linalg.cg(
                    S, b, M=preconditioner,
                    rtol=STEADY_CG_RTOL, maxiter=STEADY_CG_MAXITER,
                )
                if info != 0:
                    raise ThermalSolverError(
                        f"Steady-state CG solve failed to converge (info={info}) on the "
                        f"symmetrized SPD system at N={N:,} unknowns. This may indicate an "
                        f"ill-conditioned system (e.g. extreme material contrast or an "
                        f"extreme mesh aspect ratio) or a near-singular operator (check that "
                        f"a Robin/Dirichlet boundary is actually present)."
                    )
                backend = "iterative (CG on symmetrized SPD system)"

            T_steady = T_steady_flat.reshape(self.shape)
            if not np.all(np.isfinite(T_steady)):
                raise ThermalSolverError(
                    "solve_steady_state() produced non-finite temperatures -- likely a "
                    "singular or near-singular operator (check that a Robin/Dirichlet "
                    "boundary is actually present and h_convective/conductivities are "
                    "physical, nonzero values)."
                )

            # Verify rather than trust: both backends are checked against the
            # posed equation AND against energy conservation, and either
            # failing raises. See VERIFICATION, NOT TRUST at module level.
            checks = self._verify_steady_state(T_steady_flat, forcing)

            logger.info(
                "solve_steady_state: solved via %s backend (N=%d unknowns), peak T=%.2f K "
                "[verified: residual=%.2e, energy imbalance=%.2e K]",
                backend, N, float(np.max(T_steady)),
                checks["residual"], checks["energy_imbalance_k"],
            )

            if apply_result:
                self.T = T_steady
            return T_steady

        except ThermalSolverError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ThermalSolverError(f"solve_steady_state() failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Diagnostics / analytics
    # ------------------------------------------------------------------
    def get_temperature_field_celsius(self) -> np.ndarray:
        return self.T - 273.15

    def get_hotspots(self, threshold_c: float = 85.0) -> Dict[str, object]:
        """
        Scans the current T[x,y,z] field (per the EDA hotspot-detection
        requirement) and returns coordinates + values of cells exceeding
        threshold_c.
        """
        T_c = self.get_temperature_field_celsius()
        mask = T_c > threshold_c
        coords = np.argwhere(mask)
        values = T_c[mask]
        return {
            "count": int(coords.shape[0]),
            "coordinates_ijk": coords,
            "temperatures_c": values,
            "peak_temperature_c": float(np.max(T_c)),
            "peak_location_ijk": tuple(int(v) for v in np.unravel_index(np.argmax(T_c), T_c.shape)),
        }

    def reset(self, initial_temperature_k: Optional[np.ndarray] = None) -> None:
        """Resets time/step counters and temperature field (e.g. for a fresh
        transient run with a different power map)."""
        if initial_temperature_k is None:
            self.T = np.full(self.shape, self.bc.t_ambient_k, dtype=np.float64)
        else:
            init = np.asarray(initial_temperature_k, dtype=np.float64)
            if init.shape != self.shape:
                raise ThermalSolverError(
                    f"initial_temperature_k shape {init.shape} does not match grid shape {self.shape}"
                )
            self.T = init.copy()
        self.time_s = 0.0
        self.step_count = 0
        logger.info("ThermalSolver reset to t=0, T=ambient/custom initial field.")

    def save_checkpoint(self, filepath: str) -> None:
        """Persists solver state (T, time, step_count) for resuming a run."""
        try:
            os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
            np.savez_compressed(filepath, T=self.T, time_s=self.time_s, step_count=self.step_count)
            logger.info("Saved solver checkpoint to %s (t=%.4e s)", filepath, self.time_s)
        except OSError as exc:
            raise ThermalSolverError(f"Failed to save checkpoint to '{filepath}': {exc}") from exc

    def load_checkpoint(self, filepath: str) -> None:
        """Restores solver state from a checkpoint saved by save_checkpoint()."""
        try:
            data = np.load(filepath)
            T = data["T"]
            if T.shape != self.shape:
                raise ThermalSolverError(
                    f"Checkpoint T shape {T.shape} does not match solver grid shape {self.shape}"
                )
            self.T = T.astype(np.float64)
            self.time_s = float(data["time_s"])
            self.step_count = int(data["step_count"])
            logger.info("Loaded solver checkpoint from %s (t=%.4e s, step=%d)",
                        filepath, self.time_s, self.step_count)
        except (OSError, KeyError, ValueError) as exc:
            raise ThermalSolverError(f"Failed to load checkpoint from '{filepath}': {exc}") from exc


if __name__ == "__main__":
    # Minimal smoke test using a small synthetic grid (mirrors the
    # GridBuilder.__main__ example) to exercise CFL check + explicit stepping.
    nx, ny, nz = 20, 20, 15
    shape = (nx, ny, nz)

    kx = np.full(shape, 149.0)
    ky = np.full(shape, 149.0)
    kz = np.full(shape, 149.0)
    # Inject a high-k Cu layer at the bottom two z-slices to confirm the
    # automated stability audit correctly identifies it as the binding case.
    kx[:, :, :2] = ky[:, :, :2] = kz[:, :, :2] = 400.0

    rho = np.full(shape, 2330.0)
    rho[:, :, :2] = 8960.0
    cp = np.full(shape, 712.0)
    cp[:, :, :2] = 385.0

    q = np.zeros(shape)
    q[8:12, 8:12, 10:13] = 5.0e11  # localized hotspot source, W/m^3

    boundary_mask = np.zeros(shape, dtype=np.uint8)

    grid_arrays = {
        "kx_map": kx, "ky_map": ky, "kz_map": kz,
        "rho_map": rho, "cp_map": cp, "q_map": q,
        "boundary_mask": boundary_mask,
    }
    bc_config = SolverBCConfig(dx=25e-6, dy=25e-6, dz=10e-6,
                                t_ambient_k=298.15, h_convective=2500.0,
                                bottom_bc="robin")

    solver = ThermalSolver(grid_arrays, bc_config)
    dt_max = solver.compute_max_dt(safety_factor=0.9)
    print(f"Automated CFL bound: dt_max = {dt_max:.4e} s")

    result = solver.run(total_time_s=200 * dt_max, record_interval_s=50 * dt_max)
    hotspots = solver.get_hotspots(threshold_c=85.0)
    print(f"Final t={result['final_time_s']:.4e}s over {result['steps_taken']} steps")
    print(f"Peak temperature: {hotspots['peak_temperature_c']:.2f} C at {hotspots['peak_location_ijk']}")
    print(f"Cells > 85C: {hotspots['count']}")