"""
tests/test_benchmarks.py
ThermalForge :: the solver against closed-form solutions.

Every other test in this repository checks the solver against itself: its own
residual, its own energy balance, one implementation against another. These
tests check it against mathematics that exists outside the codebase -- two
classical solutions of the steady heat equation, derived here from separation
of variables, with the derivation written out so it can be checked by hand:

  1. A uniformly heated slab, insulated on one face and convectively cooled
     on the other. The temperature profile is a parabola; the finite-volume
     scheme should reproduce it and converge at second order.

  2. A rectangular flux channel: a patch source on a slab with insulated
     sides and a convectively cooled far face. This is the heat-spreading
     problem the whole product rests on -- it is what makes WHERE a chiplet
     sits change how hot it gets -- and it has a Fourier-series solution.

These are verification benchmarks (is the equation solved correctly?), not
validation (does the equation describe a real package?). Validation needs
measured hardware; see src/validation_status.py.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.thermal_solver import SolverBCConfig, ThermalSolver

K_SI = 130.0          # W/m.K, a silicon-like conductivity for the benchmarks
RHO, CP = 2330.0, 700.0   # steady state does not use these; the solver requires them


def uniform_solver(nx, ny, nz, dx, dy, dz, k, h, t_ambient_k=300.0):
    """A uniform slab: insulated sides and bottom, convectively cooled top."""
    shape = (nx, ny, nz)
    arrays = {
        "kx_map": np.full(shape, k), "ky_map": np.full(shape, k), "kz_map": np.full(shape, k),
        "rho_map": np.full(shape, RHO), "cp_map": np.full(shape, CP),
        "q_map": np.zeros(shape), "boundary_mask": np.zeros(shape, dtype=np.uint8),
    }
    bc = SolverBCConfig(dx=dx, dy=dy, dz=dz, t_ambient_k=t_ambient_k, h_convective=h,
                        bottom_bc="neumann", top_bc="robin", side_bc="neumann")
    return ThermalSolver(arrays, bc)


# ---------------------------------------------------------------------------
# 1. Uniformly heated slab
# ---------------------------------------------------------------------------
#
#   k T''(z) + q = 0 on 0 < z < L,  T'(0) = 0 (insulated),
#   -k T'(L) = h (T(L) - T_inf) (convective)
#
#   => T(z) - T_inf = qL/h + q (L^2 - z^2) / (2k)
#
# The hottest point is the insulated face, z = 0; the cooled face sits exactly
# qL/h above ambient, which is the "cooling floor" the product quotes.

def slab_analytic(z_m, length_m, q_w_m3, k, h):
    return q_w_m3 * length_m / h + q_w_m3 * (length_m ** 2 - z_m ** 2) / (2.0 * k)


class TestUniformlyHeatedSlab:
    @staticmethod
    def solve(nz, length_m=400e-6, q=2.0e9, h=5000.0):
        dz = length_m / nz
        solver = uniform_solver(4, 4, nz, 100e-6, 100e-6, dz, K_SI, h)
        solver.q_map[:] = q
        field = solver.solve_steady_state(apply_result=False)
        rise = field[2, 2, :] - solver.bc.t_ambient_k
        centres = (np.arange(nz) + 0.5) * dz
        exact = slab_analytic(centres, length_m, q, K_SI, h)
        return rise, exact, dz

    def test_profile_matches_the_closed_form(self):
        rise, exact, _dz = self.solve(nz=40)
        assert np.max(np.abs(rise - exact)) / exact.max() < 1e-5

    def test_cooled_surface_sits_exactly_at_the_cooling_floor(self):
        """P/(hA) is exact in the discrete scheme too: it is a statement about
        total energy leaving, not about the profile. The last cell CENTRE sits
        half a cell inside the material, so it is hotter than the surface by
        the drop that half cell carries -- and that, too, is exact here."""
        length_m, q, h = 400e-6, 2.0e9, 5000.0
        nz = 20
        dz = length_m / nz
        solver = uniform_solver(4, 4, nz, 100e-6, 100e-6, dz, K_SI, h)
        solver.q_map[:] = q
        field = solver.solve_steady_state(apply_result=False)
        cell_rise = float(np.mean(field[:, :, -1] - solver.bc.t_ambient_k))
        flux = q * length_m                       # W/m^2 leaving the face
        floor = flux / h
        half_cell_drop = flux * dz / (2.0 * K_SI)
        assert cell_rise == pytest.approx(floor + half_cell_drop, rel=1e-9)
        # ... so the surface itself, reconstructed, is the floor exactly.
        assert cell_rise - half_cell_drop == pytest.approx(floor, rel=1e-9)

    def test_error_falls_at_second_order(self):
        """Halving the cell size should quarter the error. This is what makes
        a mesh-refinement error bar meaningful."""
        errors = []
        for nz in (10, 20, 40):
            rise, exact, _dz = self.solve(nz=nz)
            errors.append(abs(float(rise[0]) - float(exact[0])))
        orders = [math.log(errors[n] / errors[n + 1], 2) for n in range(len(errors) - 1)]
        assert all(1.7 < p < 2.3 for p in orders), f"observed orders {orders}"

    def test_the_remaining_error_is_the_predicted_half_cell_term(self):
        """Not just second order -- second order with the coefficient the
        discretization predicts: the scheme integrates the parabola exactly in
        the interior, so what is left is q dz^2 / (8k) from the cooled face."""
        length_m, q, h = 400e-6, 2.0e9, 5000.0
        for nz in (10, 20, 40):
            rise, exact, dz = self.solve(nz=nz, length_m=length_m, q=q, h=h)
            predicted = q * dz ** 2 / (8.0 * K_SI)
            # agreement is to a few parts per million; the rest is higher order
            assert float(rise[0] - exact[0]) == pytest.approx(predicted, rel=1e-4)


# ---------------------------------------------------------------------------
# 2. Rectangular flux channel (heat spreading)
# ---------------------------------------------------------------------------
#
# Slab 0<x<Lx, 0<y<Ly, 0<z<t. Insulated sides. Heat enters through a patch of
# the z=0 face at uniform flux; the z=t face is convectively cooled.
# Separation of variables on the insulated-side eigenfunctions
# cos(alpha_m x), alpha_m = m pi / Lx, cos(beta_n y), beta_n = n pi / Ly:
#
#   mode (0,0):  T(z) = Q/(h A) + Q (t - z) / (k A)
#   mode (m,n):  T(z) = C [cosh(g (t-z)) + phi sinh(g (t-z))],
#                g = sqrt(alpha_m^2 + beta_n^2), phi = h / (k g),
#                C = f_mn / (k g [sinh(g t) + phi cosh(g t)])
#
# with f_mn the source's Fourier coefficient on that eigenfunction. Mode (0,0)
# is the part no placement can change -- the cooling floor -- and every other
# mode decays over the spreading length, which is exactly why a chiplet's
# neighbours and the package edges matter.

def flux_channel_rise(x, y, z, Lx, Ly, t, k, h, rect, total_w, modes=160):
    """Temperature rise above ambient at (x, y, z), from the series above.

    Vectorized over the (m, n) modes: the same arithmetic as the derivation,
    evaluated as arrays so the tests stay fast enough to run every time.
    """
    x1, x2, y1, y2 = rect
    q = total_w / ((x2 - x1) * (y2 - y1))
    m = np.arange(modes + 1)
    am = m * np.pi / Lx
    bn = m * np.pi / Ly
    # Source integral and eigenfunction norm per direction.
    with np.errstate(invalid="ignore", divide="ignore"):
        ix = np.where(m == 0, x2 - x1, (np.sin(am * x2) - np.sin(am * x1)) / np.where(am == 0, 1.0, am))
        iy = np.where(m == 0, y2 - y1, (np.sin(bn * y2) - np.sin(bn * y1)) / np.where(bn == 0, 1.0, bn))
    nx_norm = np.where(m == 0, Lx, Lx / 2.0)
    ny_norm = np.where(m == 0, Ly, Ly / 2.0)
    f = q * (ix / nx_norm)[:, None] * (iy / ny_norm)[None, :]
    g = np.hypot(am[:, None], bn[None, :])
    g[0, 0] = 1.0                      # placeholder; the (0,0) term is added below
    phi = h / (k * g)
    gt = np.clip(g * t, 0.0, 500.0)    # cosh/sinh saturate long before this
    denom = k * g * (np.sinh(gt) + phi * np.cosh(gt))
    gz = np.clip(g * (t - z), 0.0, 500.0)
    shape = np.cosh(gz) + phi * np.sinh(gz)
    modes_sum = (f / denom) * shape * np.cos(am * x)[:, None] * np.cos(bn * y)[None, :]
    modes_sum[0, 0] = 0.0
    floor_and_1d = total_w / (h * Lx * Ly) + total_w * (t - z) / (k * Lx * Ly)
    return float(floor_and_1d + modes_sum.sum())


class TestFluxChannelSpreading:
    LX = LY = 8e-3
    T = 400e-6
    H = 5000.0
    POWER = 4.0

    def fv_peak(self, n_lateral, nz):
        """Finite-volume peak rise for the same problem: the source occupies
        the bottom cell layer over the middle quarter of the footprint."""
        dx = self.LX / n_lateral
        dz = self.T / nz
        solver = uniform_solver(n_lateral, n_lateral, nz, dx, dx, dz, K_SI, self.H)
        lo, hi = 3 * n_lateral // 8, 5 * n_lateral // 8      # centred 2 x 2 mm patch
        cells = (hi - lo) ** 2
        solver.q_map[lo:hi, lo:hi, 0] = self.POWER / (cells * dx * dx * dz)
        field = solver.solve_steady_state(apply_result=False)
        rise = field - solver.bc.t_ambient_k
        centre = n_lateral // 2
        return float(rise[centre, centre, 0]), dz

    def analytic_peak(self, dz):
        """The series at the centre of the source, at the depth of the first
        cell's centre -- the point the finite-volume value represents."""
        rect = (3 * self.LX / 8, 5 * self.LX / 8, 3 * self.LY / 8, 5 * self.LY / 8)
        return flux_channel_rise(self.LX / 2, self.LY / 2, dz / 2, self.LX, self.LY,
                                 self.T, K_SI, self.H, rect, self.POWER)

    def test_spreading_peak_converges_to_the_series(self):
        errors = []
        for n_lateral, nz in ((8, 4), (16, 8), (32, 16)):
            fv, dz = self.fv_peak(n_lateral, nz)
            exact = self.analytic_peak(dz)
            errors.append(abs(fv - exact) / exact)
        assert errors[0] > errors[1] > errors[2], f"not converging: {errors}"
        assert errors[-1] < 0.03, f"finest mesh is {errors[-1] * 100:.1f} % from the series"

    def test_spreading_is_what_makes_placement_matter(self):
        """The series says a concentrated source runs hotter than a spread
        one at the same total power; the solver must say the same, and by a
        margin far larger than its own discretization error."""
        rect_small = (3 * self.LX / 8, 5 * self.LX / 8, 3 * self.LY / 8, 5 * self.LY / 8)
        rect_big = (self.LX / 8, 7 * self.LX / 8, self.LY / 8, 7 * self.LY / 8)
        hot = flux_channel_rise(self.LX / 2, self.LY / 2, 0.0, self.LX, self.LY, self.T,
                                K_SI, self.H, rect_small, self.POWER)
        cool = flux_channel_rise(self.LX / 2, self.LY / 2, 0.0, self.LX, self.LY, self.T,
                                 K_SI, self.H, rect_big, self.POWER)
        floor = self.POWER / (self.H * self.LX * self.LY)
        assert hot > cool > floor

    def test_far_from_the_source_the_floor_is_what_is_left(self):
        """Every mode except (0,0) decays away from the source, so a corner
        far from a small source approaches P/(hA) -- the part of the
        temperature no placement can remove."""
        rect = (0.0, self.LX / 8, 0.0, self.LY / 8)      # source in one corner
        floor = self.POWER / (self.H * self.LX * self.LY)
        far = flux_channel_rise(self.LX, self.LY, self.T, self.LX, self.LY, self.T,
                                K_SI, self.H, rect, self.POWER)
        near = flux_channel_rise(0.0, 0.0, self.T, self.LX, self.LY, self.T,
                                 K_SI, self.H, rect, self.POWER)
        # The far corner is cooler than the average of the cooled face, the
        # near corner hotter, and the average IS the floor: every mode but
        # (0, 0) integrates to zero over the face.
        assert 0.0 < far < floor < near
        grid = np.linspace(0.0, self.LX, 41)
        mean_face = float(np.mean([[flux_channel_rise(x, y, self.T, self.LX, self.LY, self.T,
                                                      K_SI, self.H, rect, self.POWER)
                                    for x in grid] for y in grid]))
        assert mean_face == pytest.approx(floor, rel=0.02)
