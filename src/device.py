"""
device.py
ThermalForge :: what the silicon draws, as a function of how hot it is.

The solver upstream of this file takes watts and returns temperatures. This
file goes one step earlier: it computes the watts from a device operating
point, so a design can state a bias rather than a power budget, and so the
power can move when the junction does.

Two models, because two different questions are asked of a transistor:

  Mosfet      -- the level-1 (square-law) MOSFET, Id(Vgs, Vds, T). Used when
                 the question is "what does this bias draw", and the answer
                 has to carry the two temperature dependences that decide
                 whether the feedback is positive or negative.
  LinearSwitch -- a conducting switch, Rds(T) = Rds0 (1 + alpha (T - Tref)),
                 dissipating I^2 R. Used when the device is on and the
                 question is only how its resistance drifts with heat. This
                 is the simplified feedback path: one number, one slope.

WHAT THIS IS NOT. Level 1 is a textbook long-channel model. It has no
subthreshold conduction (below threshold it returns exactly zero, not a
decade-per-80-mV tail), no velocity saturation, no DIBL, no body effect, no
gate leakage, and no self-heating inside the device itself -- the junction
temperature it is handed is the cell average the mesh resolves, which the
package solver already labels as such. Its parameters are whatever the caller
supplies; nothing here is extracted from measured silicon, and nothing here
has been compared against a measured I-V curve. It is a power *source* for the
thermal problem, stated in physics rather than typed in as a constant, and it
carries no validation claim of its own.

What it does carry is verification: the square law at the reference
temperature, continuity across the triode/saturation boundary, and the
zero-temperature-coefficient crossing that falls out of the two competing
dependences (tests/test_device.py).

The two dependences, and why the sign of the feedback is not obvious:

  threshold  Vth(T) = Vth0 + dVth/dT (T - Tref), dVth/dT about -2 mV/K.
             Hotter silicon turns on EARLIER, so this pushes the current UP.
  mobility   mu(T) = mu0 (T/Tref)^-m, m about 1.5 for phonon scattering.
             Hotter silicon conducts WORSE, so this pulls the current DOWN.

Near threshold the first wins and the current rises with temperature, which is
how a hotspot runs away. Well above threshold the second wins and the current
falls, which is why a digital part biased hard is self-limiting. The crossing
between them is the zero-temperature-coefficient point, and it is a real
property of silicon rather than an artifact of this file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Absolute zero, for the mobility ratio, which is a ratio of absolute
#: temperatures and wrong if taken in Celsius.
KELVIN = 273.15

#: Hotter than any silicon survives. Past this the model is extrapolating into
#: a regime it does not describe, so it refuses rather than returning a number.
MAX_JUNCTION_C = 500.0


class DeviceError(Exception):
    """A bias or a parameter the model cannot stand behind."""


def _check_temperature(t_c: float, where: str) -> float:
    t = float(t_c)
    if not math.isfinite(t):
        raise DeviceError(f"{where}: junction temperature is not a finite number")
    if t <= -KELVIN:
        raise DeviceError(f"{where}: junction temperature {t:.1f} degC is at or below "
                          "absolute zero")
    if t > MAX_JUNCTION_C:
        raise DeviceError(f"{where}: junction temperature {t:.1f} degC is past "
                          f"{MAX_JUNCTION_C:.0f} degC, where this model is "
                          "extrapolating rather than describing")
    return t


def _check_bias(value: float, name: str) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise DeviceError(f"{name} is not a finite number")
    if v < 0.0:
        raise DeviceError(f"{name} must be >= 0 for this n-channel model, got {v}")
    return v


@dataclass(frozen=True)
class Mosfet:
    """A level-1 n-channel enhancement MOSFET, with temperature.

    Parameters are the caller's to supply and the caller's to defend:

      vth0_v        threshold voltage at t_ref_c
      kp_a_v2       process transconductance mu0 * Cox, in A/V^2
      w_over_l      gate width over gate length, dimensionless
      lambda_1_v    channel-length modulation, in 1/V. Applied in BOTH
                    regions, which is what keeps Id continuous at the
                    triode/saturation boundary; level 1 applied in
                    saturation only has a step there.
      dvth_dt_v_k   threshold drift, in V/K. Negative for silicon.
      mu_exponent   mobility exponent m in (T/Tref)^-m. About 1.5.
    """

    vth0_v: float
    kp_a_v2: float
    w_over_l: float
    lambda_1_v: float = 0.0
    t_ref_c: float = 25.0
    dvth_dt_v_k: float = -2.0e-3
    mu_exponent: float = 1.5

    def __post_init__(self) -> None:
        if not math.isfinite(self.vth0_v) or self.vth0_v <= 0.0:
            raise DeviceError(f"vth0_v must be > 0 for an enhancement device, "
                              f"got {self.vth0_v}")
        if not math.isfinite(self.kp_a_v2) or self.kp_a_v2 <= 0.0:
            raise DeviceError(f"kp_a_v2 must be > 0, got {self.kp_a_v2}")
        if not math.isfinite(self.w_over_l) or self.w_over_l <= 0.0:
            raise DeviceError(f"w_over_l must be > 0, got {self.w_over_l}")
        if not math.isfinite(self.lambda_1_v) or self.lambda_1_v < 0.0:
            raise DeviceError(f"lambda_1_v must be >= 0, got {self.lambda_1_v}")
        if not math.isfinite(self.mu_exponent) or self.mu_exponent < 0.0:
            raise DeviceError(f"mu_exponent must be >= 0, got {self.mu_exponent}")
        if not math.isfinite(self.dvth_dt_v_k):
            raise DeviceError("dvth_dt_v_k must be a finite number")
        _check_temperature(self.t_ref_c, "t_ref_c")

    # -- the two temperature dependences, separately, so each can be checked --

    def vth_v(self, t_c: float) -> float:
        """Threshold voltage at this junction temperature."""
        t = _check_temperature(t_c, "vth_v")
        return self.vth0_v + self.dvth_dt_v_k * (t - self.t_ref_c)

    def beta_a_v2(self, t_c: float) -> float:
        """Transconductance parameter kp * (T/Tref)^-m * W/L, in A/V^2."""
        t = _check_temperature(t_c, "beta_a_v2")
        ratio = (t + KELVIN) / (self.t_ref_c + KELVIN)
        return self.kp_a_v2 * ratio ** (-self.mu_exponent) * self.w_over_l

    # -- the current, and what follows from it --

    def id_a(self, vgs_v: float, vds_v: float, t_c: float) -> float:
        """Drain current in amperes. Zero below threshold: level 1 has no
        subthreshold tail, and this does not pretend to one."""
        vgs = _check_bias(vgs_v, "vgs_v")
        vds = _check_bias(vds_v, "vds_v")
        vov = vgs - self.vth_v(t_c)
        if vov <= 0.0:
            return 0.0
        beta = self.beta_a_v2(t_c)
        clm = 1.0 + self.lambda_1_v * vds
        if vds < vov:                                   # triode
            return beta * (vov * vds - 0.5 * vds * vds) * clm
        return 0.5 * beta * vov * vov * clm             # saturation

    def power_w(self, vgs_v: float, vds_v: float, t_c: float) -> float:
        """P = Vds * Id. The gate draws no current in this model, so the
        drain path is the whole of it."""
        return float(vds_v) * self.id_a(vgs_v, vds_v, t_c)

    def rds_on_ohm(self, vgs_v: float, vds_v: float, t_c: float) -> float:
        """Vds / Id at this bias. Infinite in cutoff, where no current flows.

        Note this is the operating-point resistance at a finite Vds, not the
        small-signal slope at the origin; the two agree as Vds -> 0.
        """
        current = self.id_a(vgs_v, vds_v, t_c)
        if current <= 0.0:
            return math.inf
        return float(vds_v) / current

    def saturates_at_v(self, vgs_v: float, t_c: float) -> float:
        """The Vds above which this bias is in saturation, i.e. Vgs - Vth(T).
        Zero or negative means the device is in cutoff."""
        return _check_bias(vgs_v, "vgs_v") - self.vth_v(t_c)


@dataclass(frozen=True)
class LinearSwitch:
    """A conducting switch whose resistance drifts linearly with temperature.

    Rds(T) = rds0_ohm * (1 + alpha_per_k * (T - t_ref_c)), dissipating I^2 R.

    This is the simplified feedback path: positive alpha means hotter is more
    resistive is more power is hotter, which is the loop the coupled solve has
    to converge or refuse. Silicon power devices run about +0.4 to +0.8 %/K.
    """

    rds0_ohm: float
    alpha_per_k: float = 5.0e-3
    t_ref_c: float = 25.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.rds0_ohm) or self.rds0_ohm <= 0.0:
            raise DeviceError(f"rds0_ohm must be > 0, got {self.rds0_ohm}")
        if not math.isfinite(self.alpha_per_k):
            raise DeviceError("alpha_per_k must be a finite number")
        _check_temperature(self.t_ref_c, "t_ref_c")

    def rds_ohm(self, t_c: float) -> float:
        """Resistance at this junction temperature. Refuses to go negative:
        a linear law extrapolated far enough does, and a negative resistance
        is not a number this model may return."""
        t = _check_temperature(t_c, "rds_ohm")
        r = self.rds0_ohm * (1.0 + self.alpha_per_k * (t - self.t_ref_c))
        if r <= 0.0:
            raise DeviceError(
                f"rds_ohm: the linear law gives {r:.4g} ohm at {t:.1f} degC, which is "
                "not physical -- alpha_per_k and the temperature are outside the range "
                "a linear drift describes")
        return r

    def power_w(self, current_a: float, t_c: float) -> float:
        """I^2 R at this junction temperature."""
        i = _check_bias(current_a, "current_a")
        return i * i * self.rds_ohm(t_c)

    def vds_v(self, current_a: float, t_c: float) -> float:
        """The drop across the switch at this current and temperature."""
        return _check_bias(current_a, "current_a") * self.rds_ohm(t_c)
