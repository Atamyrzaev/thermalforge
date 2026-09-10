"""
composite_materials.py
RISC3D-AI :: effective conductivities of homogenized package layers.

SOURCE
    X. Rao, H. Liu, S. Wang, J. Song, C. Jin, C. Xiao, "Thermal modelling and
    analysis of 3-D integrated circuits with irregular structure", Thermal
    Science 27(5B), 4193-4207 (2023). doi:10.2298/TSCI220805061R
    (open access, CC BY-NC-ND 4.0)

WHAT IS TAKEN FROM IT
  - Constituent conductivities: Table 2 (TSCI_K below).
  - Microbump geometry and metallurgy: Table 1 -- 20 um diameter, 23 um tall,
    Cu 5 / Ni 3 / Ni3Sn4 1.5 / Sn2.5Ag 4 um. The four thicknesses sum to the
    stated 23 um height only as the symmetric stack Cu/Ni/Ni3Sn4/SnAg/
    Ni3Sn4/Ni/Cu (2 x (5 + 3 + 1.5) + 4 = 23), so that is the laminate used
    here; the table lists each layer once.
  - Eqs. (10)-(11): a laminate of continuous layers stacked in z -- series
    out of plane, 1/k_z = sum(v_i/k_i); parallel in plane, k_xy =
    sum(k_i v_i). Applied, as in the paper, to the metal stack INSIDE one
    bump.
  - Eq. (2): the parallel rule k_z = sum(v_i k_i) for the out-of-plane
    conductivity of a pillar cell (the paper's TSV cell). Applied here to
    bump pillars in underfill. It reproduces the paper's TSV value:
    175.20 W/mK against the stated 175.2 (see tsv_cell_k).

WHAT IS NOT
  - The paper's in-plane pillar-cell model, eqs. (3)-(7). Read from the
    published PDF, an implementation gives 91.83 W/mK on the paper's own TSV
    check cell (r_Cu 8, r_SiO2 9, pitch 40 um) against the stated 116.74, so
    that reading is not trustworthy and it is not used. In its place:
    Maxwell-Garnett for aligned cylinders, with a coated inclusion first
    homogenized by Hashin's composite-cylinder formula. On the same cell it
    gives 114.37 W/mK against the paper's FEM 115.3 (-0.81%; the paper's own
    analytical value is +1.25%).
  - Eqs. (10)-(11) applied to the bump/underfill LAYER by volume fraction.
    That treats side-by-side pillars as stacked sheets; at 40 um pitch it
    gives k_z = 0.25 W/mK, barely above the underfill, where the parallel
    rule gives 13.15. Heat in a pillar array runs up the pillars.

ASSUMPTIONS NOT IN THE PAPER
  - Bump pitch 40 um: the paper's TSV pitch, and typical of CoWoS
    microbumps. The paper's text does not state a bump pitch. k_z of the
    layer at 30 / 40 / 55 um pitch: 23.2 / 13.1 / 7.0 W/mK.
  - The bumps span the whole 40 um composite layer of config_loader's
    StackSection with the Table 1 fractions (Table 1's bump is 23 um tall).
    Fractions, not height, set the conductivities.

EFFECT ON THE PIPELINE (measured; default 40x40x12 grid, sink on top)
  Old uncited values (silicon 149; composite kx = ky 2.0, kz 20.0) -> these:
      passive-lowpower   79.8178 -> 79.7467 degC   (-0.071 K)
      cowos-coldplate    84.1108 -> 84.0864 degC   (-0.024 K)
  Split by material, passive-lowpower: silicon alone -0.126 K, composite
  alone +0.055 K (cowos-coldplate: -0.021 K and -0.003 K). The composite
  lost 35% of its kz and 85% of its kxy and moved the peak by 0.055 K:
  with the sink on the die backside it is no longer between the die and
  the sink, only on the interposer's side path.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

CITATION = (
    "Rao et al., Thermal Science 27(5B), 4193-4207 (2023), "
    "doi:10.2298/TSCI220805061R"
)

# Rao et al. 2023, Table 2 [W/(m K)].
TSCI_K: Dict[str, float] = {
    "copper": 390.0,
    "silicon": 150.0,
    "sio2": 1.5,
    "nickel": 60.7,
    "ni3sn4": 19.6,
    "sn2.5ag": 57.0,
    "underfill": 0.2,
}

# Rao et al. 2023, Table 1: microbump metal stack, bottom to top [um].
MICROBUMP_LAMINATE_UM: Tuple[Tuple[str, float], ...] = (
    ("copper", 5.0), ("nickel", 3.0), ("ni3sn4", 1.5), ("sn2.5ag", 4.0),
    ("ni3sn4", 1.5), ("nickel", 3.0), ("copper", 5.0),
)
MICROBUMP_DIAMETER_UM = 20.0
MICROBUMP_HEIGHT_UM = 23.0

# Assumed, not from the paper -- see the module docstring.
MICROBUMP_PITCH_UM = 40.0


def laminate_k(layers: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    """Eqs. (10)-(11). `layers` is (k, thickness) per continuous layer
    stacked in z. Returns (k_z, k_xy): series out of plane, parallel in
    plane. Only thickness RATIOS matter."""
    total = sum(t for _, t in layers)
    if total <= 0 or any(k <= 0 or t < 0 for k, t in layers):
        raise ValueError("laminate needs positive conductivities and non-negative "
                         "thicknesses with a positive total")
    kz = total / sum(t / k for k, t in layers)
    kxy = sum(k * t for k, t in layers) / total
    return kz, kxy


def pillar_cell_kz(parts: Sequence[Tuple[float, float]]) -> float:
    """Eq. (2): parallel rule sum(v_i k_i) for straight pillars that span the
    layer. `parts` is (volume_fraction, k); the fractions must sum to 1."""
    total = sum(v for v, _ in parts)
    if abs(total - 1.0) > 1e-9 or any(v < 0 for v, _ in parts):
        raise ValueError(f"volume fractions must be non-negative and sum to 1, got {total!r}")
    return sum(v * k for v, k in parts)


def coated_cylinder_k(k_core: float, k_shell: float, r_core: float, r_shell: float) -> float:
    """Transverse conductivity of a core/shell cylinder (exact for Hashin's
    composite-cylinder assemblage):

        k = k_s ((k_c + k_s) + (k_c - k_s) a) / ((k_c + k_s) - (k_c - k_s) a),
        a = (r_c / r_s)^2
    """
    if not 0 < r_core <= r_shell:
        raise ValueError("need 0 < r_core <= r_shell")
    a = (r_core / r_shell) ** 2
    return k_shell * ((k_core + k_shell) + (k_core - k_shell) * a) / \
        ((k_core + k_shell) - (k_core - k_shell) * a)


def maxwell_garnett_2d(k_inclusion: float, k_matrix: float, fraction: float) -> float:
    """Transverse conductivity of aligned circular cylinders at area fraction
    `fraction` in a matrix (2-D Maxwell-Garnett):

        k = k_m (k_i + k_m + f (k_i - k_m)) / (k_i + k_m - f (k_i - k_m))

    Limited to f <= pi/4, where cylinders on a square lattice touch.
    """
    if not 0.0 <= fraction <= math.pi / 4.0:
        raise ValueError(f"cylinder area fraction {fraction!r} outside [0, pi/4]")
    d = k_inclusion - k_matrix
    return k_matrix * (k_inclusion + k_matrix + fraction * d) / \
        (k_inclusion + k_matrix - fraction * d)


def microbump_k() -> Tuple[float, float]:
    """(k_z, k_xy) of one microbump: the Table 1 laminate through eqs. (10)-(11)."""
    return laminate_k([(TSCI_K[m], t) for m, t in MICROBUMP_LAMINATE_UM])


def bump_underfill_layer_k(
    pitch_um: float = MICROBUMP_PITCH_UM,
    diameter_um: float = MICROBUMP_DIAMETER_UM,
) -> Tuple[float, float, float]:
    """(kx, ky, kz) of a square array of microbumps in underfill.

    kz: eq. (2), bumps and underfill in parallel along the pillars.
    kx = ky: Maxwell-Garnett with the bump's in-plane conductivity as the
    inclusion (validated in tsv_cell_k against the paper's FEM, not its
    eqs. (3)-(7) -- see the module docstring).
    """
    if diameter_um <= 0 or pitch_um < diameter_um:
        raise ValueError(f"bump pitch {pitch_um} um must be at least its diameter "
                         f"{diameter_um} um")
    phi = math.pi * (diameter_um / 2.0) ** 2 / pitch_um ** 2
    kz_bump, kxy_bump = microbump_k()
    k_uf = TSCI_K["underfill"]
    kz = pillar_cell_kz([(phi, kz_bump), (1.0 - phi, k_uf)])
    kxy = maxwell_garnett_2d(kxy_bump, k_uf, phi)
    return kxy, kxy, kz


def tsv_cell_k(r_core_um: float, r_liner_um: float, pitch_um: float) -> Tuple[float, float]:
    """(k_xy, k_z) of the paper's TSV cell: a Cu core with an SiO2 liner on a
    square pitch in silicon. Exists to check this module against the paper's
    own numbers (r 8 / 9 um, pitch 40 um: analytical 175.2 and FEM 169.6
    out of plane; FEM 115.3 in plane)."""
    area = pitch_um ** 2
    v_cu = math.pi * r_core_um ** 2 / area
    v_ox = math.pi * (r_liner_um ** 2 - r_core_um ** 2) / area
    kz = pillar_cell_kz([(v_cu, TSCI_K["copper"]), (v_ox, TSCI_K["sio2"]),
                         (1.0 - v_cu - v_ox, TSCI_K["silicon"])])
    k_inclusion = coated_cylinder_k(TSCI_K["copper"], TSCI_K["sio2"], r_core_um, r_liner_um)
    kxy = maxwell_garnett_2d(k_inclusion, TSCI_K["silicon"], v_cu + v_ox)
    return kxy, kz


if __name__ == "__main__":
    print(f"Source: {CITATION}")
    kxy, kz = tsv_cell_k(8.0, 9.0, 40.0)
    print(f"TSV check cell:   kz {kz:8.3f} (paper 175.2)   kxy {kxy:8.3f} (paper FEM 115.3)")
    kz_b, kxy_b = microbump_k()
    print(f"Microbump:        kz {kz_b:8.3f}                kxy {kxy_b:8.3f}")
    for pitch in (30.0, 40.0, 55.0):
        kx, _, kz = bump_underfill_layer_k(pitch_um=pitch)
        print(f"Bump/underfill @ {pitch:4.0f} um pitch: kz {kz:7.3f}   kx = ky {kx:7.4f}")
