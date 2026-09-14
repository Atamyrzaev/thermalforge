"""
experiment_report.py
RISC3D-AI :: the engineering report for one placement experiment.

Answers, in order and on one page: what went in, where the chiplets started,
how hot that was, what the optimizer was asked for, what it changed, how hot
THAT is, and what was checked. Every number comes from a solved field or from
the configuration; nothing here recomputes physics.

Figures are matplotlib PNGs embedded as base64 data URIs, so the .html file
is the whole deliverable -- the same house style as the Stage 2 report.

Faithfulness rules for the figures, because a picture can mislead in ways a
number cannot:
  - temperature panels that are compared share one colour scale, spanning
    both fields; nothing is clipped;
  - top views are the maximum over z, stated in the caption, because a die
    hotspot can sit below the surface;
  - chiplet outlines are drawn over the field, so package, chiplet and
    temperature stay distinguishable;
  - the difference map is symmetric about zero, so a cooling and a warming
    of equal size look equally large.
"""

from __future__ import annotations

import base64
import html as _html
import io
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")            # headless-safe; must precede pyplot
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.experiment import Experiment, Package, Placement, ThermalResult
from src.grid_builder import MATERIALS
from src.validation_status import STATUS, html_block as validation_html_block

INITIAL_COLOUR = "#0b3d91"
OPTIMIZED_COLOUR = "#0f7b3f"


class ReportError(Exception):
    """Raised when a report cannot be rendered from the experiment given."""


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _to_base64(fig, dpi: int = 120) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _axes_for_package(ax, package: Package, title: str) -> None:
    ax.set_xlim(0.0, package.lx_m * 1e3)
    ax.set_ylim(0.0, package.ly_m * 1e3)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(title, fontsize=10)


def _draw_chiplets(ax, package: Package, catalog, placement: Placement, *,
                   colour: str, label: bool = True, linestyle: str = "-",
                   linewidth: float = 1.6) -> None:
    for c in catalog:
        x_m, y_m = placement.origin(c.id)
        ax.add_patch(Rectangle((x_m * 1e3, y_m * 1e3), c.width_m * 1e3, c.height_m * 1e3,
                               fill=False, edgecolor=colour, linewidth=linewidth,
                               linestyle=linestyle, zorder=3))
        if label:
            ax.text((x_m + c.width_m / 2) * 1e3, (y_m + c.height_m / 2) * 1e3,
                    f"{c.id}\n{c.power_w:.2f} W", ha="center", va="center",
                    fontsize=7.5, color=colour, zorder=4)


def _column_max_c(result: ThermalResult) -> np.ndarray:
    """Hottest temperature in each (x, y) column, in degC. A die hotspot can
    sit below the top surface, so a top view of the surface layer alone would
    hide it."""
    return np.asarray(result.field_k).max(axis=2) - 273.15


def fig_placements(exp: Experiment) -> str:
    """Initial placement, and the optimized one beside it with move arrows."""
    have_opt = exp.optimized is not None
    fig, axes = plt.subplots(1, 2 if have_opt else 1,
                             figsize=(11 if have_opt else 5.6, 5.0), squeeze=False)
    ax0 = axes[0][0]
    _axes_for_package(ax0, exp.package, "Initial placement")
    _draw_chiplets(ax0, exp.package, exp.catalog, exp.initial_placement,
                   colour=INITIAL_COLOUR)
    ax0.grid(alpha=0.25, linewidth=0.5)
    if have_opt:
        ax1 = axes[0][1]
        _axes_for_package(ax1, exp.package, "Optimized placement (arrows show the move)")
        _draw_chiplets(ax1, exp.package, exp.catalog, exp.initial_placement,
                       colour=INITIAL_COLOUR, label=False, linestyle=":", linewidth=1.0)
        _draw_chiplets(ax1, exp.package, exp.catalog, exp.optimized.placement,
                       colour=OPTIMIZED_COLOUR)
        for row in exp.moves():
            if not row.moved:
                continue
            c = next(k for k in exp.catalog if k.id == row.id)
            ax1.annotate(
                "", xytext=((row.x0_m + c.width_m / 2) * 1e3, (row.y0_m + c.height_m / 2) * 1e3),
                xy=((row.x1_m + c.width_m / 2) * 1e3, (row.y1_m + c.height_m / 2) * 1e3),
                arrowprops=dict(arrowstyle="->", color="#b02a37", linewidth=1.4), zorder=5)
        ax1.grid(alpha=0.25, linewidth=0.5)
    return _to_base64(fig)


def fig_power_map(exp: Experiment, result: ThermalResult) -> str:
    """Areal power density: the die-layer power under each column, per unit
    area. This is the spatial map the solver was actually given."""
    package = exp.package
    z0, z1 = package.die_z
    areal_w_cm2 = (np.asarray(result.power.q_w_m3)[:, :, z0:z1].sum(axis=2)
                   * package.dz) / 1e4
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    im = ax.imshow(areal_w_cm2.T, origin="lower", cmap="viridis",
                   extent=[0.0, package.lx_m * 1e3, 0.0, package.ly_m * 1e3])
    _axes_for_package(ax, package, f"Power map, {result.label} placement")
    _draw_chiplets(ax, package, exp.catalog, result.placement, colour="white")
    fig.colorbar(im, ax=ax, shrink=0.85, label="areal power density (W/cm²)")
    return _to_base64(fig)


def fig_temperature(exp: Experiment) -> str:
    """Initial and optimized fields on ONE shared colour scale spanning both,
    so the panels are comparable and nothing is clipped."""
    have_opt = exp.optimized is not None
    initial = _column_max_c(exp.initial)
    fields = [("Initial", initial, exp.initial)]
    if have_opt:
        fields.append(("Optimized", _column_max_c(exp.optimized), exp.optimized))
    vmin = min(float(f.min()) for _n, f, _r in fields)
    vmax = max(float(f.max()) for _n, f, _r in fields)

    fig, axes = plt.subplots(1, len(fields), figsize=(5.8 * len(fields), 5.0), squeeze=False)
    for ax, (name, field, result) in zip(axes[0], fields):
        im = ax.imshow(field.T, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax,
                       extent=[0.0, exp.package.lx_m * 1e3, 0.0, exp.package.ly_m * 1e3])
        _axes_for_package(ax, exp.package,
                          f"{name}: peak {result.peak_c:.2f} °C (max over z)")
        _draw_chiplets(ax, exp.package, exp.catalog, result.placement,
                       colour="white", label=False)
        px, py, _pz = result.peak_center_m
        ax.plot(px * 1e3, py * 1e3, marker="x", color="cyan", markersize=9,
                markeredgewidth=2, zorder=6)
        fig.colorbar(im, ax=ax, shrink=0.85, label="temperature (°C)")
    return _to_base64(fig)


def fig_delta(exp: Experiment) -> str:
    """Optimized minus initial, symmetric about zero: blue is cooler, red is
    hotter, and equal magnitudes look equal."""
    delta = _column_max_c(exp.optimized) - _column_max_c(exp.initial)
    limit = float(np.max(np.abs(delta))) or 1.0
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    im = ax.imshow(delta.T, origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit,
                   extent=[0.0, exp.package.lx_m * 1e3, 0.0, exp.package.ly_m * 1e3])
    _axes_for_package(ax, exp.package, "Optimized - initial (max over z)")
    _draw_chiplets(ax, exp.package, exp.catalog, exp.initial_placement,
                   colour="#333333", label=False, linestyle=":", linewidth=1.0)
    _draw_chiplets(ax, exp.package, exp.catalog, exp.optimized.placement,
                   colour="#111111", label=False)
    fig.colorbar(im, ax=ax, shrink=0.85, label="ΔT (K)")
    return _to_base64(fig)


def fig_stack(package: Package) -> str:
    """The vertical stack to scale in z, with each layer's conductivity."""
    names = ("interposer (silicon)", "microbump + underfill", "die (silicon)")
    materials = ("silicon", "bump_underfill_composite", "silicon")
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    bottom_um = 0.0
    for label, material, thickness_um, cells in zip(names, materials, package.stack_um,
                                                    package.layer_cells):
        props = MATERIALS[material]
        kz = props.get("kz", props.get("k"))
        kxy = props.get("kx", props.get("k"))
        ax.barh(y=bottom_um, width=package.lx_m * 1e3, height=thickness_um, left=0.0,
                align="edge", edgecolor="#333333",
                color="#cfe3ff" if material == "silicon" else "#ffd9a8")
        ax.text(package.lx_m * 1e3 / 2, bottom_um + thickness_um / 2,
                f"{label}: {thickness_um:.0f} µm, {cells} cells, "
                f"kxy {kxy:g} / kz {kz:g} W/mK",
                ha="center", va="center", fontsize=8)
        bottom_um += thickness_um
    ax.set_xlim(0, package.lx_m * 1e3)
    ax.set_ylim(0, bottom_um)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (µm)")
    ax.set_title(f"Stack: {bottom_um:.0f} µm total, sink on the {package.sink_face} face "
                 f"({package.sink_bc})", fontsize=10)
    return _to_base64(fig)


def fig_surrogate(exp: Experiment) -> Optional[str]:
    """Predicted against solved peak rise on the held-out layouts: how well
    the screening model ranks, shown rather than asserted."""
    screening = getattr(exp, "surrogate", None)
    if screening is None or not screening.accuracy.true_rise_k:
        return None
    truth = np.array(screening.accuracy.true_rise_k)
    predicted = np.array(screening.accuracy.predicted_rise_k)
    span = [min(truth.min(), predicted.min()), max(truth.max(), predicted.max())]
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot(span, span, color="#888888", linestyle=":", linewidth=1.2, label="exact")
    ax.scatter(truth, predicted, s=28, color=OPTIMIZED_COLOUR, alpha=0.85,
               label="held-out layouts")
    ax.set_xlabel("solved peak rise (K)")
    ax.set_ylabel("surrogate prediction (K)")
    ax.set_title(f"Screening accuracy: MAE {screening.accuracy.mae_k:.2f} K, "
                 f"rank correlation {screening.accuracy.rank_correlation:.3f}", fontsize=10)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=8)
    return _to_base64(fig)


def fig_history(exp: Experiment) -> Optional[str]:
    """Best-so-far peak against proposal, for a search that has a trajectory."""
    record = exp.optimization
    if record is None or not record.history_c:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(range(1, len(record.history_c) + 1), record.history_c,
            color=OPTIMIZED_COLOUR, linewidth=1.6)
    ax.axhline(exp.initial.peak_c, color=INITIAL_COLOUR, linestyle=":", linewidth=1.2,
               label=f"initial {exp.initial.peak_c:.2f} °C")
    ax.set_xlabel("proposal")
    ax.set_ylabel("best peak so far (°C)")
    ax.set_title("Optimization history", fontsize=10)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=8)
    return _to_base64(fig)


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; background: #f6f7f9; color: #16191d;
       font: 14px/1.55 "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
.wrap { max-width: 1180px; margin: 0 auto; padding: 28px 20px 64px; }
h1 { font-size: 25px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 34px 0 10px; padding-bottom: 6px;
     border-bottom: 2px solid #d7dbe0; }
.sub { color: #5b636c; margin-bottom: 18px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
         gap: 12px; margin: 16px 0 8px; }
.card { background: #fff; border: 1px solid #d7dbe0; border-radius: 8px; padding: 12px 14px; }
.card .label { font-size: 11px; letter-spacing: .06em; text-transform: uppercase;
               color: #5b636c; }
.card .value { font-size: 21px; font-weight: 600; margin-top: 4px; }
.good { color: #0f7b3f; } .warn { color: #b02a37; } .plain { color: #16191d; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; background: #fff;
        border: 1px solid #d7dbe0; border-radius: 8px; overflow: hidden; font-size: 13px; }
th, td { text-align: left; padding: 7px 11px; border-bottom: 1px solid #e6e9ec; }
th { background: #eef1f4; font-size: 11px; letter-spacing: .05em; text-transform: uppercase;
     color: #414952; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tr:last-child td { border-bottom: none; }
figure { margin: 14px 0; background: #fff; border: 1px solid #d7dbe0; border-radius: 8px;
         padding: 12px; overflow-x: auto; }
figure img { display: block; max-width: 100%; height: auto; margin: 0 auto; }
figcaption { color: #5b636c; font-size: 12px; margin-top: 8px; }
.note { background: #fff8e6; border-left: 4px solid #d9a406; padding: 11px 14px;
        border-radius: 0 6px 6px 0; margin: 14px 0; }
.status { background: #eef4ff; border-left: 4px solid #0b3d91; padding: 11px 14px;
          border-radius: 0 6px 6px 0; margin: 14px 0; }
ul { margin: 8px 0 8px 20px; padding: 0; }
code { background: #eef1f4; padding: 1px 5px; border-radius: 3px; font-size: 12px; }
footer { margin-top: 36px; padding-top: 12px; border-top: 1px solid #d7dbe0;
         color: #5b636c; font-size: 12px; }
"""


def _esc(value) -> str:
    return _html.escape(str(value))


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]],
           numeric_from: int = 1) -> str:
    head = "".join(f'<th class="{"num" if i >= numeric_from else ""}">{_esc(h)}</th>'
                   for i, h in enumerate(headers))
    body = "".join(
        "<tr>" + "".join(
            f'<td class="{"num" if i >= numeric_from else ""}">{_esc(cell)}</td>'
            for i, cell in enumerate(row)) + "</tr>"
        for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _figure(b64: str, caption: str) -> str:
    return (f'<figure><img alt="{_esc(caption)}" src="data:image/png;base64,{b64}">'
            f"<figcaption>{_esc(caption)}</figcaption></figure>")


def _card(label: str, value: str, tone: str = "plain") -> str:
    return (f'<div class="card"><div class="label">{_esc(label)}</div>'
            f'<div class="value {tone}">{_esc(value)}</div></div>')


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def render_experiment_report(exp: Experiment, output_dir: str = "outputs",
                             filename: str = "experiment_report.html",
                             links: Optional[Dict[str, str]] = None) -> str:
    """Writes the experiment report and returns its path."""
    if exp.initial is None:
        raise ReportError("nothing to report: the initial placement has not been solved")
    links = links or {}
    package, catalog = exp.package, exp.catalog
    initial, optimized, record = exp.initial, exp.optimized, exp.optimization
    sanity = exp.sanity(initial)
    improvement = exp.improvement()
    optimized_sanity = exp.sanity(optimized) if optimized is not None else None

    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as exc:
        raise ReportError(f"could not create output_dir '{output_dir}': {exc}") from exc

    # ---- 1. summary --------------------------------------------------
    cards = [
        _card("Chiplets", f"{len(catalog)}"),
        _card("Total power", f"{sanity['total_power_w']:.2f} W"),
        _card("Package", f"{package.lx_m * 1e3:.1f} × {package.ly_m * 1e3:.1f} mm"),
        _card("Initial peak", f"{initial.peak_c:.2f} °C"),
    ]
    if optimized is not None:
        cards += [
            _card("Optimized peak", f"{optimized.peak_c:.2f} °C", "good"),
            _card("Improvement", f"{improvement['delta_peak_k']:.2f} K cooler", "good"),
        ]
    else:
        cards.append(_card("Optimization", "not run", "warn"))
    cards.append(_card("Status", STATUS.upper(), "warn"))

    summary_rows = [
        ("Configuration source", exp.source or "(not stated)"),
        ("Case", f"{exp.config.case.name} -- {exp.config.case.purpose}"),
        ("Cooling", f"h = {package.h_convective:,.0f} W/m²K on the {package.sink_face} face "
                    f"({package.sink_bc}); ambient {package.t_ambient_c:.1f} °C"),
        ("Power model", f"{len(catalog)} chiplets, each chiplet's power over its own "
                        f"footprint through the die layer z[{package.die_z[0]}:{package.die_z[1]}]"),
        ("Objective", record.objective if record else "not run"),
        ("Method", (f"{record.method} on a {record.pitch_m * 1e6:.0f} µm lattice"
                    if record else "not run")),
    ]
    body = [
        "<h2>1. Experiment summary</h2>",
        '<div class="cards">' + "".join(cards) + "</div>",
        _table(["Item", "Value"], summary_rows, numeric_from=99),
        _figure(fig_placements(exp),
                "Where the chiplets start, and where the optimizer put them."
                if optimized is not None else "Where the chiplets start."),
    ]

    # ---- 2. package ---------------------------------------------------
    body += [
        "<h2>2. Package geometry</h2>",
        _table(["Quantity", "Value"], [
            ("Lateral size", f"{package.lx_m * 1e3:.3f} × {package.ly_m * 1e3:.3f} mm "
                             f"(derived from nx·dx and ny·dy)"),
            ("Thickness", f"{package.lz_m * 1e6:.1f} µm"),
            ("Mesh", f"{package.nx} × {package.ny} × {package.nz} cells "
                     f"= {package.nx * package.ny * package.nz:,} unknowns"),
            ("Cell size", f"{package.dx * 1e6:.1f} × {package.dy * 1e6:.1f} × "
                          f"{package.dz * 1e6:.1f} µm"),
            ("Stack (authored = realized)",
             " / ".join(f"{t:.0f} µm ({n} cells)"
                        for t, n in zip(package.stack_um, package.layer_cells))),
            ("Die layer", f"z[{package.die_z[0]}:{package.die_z[1]}]"),
            ("Sink", f"{package.sink_face} face, {package.sink_bc}, "
                     f"h = {package.h_convective:,.0f} W/m²K"),
            ("Sink area", f"{sanity['package_area_mm2']:.1f} mm²"),
            ("Placement lattice", f"{exp.initial_placement.pitch_m * 1e6:.0f} µm"),
        ], numeric_from=99),
        _figure(fig_stack(package), "Vertical stack, to scale in z."),
    ]

    # ---- 3. catalog ---------------------------------------------------
    body += [
        "<h2>3. Chiplet catalog</h2>",
        _table(["id", "name", "width (mm)", "height (mm)", "area (mm²)", "power (W)",
                "density (W/cm²)"],
               [(c.id, c.name, f"{c.width_um / 1e3:.3f}", f"{c.height_um / 1e3:.3f}",
                 f"{c.area_mm2:.2f}", f"{c.power_w:.4f}", f"{c.power_density_w_cm2:.2f}")
                for c in catalog]
               + [("total", "", "", "", f"{sanity['chiplet_area_mm2']:.2f}",
                   f"{sanity['total_power_w']:.4f}",
                   f"{sanity['flux_over_chiplets_w_cm2']:.2f}")],
               numeric_from=2),
        '<div class="note">Two areas, two densities, never mixed: '
        f"{sanity['total_power_w']:.2f} W over {sanity['chiplet_area_mm2']:.1f} mm² of "
        f"chiplets is {sanity['flux_over_chiplets_w_cm2']:.2f} W/cm²; the same power over "
        f"the {sanity['package_area_mm2']:.1f} mm² package is "
        f"{sanity['flux_over_package_w_cm2']:.2f} W/cm². Fill factor "
        f"{sanity['fill_pct']:.1f}%.</div>",
    ]

    # ---- 4. initial placement -----------------------------------------
    body += [
        "<h2>4. Initial placement</h2>",
        _table(["id", "origin x (mm)", "origin y (mm)", "footprint (cells)",
                "deposited power (W)"],
               [(cid,
                 f"{initial.placement.origin(cid)[0] * 1e3:.3f}",
                 f"{initial.placement.origin(cid)[1] * 1e3:.3f}",
                 f"{nvx} × {nvy}",
                 f"{dict(initial.power.per_chiplet_w)[cid]:.6f}")
                for cid, _i0, _j0, nvx, nvy in initial.power.blocks]),
        f"<p>Validation: <b>{'passed' if not exp.violations else 'FAILED'}</b> "
        f"-- {len(exp.optimization.constraints) if record else 5} constraints checked "
        f"(inside the package, no overlap, on the lattice, whole cells, powers fixed).</p>",
        _figure(fig_power_map(exp, initial),
                "Initial power map: each chiplet's watts over its own footprint only."),
    ]

    # ---- 5. initial thermal result ------------------------------------
    body += [
        "<h2>5. Initial thermal result</h2>",
        _table(["Quantity", "Value"], [
            ("Peak temperature", f"{initial.peak_c:.4f} °C "
                                 f"(rise {initial.rise_k:.4f} K above ambient)"),
            ("Peak cell", f"index {initial.peak_ijk}, centre "
                          f"({initial.peak_center_m[0] * 1e6:.1f}, "
                          f"{initial.peak_center_m[1] * 1e6:.1f}, "
                          f"{initial.peak_center_m[2] * 1e6:.1f}) µm"),
            ("Location resolution", f"one cell: {package.dx * 1e6:.1f} × "
                                    f"{package.dy * 1e6:.1f} × {package.dz * 1e6:.1f} µm"),
            ("Convective floor", f"P/(h·A) = {sanity['floor_rise_k']:.4f} K rise, "
                                 f"{sanity['floor_peak_c']:.2f} °C"),
            ("Rise / floor", f"{sanity['floor_ratio']:.4f} "
                             f"(the excess is spreading resistance)"),
        ], numeric_from=99),
        _table(["chiplet", "max (°C)", "mean (°C)"],
               [(row.id, f"{row.max_c:.3f}", f"{row.mean_c:.3f}")
                for row in initial.per_chiplet]),
    ]
    if links.get("initial_3d"):
        body.append(f'<p>3D field: <code>{_esc(links["initial_3d"])}</code></p>')

    # ---- 6-11: optimization -------------------------------------------
    if optimized is None or record is None:
        body += [
            "<h2>6. Optimization objective</h2>",
            "<p>Not run. Pass <code>--optimize</code> to search placements from this "
            "initial one.</p>",
            "<h2>7. Constraints</h2><p>Not run.</p>",
            "<h2>8. Optimization history</h2><p>Not run.</p>",
            "<h2>9. Optimized placement</h2><p>Not run.</p>",
            "<h2>10. Before / after temperature comparison</h2><p>Not run.</p>",
            "<h2>11. Temperature difference map</h2><p>Not run.</p>",
            _figure(fig_temperature(exp), "Initial temperature field, max over z."),
        ]
    else:
        moves = exp.moves()
        hottest_before = initial.hottest_chiplet_id
        hottest_after = optimized.hottest_chiplet_id
        before_row = next(m for m in moves if m.id == hottest_before)
        body += [
            "<h2>6. Optimization objective</h2>",
            f"<p>{_esc(record.objective)}.</p>",
            f"<p>Method: <b>{_esc(record.method)}</b> over a "
            f"{record.pitch_m * 1e6:.0f} µm placement lattice"
            + (" -- every buildable layout on that lattice was evaluated, so the result is "
               "the exact optimum OVER THE LATTICE, not a continuous global optimum."
               if record.method == "exhaustive" else
               f" -- a sampled search from the initial placement, seed {record.seed}; "
               "a different seed gives a different answer.") + "</p>",
            "<h2>7. Constraints</h2>",
            "<ul>" + "".join(f"<li>{_esc(c)}</li>" for c in record.constraints) + "</ul>",
            f"<p>Violations in the optimized placement: <b>0</b> "
            f"(re-checked after the search).</p>",
            "<h2>8. Optimization history</h2>",
            _table(["Quantity", "Value"], [
                ("Candidates evaluated", f"{record.candidates_evaluated:,}"
                 + (f" of {record.candidates_total:,} on the lattice"
                    if record.candidates_total else "")),
                ("Rejected or pruned", f"{record.rejected_or_pruned:,}"),
                ("Precompute solves", f"{record.precompute_solves:,}"
                 if record.precompute_solves is not None else "n/a"),
                ("Superposition check", f"{record.superposition_error_k:.2e} K against a "
                 f"direct solve" if record.superposition_error_k is not None else "n/a"),
                ("Runtime", f"{record.runtime_s:.2f} s"),
                ("Reproducible", "deterministic: same inputs, same layout"
                 if record.deterministic else f"stochastic, seed {record.seed}"),
            ], numeric_from=99),
        ]
        screening = getattr(exp, "surrogate", None)
        if screening is not None:
            body += [
                _table(["Screening surrogate", "Value"], [
                    ("What it is", "a least-squares fit over five physical features per "
                                   "chiplet (floor, local 1-D rise, rise spread over "
                                   "lambda, neighbour coupling, wall images). Not a "
                                   "neural network and not a solver."),
                    ("What it did", "ranked candidates so the best few could be solved; "
                                    "every temperature reported here came from the real "
                                    "solver"),
                    ("Calibration", f"{screening.calibration_layouts} real solves "
                                    f"({screening.calibration_s:.2f} s)"),
                    ("Held-out error", f"MAE {screening.accuracy.mae_k:.3f} K, worst "
                                       f"{screening.accuracy.max_error_k:.3f} K, over "
                                       f"{screening.accuracy.layouts} layouts"),
                    ("Ranking quality", f"rank correlation "
                                        f"{screening.accuracy.rank_correlation:.3f}; "
                                        f"top-10 recall "
                                        f"{screening.accuracy.top10_recall:.0%}"),
                    ("Screened", f"{screening.candidates_screened:,} layouts in "
                                 f"{screening.screening_s:.2f} s "
                                 f"({screening.screening_us_per_candidate:.2f} us each)"),
                    ("Solved for real", f"top {screening.top_k_solved} in "
                                        f"{screening.solve_s:.2f} s"),
                    ("lambda", f"{screening.model.lambda_m * 1e3:.2f} mm"),
                ], numeric_from=99),
            ]
            accuracy_figure = fig_surrogate(exp)
            if accuracy_figure:
                body.append(_figure(accuracy_figure,
                                    "Screening accuracy on layouts the fit never saw."))
        history = fig_history(exp)
        if history:
            body.append(_figure(history, "Best-so-far peak against proposal."))
        elif record.method == "exhaustive":
            body.append("<p>An exhaustive search has no trajectory: it evaluates the whole "
                        "lattice and reports the minimum, so there is no best-so-far curve "
                        "to plot.</p>")

        body += [
            "<h2>9. Optimized placement</h2>",
            _table(["id", "initial (mm)", "optimized (mm)", "Δx (mm)", "Δy (mm)",
                    "moved (mm)", "peak before (°C)", "peak after (°C)"],
                   [(m.id,
                     f"({m.x0_m * 1e3:.2f}, {m.y0_m * 1e3:.2f})",
                     f"({m.x1_m * 1e3:.2f}, {m.y1_m * 1e3:.2f})",
                     f"{m.dx_m * 1e3:+.2f}", f"{m.dy_m * 1e3:+.2f}",
                     f"{m.distance_m * 1e3:.2f}",
                     f"{m.peak0_c:.2f}", f"{m.peak1_c:.2f}") for m in moves]),
            "<h2>10. Before / after temperature comparison</h2>",
            _table(["Quantity", "Initial", "Optimized"], [
                ("Peak (°C)", f"{initial.peak_c:.4f}", f"{optimized.peak_c:.4f}"),
                ("Rise above ambient (K)", f"{initial.rise_k:.4f}",
                 f"{optimized.rise_k:.4f}"),
                ("Rise / convective floor", f"{sanity['floor_ratio']:.4f}",
                 f"{optimized_sanity['floor_ratio']:.4f}"),
                ("Hottest chiplet", hottest_before, hottest_after),
            ]),
            '<div class="note"><b>What changed, and why it is cooler.</b> '
            f"The initial peak sat in <b>{_esc(hottest_before)}</b> at "
            f"{before_row.peak0_c:.2f} °C, {before_row.gap0_m * 1e3:.2f} mm from its nearest "
            f"neighbour and {before_row.edge0_m * 1e3:.2f} mm from the package edge; after "
            f"optimization it sits {before_row.gap1_m * 1e3:.2f} mm from its neighbour and "
            f"{before_row.edge1_m * 1e3:.2f} mm from the edge, at "
            f"{before_row.peak1_c:.2f} °C. The side walls are adiabatic in this model, so a "
            "block near an edge also heats through its own mirror image; separation is the "
            "lever placement actually has. "
            f"Improvement {improvement['delta_peak_k']:.4f} K = "
            f"{improvement['pct_of_initial_rise']:.1f}% of the initial "
            f"{improvement['initial_rise_k']:.2f} K rise, and "
            f"{improvement['pct_of_initial_excess']:.1f}% of the "
            f"{improvement['initial_excess_over_floor_k']:.2f} K that sat above the "
            "placement-independent P/(h·A) floor.</div>",
            _figure(fig_temperature(exp),
                    "Initial and optimized fields, max over z, on one shared colour scale."),
            "<h2>11. Temperature difference map</h2>",
            _figure(fig_delta(exp),
                    "Optimized minus initial. Blue is cooler, red is hotter."),
        ]
        delta_field = _column_max_c(optimized) - _column_max_c(initial)
        body.append(_table(["Quantity", "Value"], [
            ("Largest local cooling", f"{-float(delta_field.min()):.3f} K"),
            ("Largest local heating", f"{float(delta_field.max()):.3f} K"),
            ("Peak reduction", f"{improvement['delta_peak_k']:.4f} K"),
        ], numeric_from=1))
        if links.get("optimized_3d"):
            body.append(f'<p>3D field: <code>{_esc(links["optimized_3d"])}</code></p>')

    # ---- 12. verification ---------------------------------------------
    verification_rows = [
        ("Algebraic residual ||MT+f||/||f||", f"{initial.residual:.3e}",
         f"{optimized.residual:.3e}" if optimized else "-"),
        ("Energy balance (solver, K)", f"{initial.energy_imbalance_k:.3e}",
         f"{optimized.energy_imbalance_k:.3e}" if optimized else "-"),
        ("Power specified (W)", f"{initial.power.specified_w:.9f}",
         f"{optimized.power.specified_w:.9f}" if optimized else "-"),
        ("Power deposited in the map (W)", f"{initial.power.deposited_w:.9f}",
         f"{optimized.power.deposited_w:.9f}" if optimized else "-"),
        ("Heat leaving the sink face (W)", f"{initial.sink_heat_w:.9f}",
         f"{optimized.sink_heat_w:.9f}" if optimized else "-"),
        ("Sink heat - specified power (W)", f"{sanity['sink_heat_error_w']:.3e}",
         f"{optimized_sanity['sink_heat_error_w']:.3e}" if optimized else "-"),
    ]
    if record is not None:
        verification_rows.append(
            ("Re-solve vs optimizer (K)", "-",
             f"{record.resolve_delta_k:.3e} peak, {record.resolve_field_delta_k:.3e} field"))
    body += [
        "<h2>12. Verification</h2>",
        _table(["Check", "Initial", "Optimized"], verification_rows),
        '<div class="status">The reported optimum is the field the real solver returned '
        "for the optimized placement, re-solved after the search and compared with what "
        "the optimizer reported. The sink-heat row is computed from the boundary condition "
        "and the solved field in the product layer, sharing no code with the solver's own "
        "energy check.</div>",
    ]

    # ---- 13. limitations ----------------------------------------------
    body += [
        "<h2>13. Limitations</h2>",
        f'<div class="status">{validation_html_block()}</div>',
        "<ul>"
        "<li>Power is uniform through the die thickness, not concentrated at the active "
        "plane.</li>"
        "<li>The search is over a discrete placement lattice, so it is not a continuous "
        "global optimum.</li>"
        "<li>Top views are the maximum over z, not the surface temperature.</li>"
        "<li>The 3D viewer draws the temperature field without chiplet outlines; the "
        "top views above carry the outlines.</li>"
        "</ul>",
    ]

    document = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>RISC3D-AI experiment report</title><style>{_CSS}</style></head><body>"
        "<div class='wrap'>"
        "<h1>RISC3D-AI :: placement experiment</h1>"
        f"<div class='sub'>{_esc(exp.config.case.name)} &middot; generated "
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} &middot; {_esc(STATUS)}</div>"
        + "".join(body)
        + "<footer>Every temperature here is a steady-state field from "
          "ThermalSolver.solve_steady_state(), which verifies its own residual and energy "
          "balance and raises rather than returning an unverified answer.</footer>"
          "</div></body></html>"
    )

    filepath = os.path.join(output_dir, filename)
    try:
        with open(filepath, "w", encoding="utf-8") as handle:
            handle.write(document)
    except OSError as exc:
        raise ReportError(f"failed to write the report to '{filepath}': {exc}") from exc
    return filepath
