// RISC3D Studio :: downloadable optimization report.
//
// A single self-contained HTML file (maps embedded as PNG data URLs) holding
// everything needed to review a result away from the app: the inputs, the
// before/after maps on one shared scale, every chiplet's move, the proof
// certificate and the solver checks of both solves.

import { deltaColor, tempColor } from "./colormap.js";
import { drawPackageMap, formatW } from "./mapdraw.js";
import { escapeHtml } from "./floorplan.js";

const MAP_W = 460, MAP_H = 480;

function mapImage(opts) {
  const canvas = document.createElement("canvas");
  canvas.width = MAP_W * 2; canvas.height = MAP_H * 2;
  const ctx = canvas.getContext("2d");
  ctx.scale(2, 2);
  ctx.fillStyle = "#0a0f19";
  ctx.fillRect(0, 0, MAP_W, MAP_H);
  drawPackageMap(ctx, MAP_W, MAP_H, opts);
  return canvas.toDataURL("image/png");
}

function colorbarImage(kind) {
  const w = 256, h = 12;
  const canvas = document.createElement("canvas");
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(w, h);
  for (let x = 0; x < w; x++) {
    const t = x / (w - 1);
    const [r, g, b] = kind === "delta" ? deltaColor(t * 2 - 1, 1) : tempColor(t, 0, 1);
    for (let y = 0; y < h; y++) {
      const o = (y * w + x) * 4;
      img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL("image/png");
}

const num = (v, digits) => (v === null || v === undefined || !Number.isFinite(v) ? "–" : v.toFixed(digits));

function checksRows(label, res) {
  const v = res.verification;
  const balancePct = v.specified_w > 0 ? 100 * Math.abs(v.sink_heat_w - v.specified_w) / v.specified_w : 0;
  return `<tr><td>${label}</td><td>${v.residual.toExponential(1)}</td>
    <td>${v.specified_w.toFixed(3)} W → ${v.sink_heat_w.toFixed(3)} W (${balancePct.toExponential(1)} %)</td>
    <td>${v.unknowns.toLocaleString()}</td></tr>`;
}

export function buildReport({ opt, pkg, scenarioTitle, colours }) {
  const a = opt.initial, b = opt.optimized, imp = opt.improvement, cert = opt.certificate, g = a.grid;
  const withColour = (placement) => placement.map((c) => ({ ...c, color: colours.get(c.id) || "#8aa0b8" }));
  const lo = Math.floor(Math.min(a.t_min_c, b.t_min_c));
  const hi = Math.ceil(Math.max(a.t_max_c, b.t_max_c));
  const delta = new Float32Array(a.colmax.length);
  let span = 0;
  for (let n = 0; n < delta.length; n++) {
    delta[n] = b.colmax[n] - a.colmax[n];
    span = Math.max(span, Math.abs(delta[n]));
  }
  span = Math.max(0.1, Math.ceil(span * 10) / 10);
  const temps = (part) => new Map(part.chiplets.map((c) => [c.id, c.max_c]));
  const heat = (values, kind, l, h) => ({ values, nx: g.nx, ny: g.ny, lo: l, hi: h, kind });
  const imgBefore = mapImage({ sizeMm: g.size_mm, heat: heat(a.colmax, "temp", lo, hi), chiplets: withColour(a.placement), temps: temps(a), peak: a.peak });
  const imgAfter = mapImage({ sizeMm: g.size_mm, heat: heat(b.colmax, "temp", lo, hi), chiplets: withColour(b.placement), temps: temps(b), peak: b.peak });
  const imgDelta = mapImage({ sizeMm: g.size_mm, compact: true, heat: heat(delta, "delta", -span, span), chiplets: withColour(b.placement) });

  const created = new Date();
  const headline = opt.kept_initial
    ? `Already optimal: no layout on the ${cert.pitch_mm} mm grid is cooler than the starting placement (${imp.initial_peak_c.toFixed(2)} °C).`
    : `Peak ${imp.initial_peak_c.toFixed(2)} °C → ${imp.optimized_peak_c.toFixed(2)} °C: −${imp.delta_k.toFixed(2)} K, ${imp.pct_of_rise.toFixed(1)}% of the rise above ambient removed by moving chiplets only.`;

  const hottest = new Map(a.chiplets.map((row) => [row.id, row]));
  const chipletRows = a.placement.map((c) => {
    const area = c.w_mm * c.h_mm;
    const row = hottest.get(c.id) || {};
    const peak = row.hottest_density_w_cm2;
    const pattern = (row.power_pattern || "uniform").replace(/_/g, " ");
    return `<tr><td>${escapeHtml(c.name)}</td><td>${c.w_mm} × ${c.h_mm}</td><td>${formatW(c.power_w)}</td>
      <td>${(c.power_w / (area / 100)).toFixed(1)}</td>
      <td>${escapeHtml(pattern)}${peak && peak > c.power_w / (area / 100) * 1.01
        ? ` · ${peak.toFixed(0)} W/cm² hot region` : ""}</td></tr>`;
  }).join("");
  const moveRows = opt.moves.map((m) => `<tr><td>${escapeHtml(m.name)}</td>
    <td>${m.from_mm[0].toFixed(1)}, ${m.from_mm[1].toFixed(1)}</td>
    <td>${m.to_mm[0].toFixed(1)}, ${m.to_mm[1].toFixed(1)}</td>
    <td>${m.distance_mm > 1e-9 ? m.distance_mm.toFixed(1) : "stays"}</td>
    <td>${m.max_before_c.toFixed(2)}</td><td>${m.max_after_c.toFixed(2)}</td></tr>`).join("");
  const kpi = (label, before, after, unit, digits) =>
    `<tr><td>${label}</td><td>${num(before, digits)} ${unit}</td><td>${num(after, digits)} ${unit}</td></tr>`;

  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>RISC3D placement report · ${escapeHtml(scenarioTitle)}</title>
<style>
  body { font: 14px/1.5 "Segoe UI", system-ui, sans-serif; color: #172033; background: #f4f6fa; margin: 0; }
  main { max-width: 1480px; margin: 0 auto; padding: 28px 32px 40px; }
  h1 { font-size: 24px; margin: 0; } h2 { font-size: 16px; margin: 26px 0 10px; }
  .sub { color: #5a6a85; }
  .headline { margin: 18px 0; padding: 14px 18px; border-radius: 10px; background: #e6f7ef; border: 1px solid #9ad8b9; font-size: 16px; font-weight: 600; }
  .grid3 { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }
  figure { margin: 0; background: #0a0f19; border-radius: 10px; padding: 10px; color: #dfe7f3; }
  figure img { width: 100%; display: block; }
  figcaption { font-weight: 600; margin-bottom: 6px; }
  .scale { display: flex; align-items: center; gap: 8px; font: 12px Consolas, monospace; margin-top: 6px; }
  .scale img { height: 12px; width: 180px; border-radius: 3px; }
  .cols { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 24px; }
  table { border-collapse: collapse; width: 100%; background: #fff; border-radius: 8px; overflow: hidden; }
  th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #e3e8f0; }
  th { background: #eef2f8; font-size: 12px; color: #4a5a75; }
  td { font-family: Consolas, "Cascadia Mono", monospace; font-size: 13px; }
  td:first-child { font-family: inherit; font-weight: 600; }
  .badge { display: inline-block; padding: 2px 12px; border-radius: 999px; background: #e6f7ef; color: #13784a; border: 1px solid #9ad8b9; font-weight: 700; font-size: 12px; }
  .note { color: #5a6a85; font-size: 13px; }
  @media print { body { background: #fff; } main { padding: 0; } .grid3 { gap: 8px; } }
</style></head>
<body><main>
  <h1>RISC3D Studio · placement report</h1>
  <div class="sub">${escapeHtml(scenarioTitle)} · generated ${created.toLocaleString()}</div>
  <div class="headline">${headline}</div>

  <div class="grid3">
    <figure><figcaption>Before</figcaption><img src="${imgBefore}" alt="Before: hottest temperature through the stack">
      <div class="scale">${lo} °C <img src="${colorbarImage("temp")}" alt=""> ${hi} °C</div></figure>
    <figure><figcaption>After (optimized)</figcaption><img src="${imgAfter}" alt="After: hottest temperature through the stack">
      <div class="scale">${lo} °C <img src="${colorbarImage("temp")}" alt=""> ${hi} °C</div></figure>
    <figure><figcaption>Difference (after − before)</figcaption><img src="${imgDelta}" alt="Temperature change">
      <div class="scale">−${span.toFixed(1)} K <img src="${colorbarImage("delta")}" alt=""> +${span.toFixed(1)} K</div></figure>
  </div>
  <p class="note">Maps show the hottest temperature through the stack at each point, one pixel block per mesh cell,
    on one shared scale for before and after. The crosshair marks the peak.</p>

  <div class="cols">
    <section>
      <h2>Inputs</h2>
      <table>
        <tr><td>Package</td><td>${g.size_mm} × ${g.size_mm} mm · ${(g.layers || []).map((l) => `${escapeHtml(l.name)} ${l.thickness_um} µm`).join(" / ")}</td></tr>
        <tr><td>Cooling path</td><td>${(g.cooling_layers && g.cooling_layers.length
          ? g.cooling_layers.map(([label, thickness, k]) => `${escapeHtml(label)} ${thickness} µm (k = ${k} W/m·K)`).join(" → ")
          : "none — the coolant meets the die backside")} → h = ${pkg.h_w_m2k.toLocaleString()} W/m²K · ambient ${pkg.ambient_c} °C</td></tr>
        <tr><td>Mesh</td><td>${g.nx} × ${g.ny} × ${g.nz} cells (${(g.nx * g.ny * g.nz).toLocaleString()} unknowns) · ${g.cell_mm} mm × ${g.cell_mm} mm × ${g.dz_um} µm${g.mesh === "fine" ? " · fine mesh" : ""}</td></tr>
        <tr><td>Placement grid</td><td>${cert.pitch_mm} mm</td></tr>
        <tr><td>Assembly keep-out</td><td>${cert.keepout_um ? `${cert.keepout_um} µm requested, ${cert.keepout_enforced_um} µm enforced on this mesh` : "none"}</td></tr>
      </table>
      <h2>Chiplets</h2>
      <table><tr><th>Chiplet</th><th>Size (mm)</th><th>Power (W)</th><th>Density (W/cm²)</th><th>Power inside</th></tr>${chipletRows}</table>
      <h2>Key figures</h2>
      <table><tr><th></th><th>Before</th><th>After</th></tr>
        ${kpi("Peak temperature", a.peak.t_c, b.peak.t_c, "°C", 2)}
        ${b.metrics.uncertainty ? `<tr><td>Error budget</td><td>± ${a.metrics.uncertainty.combined_k.toFixed(2)} K</td><td>± ${b.metrics.uncertainty.combined_k.toFixed(2)} K</td></tr>` : ""}
        ${kpi("Rise over ambient", a.metrics.rise_k, b.metrics.rise_k, "K", 2)}
        ${kpi("Thermal resistance", a.metrics.rja_k_per_w, b.metrics.rja_k_per_w, "K/W", 3)}
        <tr><td>Peak location</td><td>${a.peak.x_mm.toFixed(2)}, ${a.peak.y_mm.toFixed(2)} mm · ${escapeHtml(a.peak.layer)}</td>
          <td>${b.peak.x_mm.toFixed(2)}, ${b.peak.y_mm.toFixed(2)} mm · ${escapeHtml(b.peak.layer)}</td></tr>
      </table>
    </section>
    <section>
      <h2>Moves</h2>
      <table><tr><th>Chiplet</th><th>Before (x, y) mm</th><th>After (x, y) mm</th><th>Moved (mm)</th><th>Hottest before (°C)</th><th>Hottest after (°C)</th></tr>${moveRows}</table>
      <h2>Proof of optimality <span class="badge">EXACT</span></h2>
      <table>
        <tr><td>Layouts on the grid</td><td>${cert.combinations.toLocaleString()}</td></tr>
        <tr><td>Scored exactly</td><td>${cert.layouts_scored.toLocaleString()} (by superposing full 3-D solutions)</td></tr>
        <tr><td>Ruled out by bound</td><td>${cert.excluded_bound.toLocaleString()} branches (proven lower bound above the best layout)</td></tr>
        <tr><td>Ruled out by overlap</td><td>${cert.excluded_overlap.toLocaleString()} candidate positions</td></tr>
        <tr><td>Identical chiplets</td><td>${cert.identical_chiplets ? `${cert.identical_chiplets} interchangeable, duplicates counted once` : "none"}</td></tr>
        <tr><td>Re-solve check</td><td>${cert.resolve_delta_k.toExponential(1)} K between search and full re-solve (limit 1e-6 K)</td></tr>
        <tr><td>Time</td><td>${cert.unit_solves.toLocaleString()} single-chiplet solves ${cert.precompute_s.toFixed(1)} s · search ${cert.search_s.toFixed(1)} s · total ${cert.total_s.toFixed(1)} s</td></tr>
      </table>
      <h2>Solver checks</h2>
      <table><tr><th>Solve</th><th>Residual</th><th>Energy balance: generated → leaving</th><th>Unknowns</th></tr>
        ${checksRows("Before", a)}${checksRows("After", b)}</table>
    </section>
  </div>

  <h2>Method and scope</h2>
  <p class="note">Steady-state 3-D heat conduction, finite-volume discretization, direct sparse factorization.
    The optimizer uses superposition (the equation is linear, so a layout's temperature rise is the sum of each chiplet's
    rise on its own) and branch-and-bound with a proven lower bound (every chiplet only adds heat). Every layout on the
    placement grid is covered; the winner is re-solved from scratch and refused if the two disagree. Scope: steady state,
    cooling as an effective heat-transfer coefficient on the die backside, uniform power within each chiplet, placement
    on the stated grid. The solver is verified numerically: residual and energy balance on every solve, agreement
    with an independent exhaustive search, and agreement with closed-form solutions for a uniformly heated slab and
    a rectangular flux channel. The error budget combines two measured sensitivities, in quadrature: a
    2 / 1 / 0.5 mm mesh convergence study (Grid Convergence Index, safety factor 1.25) and conductivity
    perturbations (silicon ±10 %, bump composite ±30 %), both scaled to this design's rise above ambient. It
    says how much the answer moves when the model's own inputs move. None of that is validation -- the model
    has not yet been compared with measurements of real hardware.</p>
</main></body></html>`;
}

export function downloadReport(args) {
  const html = buildReport(args);
  const blob = new Blob([html], { type: "text/html" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  const stamp = new Date().toISOString().slice(0, 16).replace(/[-:T]/g, "");
  link.href = url;
  link.download = `risc3d-report-${stamp}.html`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
  return html.length;
}
