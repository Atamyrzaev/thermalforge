// RISC3D Studio :: application.

import { decodeField, getJSON, postJSON } from "./api.js";
import { drawColorbar, tempColor } from "./colormap.js";
import { FloorplanView, escapeHtml } from "./floorplan.js";
import { columnMax, drawPackageMap, formatW, setupCanvas } from "./mapdraw.js";
import { History, openDesign, saveDesign } from "./design.js";
import { drawPowerMap, readPowerMapFile } from "./powermap.js";
import { drawTimeline } from "./timeline.js";
import { downloadReport } from "./report.js";
import { Viewer3D } from "./viewer3d.js";

const $ = (id) => document.getElementById(id);

const LIBRARY = [
  { key: "cpu", name: "CPU", w_mm: 4, h_mm: 4, power_w: 10, color: "#4f8cff", power_pattern: "quad_cores" },
  { key: "gpu", name: "GPU", w_mm: 6, h_mm: 5, power_w: 14, color: "#35c48b", power_pattern: "two_engines" },
  { key: "npu", name: "NPU", w_mm: 5, h_mm: 5, power_w: 8, color: "#ffb347", power_pattern: "mac_array" },
  { key: "hbm", name: "HBM", w_mm: 8, h_mm: 8, power_w: 4, color: "#b27cff", power_pattern: "uniform" },
  { key: "rv", name: "RISC-V core", w_mm: 3, h_mm: 3, power_w: 2.5, color: "#5ad1e6", power_pattern: "uniform" },
  { key: "serdes", name: "SerDes", w_mm: 2, h_mm: 4, power_w: 1.5, color: "#ff7aa8", power_pattern: "uniform" },
  { key: "sram", name: "SRAM", w_mm: 6, h_mm: 6, power_w: 2, color: "#c38cff", power_pattern: "uniform" },
];
const MAX_CHIPLETS = 16;          // matches studio/engine.py
const COOLING_PRESETS = ["1000", "2500", "5000", "25000"];

const PATTERN_LABELS = {
  uniform: "Uniform across the footprint",
  mac_array: "MAC array — 40 % of the area, 70 % of the power",
  two_engines: "Two engines — 35 % of the power each",
  quad_cores: "Four cores — 17 % of the power each",
  custom: "Custom blocks (from the file)",
  imported: "Imported map (from a file)",
};

const state = {
  scenarios: [],
  patterns: { uniform: [] },     // served by the engine, never invented here
  pkg: null,
  chiplets: [],
  selectedId: null,
  shown: null,          // the solved result currently displayed
  opt: null,            // last optimization, fields decoded
  solving: false,
  pending: false,
  animating: false,
  busy: false,
  jobId: null,
  tab: "floorplan",
  focus: false,
  burst: null,          // the last time-stepped run
  burstJob: null,
  scenarioId: null,     // what "Reset" goes back to
  opened: null,         // design opened from a file, if any
  dragging: false,
  viewer: null,
  glMode: "current",
};

let floorplan = null;

// ------------------------------------------------------------------ helpers
const DEFAULT_STACK_UM = [100, 40, 100];
const DEFAULT_KEEPOUT_UM = 100;
// Mirrors MESH_LEVELS in studio/engine.py: how many unknowns each level will
// factorize, and therefore how fine a cell it can afford.
const MESH_CAPS = { standard: 36000, fine: 120000 };
const CELL_CHOICES_MM = [0.25, 0.5, 1, 2];

function stackOf(pkg) {
  return (pkg && pkg.stack_um) || DEFAULT_STACK_UM;
}

// Cells the cooling path adds above the die, mirroring COOLING_STACKS.
const COOLING_CELLS = { direct: 0, lid: 3, lid_heatsink: 6, coldplate: 3 };

function nzOf(pkg) {
  const stack = Math.round(stackOf(pkg).reduce((a, b) => a + b, 0) / 20);
  return stack + (COOLING_CELLS[(pkg && pkg.cooling) || "direct"] || 0);
}

// A chiplet edge only lands on a cell boundary, so any positive keep-out
// costs at least one whole empty cell. This mirrors PackageSpec.keepout_cells.
function keepoutMm(pkg) {
  const requested = pkg && pkg.keepout_um !== undefined ? pkg.keepout_um : 100;
  if (!requested) return 0;
  const cell = cellMm(pkg);
  return Math.ceil(requested / (cell * 1000) - 1e-9) * cell;
}

function sizeYOf(pkg) {
  return (pkg && pkg.size_y_mm) || (pkg && pkg.size_mm) || 20;
}

function cellMm(pkg) {
  const nz = nzOf(pkg);
  const cap = MESH_CAPS[(pkg && pkg.mesh) || "standard"] || MESH_CAPS.standard;
  const sy = sizeYOf(pkg);
  for (const cell of CELL_CHOICES_MM) {
    const nx = pkg.size_mm / cell, ny = sy / cell;
    const whole = Math.abs(nx - Math.round(nx)) < 1e-9 && Math.abs(ny - Math.round(ny)) < 1e-9;
    if (whole && Math.min(nx, ny) >= 8 && nx * ny * nz <= cap) return cell;
  }
  return 1;
}

function compact(n) {
  if (n >= 1e12) return `${(n / 1e12).toFixed(1)} trillion`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} billion`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} million`;
  return Math.round(n).toLocaleString();
}

function setStatus(kind, text) {
  $("status").className = `status ${kind}`;
  $("status-text").textContent = text;
}

function toast(message, kind = "error") {
  const el = $("toast");
  el.textContent = message;
  el.className = `toast ${kind === "info" ? "info" : ""}`;
  el.classList.remove("hidden");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => el.classList.add("hidden"), 6000);
}

function packageRequest() {
  const p = state.pkg;
  return { size_mm: p.size_mm, h_w_m2k: p.h_w_m2k, ambient_c: p.ambient_c, pitch_mm: p.pitch_mm,
           stack_um: stackOf(p), keepout_um: p.keepout_um !== undefined ? p.keepout_um : 100,
           cooling: p.cooling || "direct", mesh: p.mesh || "standard",
           interposer_material: p.interposer_material || "silicon",
           beol_um: p.beol_um || 0,
           size_y_mm: p.size_y_mm || 0 };
}

function chipletRequest(list = state.chiplets) {
  return list.map(({ id, name, w_mm, h_mm, power_w, x_mm, y_mm, power_pattern, blocks, power_grid, fixed }) => ({
    id, name, w_mm, h_mm, power_w, x_mm, y_mm,
    power_pattern: power_pattern || "uniform",
    ...(fixed ? { fixed: true } : {}),
    ...(blocks && blocks.length ? { blocks } : {}),
    ...(power_grid ? { power_grid } : {}),
  }));
}

// The blocks a chiplet actually carries: its own (a file may define them) or
// the served pattern's.
function blocksOf(c) {
  if (c.power_grid) return [];          // an imported map is drawn by the heat, not by outlines
  if (c.blocks && c.blocks.length) return c.blocks;
  return state.patterns[c.power_pattern || "uniform"] || [];
}

function withColours(placement) {
  const colours = new Map(state.chiplets.map((c) => [c.id, c.color]));
  return placement.map((c) => ({ ...c, color: colours.get(c.id) || "#8aa0b8" }));
}

function decodeResult(res) {
  res.field = decodeField(res.field_b64);
  res.colmax = columnMax(res.field, res.grid.nx, res.grid.ny, res.grid.nz);
  delete res.field_b64;
  return res;
}

function violations() {
  const size = state.pkg.size_mm, sizeY = sizeYOf(state.pkg);
  const ids = new Set();
  const messages = [];
  const list = state.chiplets;
  for (const c of list) {
    if (c.x_mm < -1e-9 || c.y_mm < -1e-9 || c.x_mm + c.w_mm > size + 1e-9 || c.y_mm + c.h_mm > sizeY + 1e-9) {
      ids.add(c.id);
      messages.push(`${c.name} is outside the package`);
    }
  }
  for (let a = 0; a < list.length; a++) {
    for (let b = a + 1; b < list.length; b++) {
      const A = list[a], B = list[b];
      const ox = Math.min(A.x_mm + A.w_mm, B.x_mm + B.w_mm) - Math.max(A.x_mm, B.x_mm);
      const oy = Math.min(A.y_mm + A.h_mm, B.y_mm + B.h_mm) - Math.max(A.y_mm, B.y_mm);
      const keepout = keepoutMm(state.pkg);
      if (ox > 1e-9 && oy > 1e-9) {
        ids.add(A.id); ids.add(B.id);
        messages.push(`${A.name} overlaps ${B.name}`);
      } else if (keepout > 0 && ox > 1e-9 - keepout && oy > 1e-9 - keepout) {
        ids.add(A.id); ids.add(B.id);
        const clearance = Math.max(-ox, -oy);
        messages.push(`${A.name} and ${B.name} clear each other by ${clearance.toFixed(2)} mm; `
          + `the keep-out needs ${keepout.toFixed(2)} mm on this mesh`);
      }
    }
  }
  return { ids, messages };
}

// ----------------------------------------------------------------- history
const history = new History();

function snapshot() {
  return JSON.stringify({ pkg: state.pkg, chiplets: state.chiplets, selectedId: state.selectedId });
}

// Call BEFORE changing the design. One entry per completed change: a drag
// records the layout it started from, not every pixel along the way.
function record() {
  if (state.pkg) history.push(snapshot());
}

function restore(json) {
  const s = JSON.parse(json);
  state.pkg = s.pkg;
  state.chiplets = s.chiplets;
  state.selectedId = s.selectedId;
  clearOptimization();
  syncPackageControls();
  renderAll();
  requestSolve();
}

function undo() {
  if (state.busy || state.animating) return;
  const s = history.undo(snapshot());
  if (s) restore(s);
}

function redo() {
  if (state.busy || state.animating) return;
  const s = history.redo(snapshot());
  if (s) restore(s);
}

function updateHistoryButtons() {
  $("btn-undo").disabled = !history.canUndo || state.busy;
  $("btn-redo").disabled = !history.canRedo || state.busy;
}

// A yes/no question with an optional third answer. Resolves true, false or "alt".
function ask({ title, body, ok = "Continue", alt = null, cancel = "Cancel" }) {
  const overlay = $("ask-overlay");
  $("ask-title").textContent = title;
  $("ask-body").textContent = body;
  $("ask-ok").textContent = ok;
  $("ask-cancel").textContent = cancel;
  const altBtn = $("ask-alt");
  altBtn.textContent = alt || "";
  altBtn.classList.toggle("hidden", !alt);
  overlay.classList.remove("hidden");
  return new Promise((resolve) => {
    const done = (value) => {
      overlay.classList.add("hidden");
      $("ask-ok").onclick = $("ask-cancel").onclick = altBtn.onclick = overlay.onclick = null;
      window.removeEventListener("keydown", onKey);
      resolve(value);
    };
    const onKey = (e) => { if (e.key === "Escape") done(false); };
    $("ask-ok").onclick = () => done(true);
    $("ask-alt").onclick = () => done("alt");
    $("ask-cancel").onclick = () => done(false);
    overlay.onclick = (e) => { if (e.target === overlay) done(false); };
    window.addEventListener("keydown", onKey);
    $("ask-ok").focus();
  });
}

// ----------------------------------------------------------------- solving
async function requestSolve() {
  if (!state.pkg) return;
  const warn = $("fp-warning");
  if (!state.chiplets.length) {
    state.shown = null;
    renderAll();
    warn.textContent = "Add a chiplet to start";
    warn.classList.remove("hidden");
    return;
  }
  const bad = violations();
  renderFloorplan(bad.ids);
  if (bad.messages.length) {
    warn.textContent = bad.messages[0] + (bad.messages.length > 1 ? `  (+${bad.messages.length - 1} more)` : "");
    warn.classList.remove("hidden");
    setStatus("error", "Placement not valid");
    return;
  }
  warn.classList.add("hidden");
  if (state.solving) { state.pending = true; return; }

  state.solving = true;
  setStatus("busy", state.shown ? "Solving…" : "Building the 3-D model…");
  try {
    const res = await postJSON("/api/solve", { package: packageRequest(), chiplets: chipletRequest() });
    if (!res.ok) {
      const message = res.violations ? res.violations[0] : res.error || "solve failed";
      warn.textContent = message;
      warn.classList.remove("hidden");
      setStatus("error", "Not solved");
    } else {
      state.shown = decodeResult(res);
      renderAll();
      const built = res.model_built_s ? ` · model built in ${res.model_built_s.toFixed(1)} s` : "";
      setStatus("ok", `Solved in ${res.solve_ms.toFixed(0)} ms${built}`);
    }
  } catch (err) {
    toast(`Cannot reach the engine: ${err.message}`);
    setStatus("error", "Engine unreachable");
  } finally {
    state.solving = false;
    if (state.pending) {
      state.pending = false;
      requestSolve();
    }
  }
}

// --------------------------------------------------------------- rendering
function renderAll() {
  renderFloorplan();
  renderKpis();
  renderBars();
  renderChecks();
  renderChipletList();
  if (state.tab === "view3d") update3D();
}

function heatFor(res) {
  if (!res || res.grid.size_mm !== state.pkg.size_mm
      || (res.grid.size_y_mm || res.grid.size_mm) !== sizeYOf(state.pkg)) return null;
  return { values: res.colmax, nx: res.grid.nx, ny: res.grid.ny,
           lo: Math.floor(res.t_min_c), hi: Math.ceil(res.t_max_c), kind: "temp" };
}

// -- the AI layer: estimates and words, never a reported number -----------

async function whatIf() {
  const part = state.chiplets.find((c) => c.id === state.selectedId);
  const note = $("ed-whatif-note");
  if (!part) return;
  if (state.hint && state.hint.partId === part.id) {      // a second click clears it
    state.hint = null;
    note.textContent = "";
    renderFloorplan();
    return;
  }
  note.textContent = "estimating…";
  try {
    const res = await postJSON("/api/surrogate/what-if", {
      package: packageRequest(), chiplets: chipletRequest(), part_id: part.id,
    });
    if (!res.ok) throw new Error(res.error);
    const spots = res.positions || [];
    if (!spots.length) { note.textContent = "no free position on the grid for this part"; return; }
    const peaks = spots.map((s) => s.peak_c);
    state.hint = {
      partId: part.id, spots, wMm: part.w_mm, hMm: part.h_mm,
      lo: Math.min(...peaks), hi: Math.max(...peaks),
      bestKey: `${res.best.x_mm},${res.best.y_mm}`,
    };
    const spread = Math.max(...peaks) - Math.min(...peaks);
    note.innerHTML = `<b>${escapeHtml(res.label)}.</b> ${spots.length} positions in `
      + `${res.ms.toFixed(0)} ms, spread ${spread.toFixed(1)} K. Green outline: the coolest estimate `
      + `(${res.best.x_mm}, ${res.best.y_mm}) mm. Move the part there and the solver gives the real number.`;
    renderFloorplan();
  } catch (err) {
    note.textContent = String(err.message || err);
  }
}

function renderCopilot(res) {
  const out = $("copilot-out");
  const bits = [];
  if (res.not_understood && res.not_understood.length) {
    bits.push(`<div class="warn">Not understood, so not applied: `
      + res.not_understood.map((w) => `<code>${escapeHtml(w)}</code>`).join(", ") + `</div>`);
  }
  // Every temperature is a citation: render it as a link to its solve record.
  const linked = escapeHtml(res.narrative).replace(
    /\[(rec|req):([0-9a-f]{12}):([A-Za-z0-9_.\[\]-]+)\]/g,
    (_m, kind, id, path) => ` <a href="#" class="cite" data-record="${id}" title="${kind === "req"
      ? "what you asked for" : "the solve this number came from"}: ${escapeHtml(path)}">[${id.slice(0, 6)}]</a>`);
  bits.push(`<p class="copilot-answer">${linked}</p>`);
  out.innerHTML = bits.join("");
  out.querySelectorAll("a.cite").forEach((a) => a.addEventListener("click", async (e) => {
    e.preventDefault();
    try {
      const r = await getJSON(`/api/records/${a.dataset.record}`);
      if (!r.ok) throw new Error(r.error);
      const rec = r.record;
      toast(`${rec.kind} ${rec.id} · commit ${rec.commit} · checks passed`
        + (rec.checks.residual === undefined ? ""
           : ` (residual ${Number(rec.checks.residual).toExponential(1)})`), "info");
    } catch (err) { toast(String(err.message || err)); }
  }));
  $("copilot-apply").classList.toggle("hidden", !res.placement);
}

async function askCopilot() {
  const text = $("copilot-text").value.trim();
  if (!text) return;
  const out = $("copilot-out");
  out.textContent = "planning, solving and proving…";
  try {
    const res = await postJSON("/api/copilot", {
      text, package: packageRequest(), chiplets: chipletRequest(),
    });
    if (!res.ok) throw new Error(res.error);
    state.copilot = res;
    renderCopilot(res);
  } catch (err) {
    out.textContent = String(err.message || err);
  }
}

function applyCopilot() {
  const res = state.copilot;
  if (!res || !res.placement) return;
  record();
  const byId = new Map(res.placement.map((p) => [p.id, p]));
  state.chiplets = state.chiplets.map((c) => (byId.has(c.id)
    ? { ...c, x_mm: byId.get(c.id).x_mm, y_mm: byId.get(c.id).y_mm } : c));
  if (res.package) {
    state.pkg = { ...state.pkg, ...res.package };
    syncPackageControls();
  }
  state.hint = null;
  packageChanged();
}

function renderFloorplan(invalidIds = new Set()) {
  if (!floorplan || !state.pkg) return;
  const res = state.shown;
  const heat = heatFor(res);
  const temps = new Map(((res && res.chiplets) || []).map((r) => [r.id, r.max_c]));
  floorplan.setModel({
    blocksOf,
    sizeMm: state.pkg.size_mm,
    sizeYMm: sizeYOf(state.pkg),
    cellMm: cellMm(state.pkg),
    pitchMm: state.pkg.pitch_mm,
    heat,
    chiplets: state.chiplets,
    selectedId: state.selectedId,
    invalidIds,
    hint: state.hint && state.hint.partId === state.selectedId ? state.hint : null,
    temps: state.animating ? null : temps,
    peak: heat && !invalidIds.size && !state.animating ? res.peak : null,
    locked: state.animating || state.busy,
  });
  $("fp-cb-min").textContent = heat ? `${heat.lo} °C` : "";
  $("fp-cb-max").textContent = heat ? `${heat.hi} °C` : "";
}

function renderKpis() {
  const res = state.shown;
  if (!res) {
    for (const id of ["kpi-peak", "kpi-rise", "kpi-rja", "kpi-power", "kpi-flux"]) $(id).textContent = "–";
    $("kpi-peak-sub").textContent = "";
    return;
  }
  const m = res.metrics;
  $("kpi-peak").textContent = res.peak.t_c.toFixed(1);
  const hottest = state.chiplets.find((c) => c.id === res.hottest_chiplet);
  const band = m.uncertainty ? ` · ±${m.uncertainty.combined_k.toFixed(2)} K (mesh + material data)` : "";
  $("kpi-peak-sub").textContent =
    `${hottest ? `inside ${hottest.name}, ` : ""}${res.peak.layer} · x ${res.peak.x_mm.toFixed(2)}, y ${res.peak.y_mm.toFixed(2)} mm${band}`;
  $("kpi-rise").textContent = `${m.rise_k.toFixed(1)} K`;
  $("kpi-rja").textContent = m.rja_k_per_w != null ? `${m.rja_k_per_w.toFixed(2)} K/W` : "–";
  $("kpi-power").textContent = `${formatW(m.total_power_w)} W`;
  $("kpi-flux").textContent = m.flux_chiplets_w_cm2 != null ? `${m.flux_chiplets_w_cm2.toFixed(1)} W/cm²` : "–";
}

function renderBars() {
  const box = $("chiplet-bars");
  const res = state.shown;
  if (!res) { box.innerHTML = '<div class="muted">No result yet.</div>'; return; }
  const rows = [...res.chiplets].sort((a, b) => b.max_c - a.max_c);
  const lo = res.ambient_c;
  const hi = Math.max(lo + 1, ...rows.map((r) => r.max_c));
  const colours = new Map(state.chiplets.map((c) => [c.id, c.color]));
  box.innerHTML = rows.map((r) => {
    const pct = Math.max(2, ((r.max_c - lo) / (hi - lo)) * 100);
    return `<div class="bar-row">
      <span>${escapeHtml(r.name)} <span class="muted">· ${formatW(r.power_w)} W · ${r.density_w_cm2.toFixed(1)} W/cm²</span></span>
      <span class="bar-temp">${r.max_c.toFixed(1)} °C</span>
      <div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${colours.get(r.id) || "#8aa0b8"}"></div></div>
    </div>`;
  }).join("");
}

// The band on every solve is scaled from reference packages, so it describes
// designs LIKE this one. This solves the design itself at both mesh levels and
// subtracts, which is the only way to state its discretization error rather
// than estimate it. It costs seconds, which is why it is a button.
async function measureMeshError() {
  const button = $("mesh-check");
  const note = $("mesh-check-note");
  button.disabled = true;
  button.textContent = "Solving at both mesh levels…";
  note.classList.remove("hidden");
  note.textContent = "This solves the design twice; the finer pass can take half a minute.";
  try {
    const out = await postJSON("/api/mesh-check", { package: packageRequest(), chiplets: chipletRequest() });
    if (out.same_mesh) {
      note.textContent = "This package already meshes at the finest cell the app builds, "
        + "so both levels are the same solve — there is nothing to compare.";
    } else {
      const [coarse, fine] = out.levels;
      note.innerHTML =
        `<b>Measured on this design: ±${out.measured_k.toFixed(3)} K.</b> `
        + `${coarse.cell_mm} mm gives ${coarse.peak_c.toFixed(3)} °C, `
        + `${fine.cell_mm} mm gives ${fine.peak_c.toFixed(3)} °C `
        + `(${coarse.unknowns.toLocaleString()} against ${fine.unknowns.toLocaleString()} unknowns). `
        + `The estimate printed above was ±${out.estimated_k.toFixed(3)} K, `
        + (out.covered ? "which holds." : "which this design exceeds — trust the measurement.");
    }
  } catch (err) {
    note.textContent = `Could not measure it: ${err.message}`;
  } finally {
    button.disabled = false;
    button.textContent = "Measure this design's mesh error";
  }
}

function renderChecks() {
  const box = $("checks");
  const res = state.shown;
  if (!res) { box.innerHTML = ""; return; }
  const v = res.verification, m = res.metrics;
  const energyPct = (Math.abs(v.sink_heat_w - v.specified_w) / Math.max(v.specified_w, 1e-12)) * 100;
  const items = [
    [energyPct < 1e-3, "Energy balance",
      `${formatW(v.specified_w)} W generated → ${formatW(v.sink_heat_w)} W leaves through the cooled face (${energyPct.toExponential(1)} % apart)`],
    [v.residual < 1e-8, "Equations solved", `residual ${v.residual.toExponential(1)} (must be below 1e-8)`],
    [m.floor_ratio == null || m.floor_ratio >= 1, "Physically bounded",
      `rise ${m.rise_k.toFixed(1)} K is above the cooling floor P/(hA) = ${m.floor_rise_k.toFixed(1)} K`],
  ];
  // A source smaller than a cell is averaged over the cell, so the peak
  // stops depending on the design: a 0.5 mm hotspot and a 0.1 mm one report
  // the same temperature. That is not a band, it is a number we cannot
  // defend, so it sits with the other checks that can fail.
  const r = res.resolution;
  if (r && r.applies) {
    const detail = r.resolved
      ? `smallest internal feature is ${r.feature_mm.toFixed(2)} mm on ${r.chiplet}, `
        + `${r.cells_across.toFixed(1)} cells across`
      : `${r.chiplet}'s smallest feature is ${r.feature_mm.toFixed(2)} mm, only `
        + `${r.cells_across.toFixed(1)} cells across — its power is averaged over the cell, `
        + `so this peak is a LOWER BOUND, not a converged answer. `
        + (r.reachable
           ? `Resolving it needs ${r.needed_cell_mm.toFixed(2)} mm cells.`
           : `Resolving it would need ${r.needed_cell_mm.toFixed(3)} mm cells, finer than this app builds.`);
    items.push([!!r.resolved, "Hotspots resolved", detail]);
  }
  const u = m.uncertainty;
  if (u) {
    items.push([true, "Error budget",
      `±${u.combined_k.toFixed(2)} K (${u.pct_of_rise.toFixed(1)} % of the rise): mesh ±${u.mesh_k.toFixed(2)} K `
      + `at ${u.cell_mm} mm cells`
      + (u.material_k === null ? "" : `, conductivities ±${u.material_k.toFixed(2)} K`)
      + (u.boundary_k == null ? "" : `, cooling coefficient ±${u.boundary_k.toFixed(2)} K`)
      + (u.resolution_limited
         ? `. The mesh half of this does NOT apply: a feature smaller than a cell is not `
           + `refined toward an answer, so the band understates it — see Hotspots resolved.`
         : `. Estimated from reference packages — press “Measure this design’s mesh error” `
           + `to replace the mesh half with this design's own.`)]);
  }
  box.innerHTML = items.map(([ok, title, detail]) =>
    `<li><span class="${ok ? "ok" : "no"}">${ok ? "✓" : "✗"}</span><span>${title}<small>${detail}</small></span></li>`).join("")
    + `<li><span class="muted">·</span><span>Full 3-D model<small>${v.unknowns.toLocaleString()} cells · ${res.grid.nx} × ${res.grid.ny} × ${res.grid.nz} · solved in ${res.solve_ms.toFixed(0)} ms</small></span></li>`;
}

function renderChipletList() {
  const ul = $("chiplet-list");
  const temps = new Map(((state.shown && state.shown.chiplets) || []).map((r) => [r.id, r.max_c]));
  ul.innerHTML = "";
  for (const c of state.chiplets) {
    const li = document.createElement("li");
    if (c.id === state.selectedId) li.className = "selected";
    li.innerHTML = `<span class="swatch" style="background:${c.color}"></span>
      <div><div class="c-name"></div><div class="c-meta"></div></div><span class="c-temp"></span>`;
    li.querySelector(".c-name").textContent = c.name;
    li.querySelector(".c-meta").textContent = `${c.w_mm} × ${c.h_mm} mm · ${formatW(c.power_w)} W`;
    li.querySelector(".c-temp").textContent = temps.has(c.id) ? `${temps.get(c.id).toFixed(1)} °C` : "";
    li.addEventListener("click", () => select(c.id));
    ul.appendChild(li);
  }
  const total = state.chiplets.reduce((s, c) => s + c.power_w, 0);
  $("power-total").textContent = `${state.chiplets.length} · ${formatW(total)} W`;
  renderEditor();
  const selected = ul.querySelector("li.selected");
  if (selected) selected.scrollIntoView({ block: "nearest" });
}

// The imported map, turned 90 degrees with its chiplet.
function rotateGrid({ cols, rows, weights }) {
  const turned = [];
  for (let c = cols - 1; c >= 0; c--) {
    for (let r = 0; r < rows; r++) turned.push(weights[r * cols + c]);
  }
  return { cols: rows, rows: cols, weights: turned };
}

function renderMapPreview(c) {
  const canvas = $("ed-map"), clear = $("ed-map-clear"), note = $("ed-map-note");
  const grid = c && c.power_grid;
  canvas.classList.toggle("hidden", !grid);
  clear.classList.toggle("hidden", !grid);
  $("ed-pattern").disabled = !!grid;
  if (!grid) {
    note.textContent = "Have the real numbers? Import a power map from a CSV.";
    return;
  }
  drawPowerMap(canvas, grid, (t) => tempColor(t, 0, 1));
  note.textContent = `${grid.cols} × ${grid.rows} tiles${c.map_source ? ` from ${c.map_source}` : ""}`
    + " · the top of the picture is the top of the chiplet";
}

function renderPatternOptions() {
  const names = Object.keys(state.patterns);
  $("ed-pattern").innerHTML = names
    .map((name) => `<option value="${name}">${escapeHtml(PATTERN_LABELS[name] || name)}</option>`)
    .join("");
}

// What the hottest region of this chiplet actually runs at, per cm^2 -- the
// number a junction follows, which the chiplet average hides.
function hottestDensity(c) {
  const blocks = blocksOf(c);
  const area_cm2 = c.w_mm * c.h_mm / 100;
  if (!area_cm2) return 0;
  if (c.power_grid) {
    const { cols, rows, weights } = c.power_grid;
    const total = weights.reduce((a, b) => a + b, 0);
    if (total <= 0) return c.power_w / area_cm2;
    return c.power_w / area_cm2 * (Math.max(...weights) / total) * cols * rows;
  }
  let factor = 1;
  if (blocks.length) {
    const rest = 1 - blocks.reduce((sum, b) => sum + b[4], 0);
    factor = rest;
    for (const [, , w, h, share] of blocks) factor = Math.max(factor, share / (w * h));
  }
  return c.power_w / area_cm2 * factor;
}

function renderEditor() {
  const c = state.chiplets.find((q) => q.id === state.selectedId);
  $("editor-empty").classList.toggle("hidden", !!c);
  $("editor-form").classList.toggle("hidden", !c);
  if (!c) return;
  const cell = cellMm(state.pkg);
  const set = (id, value) => { const el = $(id); if (document.activeElement !== el) el.value = value; };
  set("ed-name", c.name);
  $("ed-fixed").checked = !!c.fixed;
  set("ed-w", +c.w_mm.toFixed(3));
  set("ed-h", +c.h_mm.toFixed(3));
  set("ed-p", +c.power_w.toFixed(4));
  for (const [id, span] of [["ed-w", state.pkg.size_mm], ["ed-h", sizeYOf(state.pkg)]]) {
    $(id).step = cell; $(id).min = cell; $(id).max = span;
  }
  $("ed-pos").textContent = `${c.x_mm.toFixed(1)}, ${c.y_mm.toFixed(1)} mm`;
  const pattern = c.power_pattern || "uniform";
  if (!$("ed-pattern").querySelector(`option[value="${pattern}"]`)) {
    const option = document.createElement("option");
    option.value = pattern;
    option.textContent = PATTERN_LABELS[pattern] || pattern;
    $("ed-pattern").appendChild(option);
  }
  $("ed-pattern").value = pattern;
  renderMapPreview(c);
  const average = c.power_w / (c.w_mm * c.h_mm / 100);
  const hottest = hottestDensity(c);
  $("ed-density").textContent = hottest > average * 1.01
    ? `${average.toFixed(0)} W/cm² average · ${hottest.toFixed(0)} W/cm² in the hot region`
    : `${average.toFixed(0)} W/cm² across the footprint`;
}

// ------------------------------------------------------------ interactions
function select(id) {
  state.selectedId = id;
  renderChipletList();
  renderFloorplan();
}

function moveChiplet(id, x, y, dragging) {
  const c = state.chiplets.find((q) => q.id === id);
  if (!c) return;
  if (dragging) {
    if (!state.dragging) { record(); state.dragging = true; }
  } else if (state.dragging) {
    state.dragging = false;          // end of a drag: already recorded
  } else {
    record();                        // a discrete move (arrow key)
  }
  c.x_mm = x;
  c.y_mm = y;
  renderFloorplan(violations().ids);
  if (!dragging) renderChipletList();
  requestSolve();
}

function fitInside(c) {
  const size = state.pkg.size_mm, sizeY = sizeYOf(state.pkg), cell = cellMm(state.pkg);
  c.w_mm = Math.min(size, Math.max(cell, Math.round(c.w_mm / cell) * cell));
  c.h_mm = Math.min(sizeY, Math.max(cell, Math.round(c.h_mm / cell) * cell));
  c.x_mm = Math.max(0, Math.min(size - c.w_mm, Math.round(c.x_mm / cell) * cell));
  c.y_mm = Math.max(0, Math.min(sizeY - c.h_mm, Math.round(c.y_mm / cell) * cell));
}

function rotateSelected() {
  const c = state.chiplets.find((q) => q.id === state.selectedId);
  if (!c) return;
  record();
  [c.w_mm, c.h_mm] = [c.h_mm, c.w_mm];
  // A point (u, v) of the old footprint lands at (1 - v, u) of the new one.
  if (c.blocks && c.blocks.length) {
    c.blocks = c.blocks.map(([x, y, w, h, share]) => [1 - y - h, x, h, w, share]);
  }
  if (c.power_grid) c.power_grid = rotateGrid(c.power_grid);
  fitInside(c);
  renderChipletList();
  requestSolve();
}

function removeSelected() {
  const n = state.chiplets.findIndex((q) => q.id === state.selectedId);
  if (n < 0) return;
  record();
  state.chiplets.splice(n, 1);
  state.selectedId = null;
  renderChipletList();
  requestSolve();
}

function overlapsAny(c) {
  return state.chiplets.some((o) => o.id !== c.id
    && Math.min(c.x_mm + c.w_mm, o.x_mm + o.w_mm) - Math.max(c.x_mm, o.x_mm) > 1e-9
    && Math.min(c.y_mm + c.h_mm, o.y_mm + o.h_mm) - Math.max(c.y_mm, o.y_mm) > 1e-9);
}

function addChiplet(key) {
  const tpl = LIBRARY.find((l) => l.key === key);
  if (!tpl) return;
  if (state.chiplets.length >= MAX_CHIPLETS) { toast(`At most ${MAX_CHIPLETS} chiplets`); return; }
  const size = state.pkg.size_mm, sizeY = sizeYOf(state.pkg);
  let n = 1;
  while (state.chiplets.some((c) => c.id === `${key}${n}`)) n++;
  const c = { id: `${key}${n}`, name: `${tpl.name} ${n}`, w_mm: tpl.w_mm, h_mm: tpl.h_mm,
              power_w: tpl.power_w, x_mm: 0, y_mm: 0, color: tpl.color,
              power_pattern: tpl.power_pattern || "uniform" };
  const step = Math.max(cellMm(state.pkg), state.pkg.pitch_mm);
  for (let y = 0; y + c.h_mm <= size + 1e-9; y += step) {
    for (let x = 0; x + c.w_mm <= size + 1e-9; x += step) {
      c.x_mm = x; c.y_mm = y;
      if (!overlapsAny(c)) {
        record();
        state.chiplets.push(c);
        select(c.id);
        requestSolve();
        const density = c.power_w / (c.w_mm * c.h_mm / 100);
        toast(`${c.name} added: ${formatW(c.power_w)} W over ${c.w_mm} × ${c.h_mm} mm `
          + `= ${density.toFixed(0)} W/cm². Set its power in the editor if that is not your part.`, "info");
        return;
      }
    }
  }
  toast(`There is no free space for another ${tpl.name}. Remove or move a chiplet first.`);
}

function clearOptimization() {
  state.opt = null;
  state.burst = null;          // a different design means a different curve
  if (state.tab === "timeline") renderTimeline();
  $("opt-card").classList.add("hidden");
  $("tab-compare-btn").disabled = true;
  for (const b of document.querySelectorAll("#gl-mode button")) b.disabled = b.dataset.mode !== "current";
  setGlMode("current");
  if (state.tab === "compare") switchTab("floorplan");
}

function syncPackageControls() {
  const p = state.pkg;
  $("pkg-size").value = String(p.size_mm);
  $("pkg-ambient").value = p.ambient_c;
  $("pkg-path").value = p.cooling || "direct";
  $("pkg-interposer").value = p.interposer_material || "silicon";
  $("pkg-beol").value = String(p.beol_um || 0);
  $("pkg-mesh").value = p.mesh || "standard";
  $("pkg-size-y").value = sizeYOf(p);
  const preset = COOLING_PRESETS.includes(String(p.h_w_m2k)) ? String(p.h_w_m2k) : "custom";
  $("pkg-cooling").value = preset;
  $("pkg-h-wrap").classList.toggle("hidden", preset !== "custom");
  $("pkg-h").value = p.h_w_m2k;
  $("pkg-pitch").value = String(p.pitch_mm);
  $("pkg-keepout").value = p.keepout_um !== undefined ? p.keepout_um : DEFAULT_KEEPOUT_UM;
  const stack = stackOf(p);
  $("pkg-int").value = stack[0];
  $("pkg-bump").value = stack[1];
  $("pkg-die").value = stack[2];
  renderStackStrip();
}

// The strip is the stack to scale, so a 300 µm interposer looks like one.
function renderStackStrip() {
  const [interposer, bumps, die] = stackOf(state.pkg);
  $("stack-strip").innerHTML =
    `<span style="flex:${interposer}" class="s-int">Interposer ${interposer} µm</span>`
    + `<span style="flex:${bumps}" class="s-bump">Bumps ${bumps} µm</span>`
    + `<span style="flex:${die}" class="s-die">Die ${die} µm</span>`;
  const cell = cellMm(state.pkg), nz = nzOf(state.pkg);
  const n = Math.round(state.pkg.size_mm / cell);
  const ny = Math.round(sizeYOf(state.pkg) / cell);
  const keepout = keepoutMm(state.pkg);
  const requested = state.pkg.keepout_um !== undefined ? state.pkg.keepout_um : DEFAULT_KEEPOUT_UM;
  const keepoutNote = keepout > 0
    ? ` · keep-out ${requested} µm → ${(keepout * 1000).toFixed(0)} µm on this mesh`
    : " · no keep-out";
  // The error budget belongs to the checks panel, which is rebuilt with each
  // result. Repeating it here showed the PREVIOUS solve's band beside the new
  // mesh whenever the mesh changed, which is worse than not saying it.
  $("stack-note").textContent =
    `Mesh ${n} × ${ny} × ${nz} = ${(n * ny * nz).toLocaleString()} cells · ${cell} mm × ${cell} mm × 20 µm`
    + keepoutNote;
}

function loadScenario(sc) {
  $("scenario").title = sc.summary || "";
  state.scenarioId = sc.id;
  state.pkg = { stack_um: [...DEFAULT_STACK_UM], keepout_um: DEFAULT_KEEPOUT_UM,
                cooling: "direct", mesh: "standard", ...sc.package };
  state.chiplets = sc.chiplets.map((c) => ({ ...c }));
  state.selectedId = null;
  state.shown = null;
  history.clear();
  clearOptimization();
  syncPackageControls();
  renderAll();
  requestSolve();
}

function designTitle() {
  if (state.opened && !state.scenarioId) return state.opened.title;
  const sel = $("scenario");
  return sel.selectedIndex >= 0 ? sel.options[sel.selectedIndex].textContent : "Custom design";
}

// An opened file becomes the entry the scenario list shows, so "Reset" has
// something to go back to and the report knows what it is looking at.
function showOpenedInScenarioList(title) {
  const sel = $("scenario");
  let option = sel.querySelector('option[value="__opened"]');
  if (!option) {
    option = document.createElement("option");
    option.value = "__opened";
    sel.appendChild(option);
  }
  option.textContent = title;
  sel.value = "__opened";
  sel.title = "Opened from a file";
}

function resetDesign() {
  if (state.scenarioId) {
    const sc = state.scenarios.find((s) => s.id === state.scenarioId);
    if (sc) { loadScenario(sc); return; }
  }
  if (!state.opened) return;
  const { pkg, chiplets, title } = state.opened;
  state.pkg = { ...pkg };
  state.chiplets = chiplets.map((c) => ({ ...c }));
  state.selectedId = null;
  state.shown = null;
  history.clear();
  clearOptimization();
  syncPackageControls();
  showOpenedInScenarioList(title);
  renderAll();
  requestSolve();
}

function packageChanged(physics = true) {
  for (const c of state.chiplets) fitInside(c);
  // A taller stack can force a coarser lateral mesh, which can in turn make
  // the placement grid finer than the mesh; keep the two consistent.
  const cell = cellMm(state.pkg);
  if (state.pkg.pitch_mm < cell) state.pkg.pitch_mm = cell;
  syncPackageControls();
  if (physics) { clearOptimization(); state.shown = null; }
  renderAll();
  requestSolve();
}

// ------------------------------------------------------------ optimization
function setBusy(busy) {
  state.busy = busy;
  for (const id of ["btn-simulate", "btn-optimize", "scenario", "btn-add",
                    "btn-open", "btn-reset", "ed-rotate", "ed-delete"]) $(id).disabled = busy;
  updateHistoryButtons();
  renderFloorplan();
}

// Rough cost of the proof, from the work it must do: one solve per
// (chiplet, grid position), measured at ~17 ms per solve on a 19,200-cell
// model. The search after that is not predictable, which the warning says.
function estimateProof() {
  const cell = cellMm(state.pkg), size = state.pkg.size_mm, pitch = state.pkg.pitch_mm;
  const sizeY = sizeYOf(state.pkg);
  const unknowns = (size / cell) * (sizeY / cell) * 12;
  let solves = 0, combos = 1;
  for (const c of state.chiplets) {
    const nx = Math.floor((size - c.w_mm) / pitch) + 1;
    const ny = Math.floor((sizeY - c.h_mm) / pitch) + 1;
    const n = Math.max(0, nx) * Math.max(0, ny);
    solves += n;
    combos *= Math.max(1, n);
  }
  return { solves, combos, precompute_s: solves * 0.017 * (unknowns / 19200) };
}

async function optimize() {
  if (state.busy || state.animating) return;
  const bad = violations();
  if (!state.chiplets.length) { toast("Add chiplets first"); return; }
  if (bad.messages.length) { toast(`Fix the placement first: ${bad.messages[0]}`); return; }

  const cost = estimateProof();
  if (cost.precompute_s > 10) {
    const coarser = state.pkg.pitch_mm < 2;
    const answer = await ask({
      title: "This proof will take a while",
      body: `A ${state.pkg.pitch_mm} mm placement grid means ${cost.solves.toLocaleString()} single-chiplet `
        + `solves (about ${Math.round(cost.precompute_s)} s) before the search over `
        + `${compact(cost.combos)} layouts even starts, and the search itself can take longer. `
        + (coarser ? "A 2 mm grid answers in a few seconds." : ""),
      ok: "Run it anyway",
      alt: coarser ? "Use the 2 mm grid" : null,
    });
    if (!answer) return;
    if (answer === "alt") {
      record();
      state.pkg.pitch_mm = 2;
      syncPackageControls();
      clearOptimization();
      renderFloorplan();
    }
  }
  setBusy(true);
  for (const id of ["opt-best", "opt-combos", "opt-leaves", "opt-elapsed"]) $(id).textContent = "–";
  $("opt-stage").textContent = "Starting…";
  $("opt-overlay").classList.remove("hidden");
  setStatus("busy", "Optimizing…");
  try {
    const res = await postJSON("/api/optimize", { package: packageRequest(), chiplets: chipletRequest() });
    if (!res.ok) throw new Error(res.error || "could not start");
    state.jobId = res.job_id;
    pollJob();
  } catch (err) {
    $("opt-overlay").classList.add("hidden");
    setBusy(false);
    toast(`Optimization failed: ${err.message}`);
    setStatus("error", "Optimization failed");
  }
}

function stageText(p) {
  if (!p.stage) return "Working…";
  let text = p.stage.charAt(0).toUpperCase() + p.stage.slice(1);
  if (p.total) text += ` (${p.done} of ${p.total} chiplets, ${p.solves} full solves)`;
  if (p.nodes) text += ` · ${p.nodes.toLocaleString()} branches explored`;
  return text;
}

async function pollJob() {
  const id = state.jobId;
  if (!id) return;
  let job;
  try {
    const res = await getJSON(`/api/jobs/${id}`);
    if (!res.ok) throw new Error(res.error || "job lost");
    job = res.job;
  } catch (err) {
    $("opt-overlay").classList.add("hidden");
    state.jobId = null;
    setBusy(false);
    toast(`Optimization failed: ${err.message}`);
    return;
  }
  const p = job.progress || {};
  $("opt-stage").textContent = stageText(p);
  const paused = job.state === "paused", pausing = job.state === "pausing";
  state.jobPaused = paused || pausing;
  $("btn-pause").textContent = paused ? "Resume" : (pausing ? "Pausing…" : "Pause");
  $("btn-pause").disabled = pausing;
  if (p.best_c != null) $("opt-best").textContent = `${p.best_c.toFixed(2)} °C`;
  if (p.combinations) $("opt-combos").textContent = compact(p.combinations);
  if (p.leaves != null) $("opt-leaves").textContent = p.leaves.toLocaleString();
  $("opt-elapsed").textContent = `${(p.elapsed_s || 0).toFixed(1)} s`;
  // A held job is still a live job: keep polling, or the overlay falls
  // through to the completion branch and reports a finish that never happened.
  if (job.state === "running" || paused || pausing) { setTimeout(pollJob, 250); return; }

  $("opt-overlay").classList.add("hidden");
  state.jobId = null;
  setBusy(false);
  if (job.state === "done") {
    onOptimized(job.result);
  } else {
    toast(job.error || "Optimization stopped");
    setStatus(job.state === "cancelled" ? "ok" : "error",
      job.state === "cancelled" ? "Optimization stopped" : "Optimization failed");
  }
}

function onOptimized(result) {
  decodeResult(result.initial);
  decodeResult(result.optimized);
  state.opt = result;
  const from = withColours(result.initial.placement);
  const to = withColours(result.optimized.placement);
  switchTab("floorplan");
  state.shown = result.initial;
  state.selectedId = null;
  renderAll();
  animateMove(from, to, result.kept_initial ? 10 : 1600, () => {
    state.chiplets = to;
    state.shown = result.optimized;
    renderAll();
    renderOptSummary();
    $("tab-compare-btn").disabled = false;
    for (const b of document.querySelectorAll("#gl-mode button")) b.disabled = false;
    const cert = result.certificate;
    setStatus("ok", result.kept_initial
      ? "Already the coolest layout on the grid"
      : `Optimized: −${result.improvement.delta_k.toFixed(1)} K (proven over ${compact(cert.combinations)} layouts)`);
  });
}

function animateMove(from, to, ms, done) {
  // Browsers pause animation frames in hidden tabs; never let a result wait
  // on an animation nobody can see.
  if (document.hidden) { state.animating = false; done(); return; }
  const t0 = performance.now();
  state.animating = true;
  const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
  const frame = (now) => {
    const t = Math.min(1, (now - t0) / ms);
    const e = ease(t);
    state.chiplets = from.map((c, n) => ({ ...c, x_mm: c.x_mm + (to[n].x_mm - c.x_mm) * e,
                                                 y_mm: c.y_mm + (to[n].y_mm - c.y_mm) * e }));
    renderFloorplan();
    if (t < 1) { requestAnimationFrame(frame); return; }
    state.animating = false;
    done();
  };
  requestAnimationFrame(frame);
}

function renderOptSummary() {
  const r = state.opt;
  if (!r) return;
  const imp = r.improvement, cert = r.certificate;
  $("opt-card").classList.remove("hidden");
  $("opt-summary").innerHTML = r.kept_initial
    ? `<div class="big">Already optimal</div>
       <div>No layout on the ${cert.pitch_mm} mm grid runs cooler than this one (${compact(cert.combinations)} layouts covered).</div>`
    : `<div class="big">−${imp.delta_k.toFixed(1)} K</div>
       <div>Peak ${imp.initial_peak_c.toFixed(1)} °C → <b>${imp.optimized_peak_c.toFixed(1)} °C</b>,
       ${imp.pct_of_rise.toFixed(0)}% of the temperature rise removed</div>
       <div class="muted">Proven coolest of ${compact(cert.combinations)} layouts on the ${cert.pitch_mm} mm grid · ${cert.total_s.toFixed(1)} s</div>`;
}

function showPlacement(which) {
  const r = state.opt;
  if (!r || state.animating) return;
  const part = which === "initial" ? r.initial : r.optimized;
  state.chiplets = withColours(part.placement);
  state.shown = part;
  renderAll();
}

// ----------------------------------------------------------------- compare
function renderCompare() {
  const r = state.opt;
  if (!r) return;
  const a = r.initial, b = r.optimized;
  const lo = Math.floor(Math.min(a.t_min_c, b.t_min_c));
  const hi = Math.ceil(Math.max(a.t_max_c, b.t_max_c));
  const delta = new Float32Array(a.colmax.length);
  let span = 0;
  for (let n = 0; n < delta.length; n++) {
    delta[n] = b.colmax[n] - a.colmax[n];
    span = Math.max(span, Math.abs(delta[n]));
  }
  span = Math.max(0.1, Math.ceil(span * 10) / 10);
  const g = a.grid;

  drawColorbar($("cmp-colorbar"), "temp");
  drawColorbar($("cmp-dcolorbar"), "delta");
  $("cmp-cb-min").textContent = `${lo} °C`;
  $("cmp-cb-max").textContent = `${hi} °C`;
  $("cmp-d-min").textContent = `−${span.toFixed(1)} K`;
  $("cmp-d-max").textContent = `+${span.toFixed(1)} K`;

  const imp = r.improvement, cert = r.certificate;
  $("compare-banner").innerHTML = r.kept_initial
    ? `<b>Already optimal.</b> No layout on the ${cert.pitch_mm} mm grid is cooler than the starting placement.`
    : `<b>Peak ${imp.initial_peak_c.toFixed(2)} °C → ${imp.optimized_peak_c.toFixed(2)} °C (−${imp.delta_k.toFixed(2)} K)</b>
       &nbsp;·&nbsp; ${imp.pct_of_rise.toFixed(1)}% of the rise above ambient removed by moving chiplets only:
       same chiplets, same power, same package and cooling.`;

  $("moves-table").querySelector("tbody").innerHTML = r.moves.map((m) => {
    const moved = m.distance_mm > 1e-9 ? `${m.distance_mm.toFixed(1)} mm` : '<span class="muted">stays</span>';
    const change = m.max_after_c - m.max_before_c;
    const cls = change < -0.05 ? "good" : change > 0.05 ? "bad" : "";
    return `<tr><td>${escapeHtml(m.name)}</td>
      <td>${m.from_mm[0].toFixed(1)}, ${m.from_mm[1].toFixed(1)}</td>
      <td>${m.to_mm[0].toFixed(1)}, ${m.to_mm[1].toFixed(1)}</td>
      <td>${moved}</td>
      <td>${m.max_before_c.toFixed(1)} → <span class="${cls}">${m.max_after_c.toFixed(1)} °C</span></td></tr>`;
  }).join("");

  const identical = cert.identical_chiplets ? " Identical chiplets are interchangeable, so duplicates were counted once." : "";
  const stat = (label, value, title = "") => `<div title="${title}"><span>${label}</span><b>${value}</b></div>`;
  $("certificate").innerHTML = `
    <div class="badge">EXACT · PROVEN OPTIMUM</div>
    <p>All layouts on the ${cert.pitch_mm} mm grid are covered. A branch is skipped only when a proven lower bound
    shows it is hotter than the best layout found (a chiplet can only add heat).${identical}</p>
    <div class="cert-stats">
      ${stat("Layouts on the grid", cert.combinations.toLocaleString())}
      ${stat("Scored exactly", cert.layouts_scored.toLocaleString(), "Complete layouts evaluated by adding full 3-D solutions")}
      ${stat("Ruled out by bound", cert.excluded_bound.toLocaleString(), "Branches whose proven lower bound already exceeded the best layout")}
      ${stat("Ruled out by overlap", cert.excluded_overlap.toLocaleString(), "Candidate positions where chiplets would overlap")}
      ${stat("Re-solve check", `${cert.resolve_delta_k.toExponential(1)} K`, "Winner re-solved from scratch with the full solver; difference from the search value")}
      ${stat("Time", `${cert.total_s.toFixed(1)} s`, `${cert.unit_solves} single-chiplet solves ${cert.precompute_s.toFixed(1)} s · search ${cert.search_s.toFixed(1)} s`)}
    </div>`;

  // Maps last: the text above changes the layout, and the maps must be sized to the final layout.
  const panel = (canvasId, part, heat, showTemps) => {
    const canvas = $(canvasId);
    const { ctx, width, height } = setupCanvas(canvas);
    drawPackageMap(ctx, width, height, {
      sizeMm: g.size_mm, sizeYMm: g.size_y_mm || g.size_mm, compact: true, heat,
      chiplets: withColours(part.placement),
      temps: showTemps ? new Map(part.chiplets.map((c) => [c.id, c.max_c])) : null,
      peak: showTemps ? part.peak : null,
    });
  };
  panel("cmp-initial", a, { values: a.colmax, nx: g.nx, ny: g.ny, lo, hi, kind: "temp" }, true);
  panel("cmp-optimized", b, { values: b.colmax, nx: g.nx, ny: g.ny, lo, hi, kind: "temp" }, true);
  panel("cmp-delta", b, { values: delta, nx: g.nx, ny: g.ny, lo: -span, hi: span, kind: "delta" }, false);
}

// ---------------------------------------------------------------------- 3D
function ensureViewer() {
  if (state.viewer) return state.viewer;
  try {
    state.viewer = new Viewer3D($("gl"), $("gl-labels"), $("gl-tooltip"));
    state.viewer.onSpinChange = (on) => $("gl-spin").classList.toggle("active", on);
  } catch (err) {
    toast(err.message);
    return null;
  }
  return state.viewer;
}

function setGlMode(mode) {
  state.glMode = mode;
  for (const b of document.querySelectorAll("#gl-mode button")) b.classList.toggle("active", b.dataset.mode === mode);
  if (state.tab === "view3d") update3D();
}

function update3D() {
  const viewer = ensureViewer();
  if (!viewer) return;
  const r = state.opt;
  let res = state.shown, kind = "temp", values, lo, hi, chiplets = state.chiplets, peak;
  if (state.glMode !== "current" && r) {
    res = state.glMode === "initial" ? r.initial : r.optimized;
    chiplets = withColours(res.placement);
  }
  if (!res) return;
  const g = res.grid;
  if (state.glMode === "delta" && r) {
    kind = "delta";
    values = new Float32Array(r.optimized.field.length);
    let span = 0;
    for (let n = 0; n < values.length; n++) {
      values[n] = r.optimized.field[n] - r.initial.field[n];
      span = Math.max(span, Math.abs(values[n]));
    }
    hi = Math.max(0.1, Math.ceil(span * 10) / 10);
    lo = -hi;
    peak = null;
    $("gl-cb-min").textContent = `−${hi.toFixed(1)} K`;
    $("gl-cb-max").textContent = `+${hi.toFixed(1)} K`;
    $("gl-cb-label").textContent = "Temperature change after optimization (K), blue = cooler · height exaggerated";
    drawColorbar($("gl-colorbar"), "delta");
  } else {
    values = res.field;
    if (r && state.glMode !== "current") {
      lo = Math.floor(Math.min(r.initial.t_min_c, r.optimized.t_min_c));
      hi = Math.ceil(Math.max(r.initial.t_max_c, r.optimized.t_max_c));
    } else {
      lo = Math.floor(res.t_min_c);
      hi = Math.ceil(res.t_max_c);
    }
    peak = { ...res.peak };
    $("gl-cb-min").textContent = `${lo} °C`;
    $("gl-cb-max").textContent = `${hi} °C`;
    $("gl-cb-label").textContent = state.glMode === "current"
      ? "Temperature (°C) · height exaggerated for visibility"
      : "Temperature (°C), same scale for before and after · height exaggerated";
    drawColorbar($("gl-colorbar"), "temp");
  }
  const peel = $("gl-peel"), cut = $("gl-cut");
  peel.max = g.nz - 1;
  cut.max = g.ny - 1;
  viewer.setData({
    values, nx: g.nx, ny: g.ny, nz: g.nz, cell_mm: g.cell_mm, dz_um: g.dz_um,
    layers: g.layers, dieStart: g.die_k0 !== undefined ? g.die_k0 : g.layers[g.layers.length - 1].k0,
    chiplets, kind, lo, hi, peak,
  });
}

// --------------------------------------------------------------- over time
function renderTimeline() {
  const canvas = $("timeline");
  if (!state.burst) {
    const { ctx, width, height } = setupCanvas(canvas);
    ctx.clearRect(0, 0, width, height);
    return;
  }
  const colours = new Map(state.chiplets.map((c) => [c.id, { color: c.color, name: c.name }]));
  drawTimeline(canvas, state.burst, colours);
  renderBurstStats();
}

function renderBurstStats() {
  const r = state.burst;
  if (!r) { $("tr-stats").innerHTML = ""; return; }
  const ms = (seconds) => (seconds == null ? "–"
    : seconds < 1 ? `${Math.round(seconds * 1000)} ms` : `${seconds.toFixed(2)} s`);
  const reached = Math.round((r.reached_fraction || 0) * 100);
  const limitCell = r.limit_c == null ? ["Limit", "not set", ""]
    : r.time_to_limit_s == null
      ? [`Stays under ${r.limit_c} °C`, "for this burst", "good-value"]
      : [`Reaches ${r.limit_c} °C`, ms(r.time_to_limit_s), "warn-value"];
  const stat = (label, value, cls = "") =>
    `<div><span>${label}</span><b class="${cls}">${value}</b></div>`;
  $("tr-stats").innerHTML =
    stat("Idle", `${r.idle_peak_c.toFixed(1)} °C`)
    + stat("Peak in the burst", `${r.burst_peak_c.toFixed(1)} °C`)
    + stat("Steady state would be", `${r.steady_peak_c.toFixed(1)} °C`)
    + stat("Of that rise, reached", `${reached} %`)
    + stat(limitCell[0], limitCell[1], limitCell[2])
    + stat("Hottest point, 63 % of its rise", ms(r.t63_peak_s))
    + stat("Package time constant", ms(r.tau_package_s))
    + stat("Time step", `${(r.dt_s * 1000).toFixed(1)} ms · ${r.t_s.length} points`)
    + stat("Time-step artefact", `≤ ${(r.step_artefact_k * 1000).toFixed(0)} mK`);
}

async function runBurst() {
  if (state.busy || state.animating) return;
  const bad = violations();
  if (!state.chiplets.length) { toast("Add chiplets first"); return; }
  if (bad.messages.length) { toast(`Fix the placement first: ${bad.messages[0]}`); return; }
  const limit = Number($("tr-limit").value);
  const profile = {
    burst_s: Number($("tr-burst").value),
    cool_s: Number($("tr-burst").value),
    idle_fraction: Number($("tr-idle").value),
    limit_c: Number.isFinite(limit) ? limit : null,
  };
  setBusy(true);
  $("btn-burst").disabled = true;
  $("tr-status").textContent = "Stepping…";
  setStatus("busy", "Stepping through the burst…");
  try {
    const res = await postJSON("/api/transient", {
      package: packageRequest(), chiplets: chipletRequest(), transient: profile,
    });
    if (!res.ok) throw new Error(res.error || "could not start");
    state.burstJob = res.job_id;
    pollBurst();
  } catch (err) {
    setBusy(false);
    $("btn-burst").disabled = false;
    $("tr-status").textContent = "";
    toast(`Could not run the burst: ${err.message}`);
    setStatus("error", "Burst failed");
  }
}

async function pollBurst() {
  const id = state.burstJob;
  if (!id) return;
  let job;
  try {
    const res = await getJSON(`/api/jobs/${id}`);
    if (!res.ok) throw new Error(res.error || "job lost");
    job = res.job;
  } catch (err) {
    state.burstJob = null;
    setBusy(false);
    $("btn-burst").disabled = false;
    $("tr-status").textContent = "";
    toast(`Burst failed: ${err.message}`);
    return;
  }
  const p = job.progress || {};
  // Same rule as the optimizer's overlay: paused and pausing are live states.
  if (job.state === "running" || job.state === "paused" || job.state === "pausing") {
    if (job.state !== "running") {
      $("tr-status").textContent = job.state === "paused" ? "Paused" : "Pausing…";
    } else if (p.steps) {
      $("tr-status").textContent =
        `Stepping… ${p.step} of ${p.steps} · t = ${(p.t_s || 0).toFixed(2)} s · ${(p.peak_c || 0).toFixed(1)} °C`;
    }
    setTimeout(pollBurst, 200);
    return;
  }
  state.burstJob = null;
  setBusy(false);
  $("btn-burst").disabled = false;
  if (job.state === "done" && job.result && job.result.ok) {
    state.burst = job.result;
    $("tr-status").textContent = `Stepped ${job.result.t_s.length} points in ${job.result.solve_s.toFixed(1)} s`;
    setStatus("ok", `Burst: ${job.result.burst_peak_c.toFixed(1)} °C peak`);
    renderTimeline();
  } else if (job.state === "done" && job.result && job.result.violations) {
    $("tr-status").textContent = "";
    toast(job.result.violations[0]);
    setStatus("error", "Placement not valid");
  } else {
    $("tr-status").textContent = "";
    toast(job.error || "Burst stopped");
    setStatus(job.state === "cancelled" ? "ok" : "error",
              job.state === "cancelled" ? "Burst stopped" : "Burst failed");
  }
}

// -------------------------------------------------------------------- tabs
// The comparison always gets the full width; other tabs follow "Full view".
function applyLayout() {
  document.querySelector(".layout").classList.toggle("focus", state.focus || state.tab === "compare");
  $("btn-focus").classList.toggle("active", state.focus);
  $("btn-focus").disabled = state.tab === "compare";
}

function switchTab(name) {
  state.tab = name;
  applyLayout();
  if (name !== "view3d" && state.viewer && state.viewer.spinning) state.viewer.setSpin(false);
  for (const b of document.querySelectorAll(".tab")) b.classList.toggle("active", b.dataset.tab === name);
  $("tab-floorplan").classList.toggle("active", name === "floorplan");
  $("tab-view3d").classList.toggle("active", name === "view3d");
  $("tab-timeline").classList.toggle("active", name === "timeline");
  $("tab-compare").classList.toggle("active", name === "compare");
  requestAnimationFrame(() => {
    if (name === "floorplan") { drawColorbar($("fp-colorbar"), "temp"); renderFloorplan(violations().ids); }
    if (name === "view3d") update3D();
    if (name === "compare") renderCompare();
    if (name === "timeline") renderTimeline();
  });
}

// -------------------------------------------------------------------- init
function wire() {
  floorplan = new FloorplanView($("floorplan"), $("fp-tooltip"), { onSelect: select, onMove: moveChiplet });

  for (const b of document.querySelectorAll(".tab")) b.addEventListener("click", () => { if (!b.disabled) switchTab(b.dataset.tab); });
  $("scenario").addEventListener("change", () => {
    const sc = state.scenarios.find((s) => s.id === $("scenario").value);
    if (sc) loadScenario(sc);
  });
  $("btn-simulate").addEventListener("click", () => requestSolve());
  $("btn-optimize").addEventListener("click", () => optimize());
  $("btn-cancel").addEventListener("click", async () => {
    if (state.jobId) await postJSON(`/api/jobs/${state.jobId}/cancel`, {});
  });
  // A proof that can only be stopped is a proof you have to start again. The
  // hold takes effect at the solver's next safe point, which is why the button
  // reads "Pausing…" until the server confirms the job has actually stopped.
  $("btn-pause").addEventListener("click", async () => {
    if (!state.jobId) return;
    const held = state.jobPaused;
    await postJSON(`/api/jobs/${state.jobId}/${held ? "resume" : "pause"}`, {});
  });
  $("btn-show-initial").addEventListener("click", () => showPlacement("initial"));
  $("btn-show-optimized").addEventListener("click", () => showPlacement("optimized"));

  $("pkg-size").addEventListener("change", () => { record(); state.pkg.size_mm = Number($("pkg-size").value); packageChanged(); });
  $("pkg-size-y").addEventListener("change", () => {
    const v = Number($("pkg-size-y").value);
    if (!Number.isFinite(v)) return;
    record();
    // Equal sides mean "square", and a square package stores no second side,
    // so a design written before rectangles existed still round-trips.
    state.pkg.size_y_mm = (v === state.pkg.size_mm) ? 0 : Math.max(5, Math.min(40, v));
    packageChanged();
  });
  $("pkg-ambient").addEventListener("change", () => {
    const v = Number($("pkg-ambient").value);
    if (Number.isFinite(v)) { record(); state.pkg.ambient_c = Math.max(-40, Math.min(125, v)); packageChanged(); }
  });
  $("pkg-cooling").addEventListener("change", () => {
    const v = $("pkg-cooling").value;
    $("pkg-h-wrap").classList.toggle("hidden", v !== "custom");
    if (v !== "custom") { record(); state.pkg.h_w_m2k = Number(v); packageChanged(); }
    else $("pkg-h").focus();
  });
  $("pkg-h").addEventListener("change", () => {
    const v = Number($("pkg-h").value);
    if (Number.isFinite(v) && v >= 10) { record(); state.pkg.h_w_m2k = v; packageChanged(); }
  });
  const stackField = (id, index) => $(id).addEventListener("change", () => {
    const value = Math.round(Number($(id).value) / 20) * 20;
    if (!Number.isFinite(value) || value < 20 || value > 400) { syncPackageControls(); return; }
    const stack = [...stackOf(state.pkg)];
    stack[index] = value;
    if (stack.reduce((a, b) => a + b, 0) > 600) {
      toast("The stack can total at most 600 µm on this mesh.");
      syncPackageControls();
      return;
    }
    record();
    state.pkg.stack_um = stack;
    packageChanged();
  });
  $("pkg-interposer").addEventListener("change", () => {
    record();
    state.pkg.interposer_material = $("pkg-interposer").value;
    packageChanged();
  });
  $("pkg-beol").addEventListener("change", () => {
    record();
    state.pkg.beol_um = Number($("pkg-beol").value) || 0;
    packageChanged();
  });
  $("pkg-path").addEventListener("change", () => {
    record();
    state.pkg.cooling = $("pkg-path").value;
    packageChanged();
  });
  $("pkg-mesh").addEventListener("change", () => {
    record();
    state.pkg.mesh = $("pkg-mesh").value;
    packageChanged();
  });
  $("mesh-check").addEventListener("click", measureMeshError);
  $("pkg-keepout").addEventListener("change", () => {
    const value = Number($("pkg-keepout").value);
    if (!Number.isFinite(value) || value < 0 || value > 2000) { syncPackageControls(); return; }
    record();
    state.pkg.keepout_um = value;
    clearOptimization();
    syncPackageControls();
    requestSolve();
  });
  stackField("pkg-int", 0);
  stackField("pkg-bump", 1);
  stackField("pkg-die", 2);

  $("pkg-pitch").addEventListener("change", () => {
    record();
    state.pkg.pitch_mm = Number($("pkg-pitch").value);
    clearOptimization();
    renderFloorplan();
  });

  $("library").innerHTML = LIBRARY.map((l) =>
    `<option value="${l.key}">${l.name} · ${l.w_mm} × ${l.h_mm} mm · ${l.power_w} W · `
    + `${(l.power_w / (l.w_mm * l.h_mm / 100)).toFixed(0)} W/cm²</option>`).join("");
  $("btn-add").addEventListener("click", () => addChiplet($("library").value));

  const editNumber = (id, apply) => $(id).addEventListener("change", () => {
    const c = state.chiplets.find((q) => q.id === state.selectedId);
    const v = Number($(id).value);
    if (!c || !Number.isFinite(v)) return;
    record();
    apply(c, v);
    fitInside(c);
    renderChipletList();
    requestSolve();
  });
  editNumber("ed-w", (c, v) => { c.w_mm = v; });
  editNumber("ed-h", (c, v) => { c.h_mm = v; });
  editNumber("ed-p", (c, v) => { c.power_w = Math.max(0, Math.min(2000, v)); });
  $("ed-fixed").addEventListener("change", () => {
    const c = state.chiplets.find((q) => q.id === state.selectedId);
    if (!c) return;
    record();
    c.fixed = $("ed-fixed").checked;
    renderChipletList();
    renderFloorplan();
  });
  $("ed-name").addEventListener("change", () => {
    const c = state.chiplets.find((q) => q.id === state.selectedId);
    if (c && $("ed-name").value.trim()) { record(); c.name = $("ed-name").value.trim(); renderChipletList(); renderFloorplan(); }
  });
  $("ed-import").addEventListener("click", async () => {
    const c = state.chiplets.find((q) => q.id === state.selectedId);
    if (!c) { toast("Select a chiplet first."); return; }
    let imported;
    try {
      imported = await readPowerMapFile($("file-map"), c);
    } catch (err) {
      toast(err.message);
      return;
    }
    if (!imported) return;
    record();
    c.power_grid = imported.grid;
    c.power_pattern = "imported";
    c.map_source = imported.filename;
    delete c.blocks;
    renderChipletList();
    renderFloorplan();
    requestSolve();
    const tiles = `${imported.grid.cols} × ${imported.grid.rows} tiles`;
    const shape = imported.form === "table" ? `${imported.regions} regions → ${tiles}` : tiles;
    toast(`Imported ${shape} from ${imported.filename}.`, "info");
    // If the file was in watts, its total is probably the chiplet's power.
    const total = imported.total;
    if (total > 0 && Math.abs(total - c.power_w) > 0.01 * Math.max(total, c.power_w)) {
      const yes = await ask({
        title: "Set the chiplet's power to the file's total?",
        body: `The map adds up to ${total.toFixed(2)}, and ${escapeHtml(c.name)} is set to `
          + `${c.power_w.toFixed(2)} W. If the file is in watts, match them. If it is in `
          + `percentages or activity counts, keep the power you have — the map's shape is `
          + `used either way.`,
        ok: `Set ${total.toFixed(2)} W`,
        cancel: "Keep " + c.power_w.toFixed(2) + " W",
      });
      if (yes === true) {
        record();
        c.power_w = Math.max(0, Math.min(2000, total));
        renderChipletList();
        requestSolve();
      }
    }
  });
  $("ed-map-clear").addEventListener("click", () => {
    const c = state.chiplets.find((q) => q.id === state.selectedId);
    if (!c || !c.power_grid) return;
    record();
    delete c.power_grid;
    delete c.map_source;
    c.power_pattern = "uniform";
    renderChipletList();
    renderFloorplan();
    requestSolve();
  });
  $("ed-pattern").addEventListener("change", () => {
    const c = state.chiplets.find((q) => q.id === state.selectedId);
    if (!c) return;
    record();
    c.power_pattern = $("ed-pattern").value;
    delete c.blocks;                 // a named pattern replaces custom blocks
    delete c.power_grid;
    delete c.map_source;
    renderChipletList();
    renderFloorplan();
    requestSolve();
  });
  $("ed-rotate").addEventListener("click", rotateSelected);
  $("ed-whatif").addEventListener("click", whatIf);
  $("copilot-ask").addEventListener("click", askCopilot);
  $("copilot-text").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) askCopilot();
  });
  $("copilot-apply").addEventListener("click", applyCopilot);
  $("ed-delete").addEventListener("click", removeSelected);

  for (const b of document.querySelectorAll("#gl-mode button")) b.addEventListener("click", () => { if (!b.disabled) setGlMode(b.dataset.mode); });
  for (const b of document.querySelectorAll("#gl-views button[data-view]")) b.addEventListener("click", () => { const v = ensureViewer(); if (v) v.setView(b.dataset.view); });
  $("gl-spin").addEventListener("click", () => { const v = ensureViewer(); if (v) v.setSpin(!v.spinning); });
  $("btn-burst").addEventListener("click", runBurst);
  for (const id of ["tr-burst", "tr-idle", "tr-limit"]) {
    $(id).addEventListener("change", () => { if (state.burst) renderTimeline(); });
  }
  new ResizeObserver(() => { if (state.tab === "timeline") renderTimeline(); }).observe($("timeline"));
  $("btn-report").addEventListener("click", () => {
    if (!state.opt) return;
    const sel = $("scenario");
    downloadReport({
      opt: state.opt, pkg: state.pkg,
      scenarioTitle: sel.selectedIndex >= 0 ? sel.options[sel.selectedIndex].textContent : "Custom layout",
      colours: new Map(state.chiplets.map((c) => [c.id, c.color])),
    });
  });
  $("btn-focus").addEventListener("click", () => {
    state.focus = !state.focus;
    applyLayout();
    requestAnimationFrame(() => {
      if (state.tab === "floorplan") renderFloorplan(violations().ids);
      if (state.tab === "view3d" && state.viewer) state.viewer.requestDraw();
    });
  });
  const help = $("help-overlay");
  $("btn-help").addEventListener("click", () => help.classList.remove("hidden"));
  $("btn-help-close").addEventListener("click", () => help.classList.add("hidden"));
  help.addEventListener("click", (e) => { if (e.target === help) help.classList.add("hidden"); });
  window.addEventListener("keydown", (e) => { if (e.key === "Escape") help.classList.add("hidden"); });
  $("gl-peel").addEventListener("input", () => {
    const v = Number($("gl-peel").value);
    $("gl-peel-val").textContent = v ? `${v} layer${v > 1 ? "s" : ""}` : "none";
    const viewer = ensureViewer(); if (viewer) viewer.setPeel(v);
  });
  $("gl-cut").addEventListener("input", () => {
    const v = Number($("gl-cut").value);
    const cell = state.shown ? state.shown.grid.cell_mm : 0.5;
    $("gl-cut-val").textContent = v ? `${(v * cell).toFixed(1)} mm` : "off";
    const viewer = ensureViewer(); if (viewer) viewer.setCut(v);
  });
  $("gl-exag").addEventListener("input", () => {
    const v = Number($("gl-exag").value);
    $("gl-exag-val").textContent = v;
    const viewer = ensureViewer(); if (viewer) viewer.setExaggeration(v);
  });

  window.addEventListener("keydown", (e) => {
    if (state.tab !== "floorplan" || state.animating || state.busy) return;
    const tag = document.activeElement && document.activeElement.tagName;
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
    const c = state.chiplets.find((q) => q.id === state.selectedId);
    if (!c) return;
    const cell = cellMm(state.pkg);
    const steps = { ArrowLeft: [-cell, 0], ArrowRight: [cell, 0], ArrowUp: [0, cell], ArrowDown: [0, -cell] };
    if (steps[e.key]) {
      e.preventDefault();
      record();
      c.x_mm = +(c.x_mm + steps[e.key][0]).toFixed(6);
      c.y_mm = +(c.y_mm + steps[e.key][1]).toFixed(6);
      fitInside(c);
      renderChipletList();
      requestSolve();
    } else if (e.key === "r" || e.key === "R") {
      rotateSelected();
    } else if (e.key === "Delete") {
      removeSelected();
    }
  });
  // ---- designs on disk, undo and redo
  history.onChange = updateHistoryButtons;
  $("btn-undo").addEventListener("click", undo);
  $("btn-redo").addEventListener("click", redo);
  $("btn-save").addEventListener("click", () => {
    const design = saveDesign(state.pkg, state.chiplets, designTitle());
    toast(`Saved ${design.chiplets.length} chiplets as a RISC3D design file.`, "info");
  });
  $("btn-open").addEventListener("click", async () => {
    try {
      const design = await openDesign($("file-open"));
      if (!design) return;
      record();
      state.opened = design;
      state.scenarioId = null;
      state.pkg = design.pkg;
      state.chiplets = design.chiplets;
      state.selectedId = null;
      state.shown = null;
      for (const c of state.chiplets) fitInside(c);
      clearOptimization();
      syncPackageControls();
      showOpenedInScenarioList(design.title);
      renderAll();
      requestSolve();
      toast(`Opened "${design.title}".`, "info");
    } catch (err) {
      toast(err.message);
    }
  });
  $("btn-reset").addEventListener("click", async () => {
    const target = state.opened && !state.scenarioId ? `"${state.opened.title}"` : "this scenario";
    const yes = await ask({
      title: "Reset the design?",
      body: `Everything changed since ${target} was loaded is discarded. Save it first if you want to keep it.`,
      ok: "Reset",
    });
    if (yes === true) resetDesign();
  });
  window.addEventListener("keydown", (e) => {
    const tag = document.activeElement && document.activeElement.tagName;
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
    if (!(e.ctrlKey || e.metaKey)) return;
    const key = e.key.toLowerCase();
    if (key === "z" && !e.shiftKey) { e.preventDefault(); undo(); }
    else if ((key === "z" && e.shiftKey) || key === "y") { e.preventDefault(); redo(); }
    else if (key === "s") { e.preventDefault(); $("btn-save").click(); }
    else if (key === "o") { e.preventDefault(); $("btn-open").click(); }
  });
  window.addEventListener("resize", () => drawColorbar($("fp-colorbar"), "temp"));
  // Redraw the comparison whenever its maps change size (window resize or the banner/tables reflowing).
  new ResizeObserver(() => { if (state.tab === "compare" && state.opt) renderCompare(); }).observe($("cmp-initial"));
}

async function init() {
  wire();
  drawColorbar($("fp-colorbar"), "temp");
  setStatus("busy", "Loading scenarios…");
  try {
    const res = await getJSON("/api/scenarios");
    if (!res.ok) throw new Error(res.error || "no scenarios");
    state.scenarios = res.scenarios;
    if (res.power_patterns) state.patterns = res.power_patterns;
    renderPatternOptions();
    $("scenario").innerHTML = res.scenarios.map((s) => `<option value="${s.id}">${escapeHtml(s.title)}</option>`).join("");
    loadScenario(res.scenarios[0]);
  } catch (err) {
    setStatus("error", "Engine unreachable");
    toast(`Cannot reach the RISC3D engine: ${err.message}. Start it with "python run_studio.py".`);
  }
}

// Developer handle for inspection from the browser console.
window.__risc3d = { state, update3D, renderCompare, switchTab, setGlMode };

init();
