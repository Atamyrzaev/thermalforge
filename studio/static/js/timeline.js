// RISC3D Studio :: the burst, drawn.
//
// A steady-state answer says where a package ends up. This says how it gets
// there -- which is the question a power-management engineer actually asks,
// because a part that settles at 95 degC may still trip a 100 degC limit
// 200 ms into a burst, and one that settles above its limit may never reach
// it inside a workload that only lasts a second.
//
// One canvas, no libraries: axes to the data's own range, the burst window
// shaded, the steady-state asymptote and the limit as reference lines.

import { setupCanvas } from "./mapdraw.js";

const PAD = { left: 58, right: 16, top: 16, bottom: 34 };

function niceStep(span, target) {
  const raw = span / Math.max(1, target);
  const power = Math.pow(10, Math.floor(Math.log10(raw)));
  for (const step of [1, 2, 2.5, 5, 10]) {
    if (power * step >= raw) return power * step;
  }
  return power * 10;
}

export function drawTimeline(canvas, result, colours) {
  if (!result || !result.t_s || result.t_s.length < 2) return;
  const { ctx, width, height } = setupCanvas(canvas);
  ctx.clearRect(0, 0, width, height);

  const x0 = PAD.left, y0 = PAD.top;
  const plotW = Math.max(10, width - PAD.left - PAD.right);
  const plotH = Math.max(10, height - PAD.top - PAD.bottom);

  const tMax = result.t_s[result.t_s.length - 1];
  // The package peak is drawn LAST, over the per-chiplet curves: it is the
  // number the rest of the app quotes, and it usually runs along whichever
  // chiplet is hottest, so underneath it would simply disappear.
  const series = (result.chiplets || []).map((c) => ({
    id: c.id, label: (colours.get(c.id) || {}).name || c.id, values: c.peak_c, width: 1.4,
    colour: (colours.get(c.id) || {}).color || "#8aa0b8",
  })).concat([{ id: "__peak", label: "Hottest point in the package", values: result.peak_c,
                width: 2.5, colour: "#ff5d6c" }]);

  let lo = Infinity, hi = -Infinity;
  for (const s of series) for (const v of s.values) { if (v < lo) lo = v; if (v > hi) hi = v; }
  for (const reference of [result.steady_peak_c, result.limit_c]) {
    if (reference != null) { lo = Math.min(lo, reference); hi = Math.max(hi, reference); }
  }
  const margin = Math.max(0.5, (hi - lo) * 0.08);
  lo -= margin; hi += margin;

  const toX = (t) => x0 + (t / tMax) * plotW;
  const toY = (v) => y0 + plotH - ((v - lo) / (hi - lo)) * plotH;

  // The burst window, behind everything: the reason the curve moves.
  const burstEnd = Math.min(tMax, result.profile ? result.profile.burst_s : tMax);
  ctx.fillStyle = "rgba(255,122,69,0.10)";
  ctx.fillRect(x0, y0, toX(burstEnd) - x0, plotH);
  ctx.fillStyle = "#8b9ab3";
  ctx.font = "11px Segoe UI, system-ui, sans-serif";
  ctx.textAlign = "left"; ctx.textBaseline = "top";
  ctx.fillText("full power", x0 + 6, y0 + 4);
  if (burstEnd < tMax) ctx.fillText("idle", toX(burstEnd) + 6, y0 + 4);

  // Grid and axes.
  const tStep = niceStep(tMax, 6), vStep = niceStep(hi - lo, 5);
  ctx.strokeStyle = "rgba(140,160,190,0.14)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let t = 0; t <= tMax + 1e-9; t += tStep) { ctx.moveTo(toX(t), y0); ctx.lineTo(toX(t), y0 + plotH); }
  for (let v = Math.ceil(lo / vStep) * vStep; v <= hi; v += vStep) {
    ctx.moveTo(x0, toY(v)); ctx.lineTo(x0 + plotW, toY(v));
  }
  ctx.stroke();

  ctx.fillStyle = "#8b9ab3";
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let t = 0; t <= tMax + 1e-9; t += tStep) {
    ctx.fillText(`${t < 1 ? Math.round(t * 1000) + " ms" : t.toFixed(t < 10 ? 1 : 0) + " s"}`,
                 toX(t), y0 + plotH + 8);
  }
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let v = Math.ceil(lo / vStep) * vStep; v <= hi; v += vStep) {
    ctx.fillText(`${v.toFixed(0)} °C`, x0 - 8, toY(v));
  }

  // Reference lines: where it would end up, and where it must not go.
  const reference = (value, colour, label) => {
    if (value == null) return;
    ctx.strokeStyle = colour;
    ctx.setLineDash([5, 4]);
    ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(x0, toY(value)); ctx.lineTo(x0 + plotW, toY(value)); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = colour;
    ctx.textAlign = "right"; ctx.textBaseline = "bottom";
    ctx.fillText(label, x0 + plotW - 4, toY(value) - 3);
  };
  reference(result.steady_peak_c, "#8b9ab3", "steady state");
  reference(result.limit_c, "#ffc452", `limit ${result.limit_c} °C`);

  // The curves.
  for (const s of series) {
    ctx.strokeStyle = s.colour;
    ctx.lineWidth = s.width;
    ctx.beginPath();
    result.t_s.forEach((t, n) => {
      const x = toX(t), y = toY(s.values[n]);
      if (n === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  // Where the limit was crossed, if it was.
  if (result.time_to_limit_s != null && result.limit_c != null) {
    const x = toX(result.time_to_limit_s);
    ctx.strokeStyle = "#ffc452";
    ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y0 + plotH); ctx.stroke();
    ctx.fillStyle = "#ffc452";
    ctx.textAlign = "left"; ctx.textBaseline = "top";
    const label = result.time_to_limit_s < 1
      ? `${Math.round(result.time_to_limit_s * 1000)} ms to the limit`
      : `${result.time_to_limit_s.toFixed(2)} s to the limit`;
    ctx.fillText(label, Math.min(x + 6, x0 + plotW - 120), y0 + plotH - 18);
  }

  // Legend, top left under the phase labels: a curve nobody can name is
  // decoration.
  ctx.font = "11px Segoe UI, system-ui, sans-serif";
  ctx.textAlign = "left"; ctx.textBaseline = "middle";
  let legendY = y0 + 24;
  for (const s of series) {
    ctx.strokeStyle = s.colour;
    ctx.lineWidth = s.width;
    ctx.beginPath(); ctx.moveTo(x0 + 8, legendY); ctx.lineTo(x0 + 26, legendY); ctx.stroke();
    ctx.fillStyle = "#c7d3e6";
    ctx.fillText(s.label, x0 + 32, legendY);
    legendY += 15;
  }

  // Axis frame last, over the fills.
  ctx.strokeStyle = "#3a4d6c";
  ctx.lineWidth = 1;
  ctx.strokeRect(x0 + 0.5, y0 + 0.5, plotW - 1, plotH - 1);
  return series;
}
