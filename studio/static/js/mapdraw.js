// RISC3D Studio :: top-view package map, shared by the floorplan editor and
// the before/after comparison.
//
// Heat is drawn one pixel per mesh cell and scaled without smoothing, so what
// you see is exactly what was solved -- no interpolation invents detail the
// mesh does not have. y points up, x right, origin at the lower-left corner.

import { deltaColor, tempColor } from "./colormap.js";

// Hottest temperature through the stack at each (i, j): the quantity a
// top-view thermal map should show for a 3-D field.
export function columnMax(field, nx, ny, nz) {
  const out = new Float32Array(nx * ny);
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      let m = -Infinity;
      const base = (i * ny + j) * nz;
      for (let k = 0; k < nz; k++) if (field[base + k] > m) m = field[base + k];
      out[i * ny + j] = m;
    }
  }
  return out;
}

export function fitSquare(width, height, pad = { left: 46, right: 18, top: 16, bottom: 40 }) {
  const size = Math.max(10, Math.min(width - pad.left - pad.right, height - pad.top - pad.bottom));
  const x0 = pad.left + (width - pad.left - pad.right - size) / 2;
  const y0 = pad.top + (height - pad.top - pad.bottom - size) / 2;
  return { x0, y0, size };
}

function niceStep(sizeMm) {
  if (sizeMm <= 10) return 2;
  if (sizeMm <= 20) return 4;
  if (sizeMm <= 30) return 5;
  return 10;
}

function heatImage(values, nx, ny, colour) {
  const off = document.createElement("canvas");
  off.width = nx; off.height = ny;
  const ctx = off.getContext("2d");
  const img = ctx.createImageData(nx, ny);
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      const [r, g, b] = colour(values[i * ny + j]);
      const o = ((ny - 1 - j) * nx + i) * 4;
      img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return off;
}

/**
 * opts: {
 *   sizeMm, pitchMm, heat: {values, nx, ny, lo, hi, kind: "temp"|"delta"} | null,
 *   chiplets: [{id, name, w_mm, h_mm, x_mm, y_mm, power_w, color}],
 *   selectedId, invalidIds: Set, temps: Map(id -> max °C), peak: {x_mm, y_mm, t_c} | null,
 *   compact: bool
 * }
 * Returns the square geometry used, for hit-testing.
 */
export function drawPackageMap(ctx, width, height, opts) {
  const geo = fitSquare(width, height, opts.compact
    ? { left: 30, right: 8, top: 8, bottom: 24 } : undefined);
  const { x0, y0, size } = geo;
  const s = size / opts.sizeMm;
  const toX = (mm) => x0 + mm * s;
  const toY = (mm) => y0 + size - mm * s;

  ctx.clearRect(0, 0, width, height);

  // Package body and heat.
  if (opts.heat) {
    const { values, nx, ny, lo, hi, kind } = opts.heat;
    const colour = kind === "delta" ? (v) => deltaColor(v, hi) : (v) => tempColor(v, lo, hi);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(heatImage(values, nx, ny, colour), x0, y0, size, size);
  } else {
    ctx.fillStyle = "#1a2536";
    ctx.fillRect(x0, y0, size, size);
  }

  // Placement grid.
  if (opts.pitchMm && !opts.compact) {
    ctx.strokeStyle = opts.heat ? "rgba(0,0,0,0.16)" : "rgba(140,160,190,0.12)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let mm = opts.pitchMm; mm < opts.sizeMm; mm += opts.pitchMm) {
      ctx.moveTo(toX(mm) + 0.5, y0); ctx.lineTo(toX(mm) + 0.5, y0 + size);
      ctx.moveTo(x0, toY(mm) + 0.5); ctx.lineTo(x0 + size, toY(mm) + 0.5);
    }
    ctx.stroke();
  }

  // Axes.
  ctx.strokeStyle = "#3a4d6c";
  ctx.lineWidth = 1;
  ctx.strokeRect(x0 + 0.5, y0 + 0.5, size - 1, size - 1);
  ctx.fillStyle = "#8b9ab3";
  ctx.font = `${opts.compact ? 10 : 11}px Segoe UI, system-ui, sans-serif`;
  const step = niceStep(opts.sizeMm) * (opts.compact ? 2 : 1);
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let mm = 0; mm <= opts.sizeMm + 1e-9; mm += step) ctx.fillText(`${mm}`, toX(mm), y0 + size + 5);
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let mm = 0; mm <= opts.sizeMm + 1e-9; mm += step) ctx.fillText(`${mm}`, x0 - 6, toY(mm));
  if (!opts.compact) {
    ctx.textAlign = "left"; ctx.textBaseline = "top";
    ctx.fillText("x, y in mm", x0, y0 + size + 19);
  }

  // Chiplets.
  for (const c of opts.chiplets) {
    const px = toX(c.x_mm), py = toY(c.y_mm + c.h_mm);
    const pw = c.w_mm * s, ph = c.h_mm * s;
    const invalid = opts.invalidIds && opts.invalidIds.has(c.id);
    const selected = c.id === opts.selectedId;
    ctx.fillStyle = hexAlpha(c.color || "#8aa0b8", opts.heat ? 0.12 : 0.55);
    ctx.fillRect(px, py, pw, ph);
    ctx.lineWidth = selected ? 3 : 2;
    ctx.setLineDash(invalid ? [6, 4] : []);
    ctx.strokeStyle = invalid ? "#ff5d6c" : selected ? "#ffffff" : (opts.heat ? "rgba(255,255,255,0.85)" : c.color || "#8aa0b8");
    ctx.strokeRect(px + 1, py + 1, pw - 2, ph - 2);
    ctx.setLineDash([]);

    if (!opts.compact || pw > 46) {
      const lines = [c.name];
      if (!opts.compact) lines.push(`${formatW(c.power_w)} W`);
      if (opts.temps && opts.temps.has(c.id)) lines.push(`${opts.temps.get(c.id).toFixed(1)} °C`);
      // Largest font (up to the cap) at which the widest line fits the block.
      ctx.font = "700 10px Segoe UI, system-ui, sans-serif";
      const widest = Math.max(...lines.map((text) => ctx.measureText(text).width)) / 10;
      const cap = opts.compact ? 12 : 15;
      const fontPx = Math.min(cap, (pw - 8) / Math.max(1, widest), ph / (lines.length * 1.35));
      if (fontPx >= 8 && pw > 24 && ph > fontPx * 1.2) {
        ctx.save();
        ctx.beginPath(); ctx.rect(px, py, pw, ph); ctx.clip();
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        const lineH = fontPx * 1.3;
        const top = py + ph / 2 - (lines.length - 1) * lineH / 2;
        lines.forEach((text, n) => {
          ctx.font = `${n === 0 ? 700 : 500} ${fontPx}px Segoe UI, system-ui, sans-serif`;
          ctx.lineWidth = 3; ctx.strokeStyle = "rgba(0,0,0,0.75)";
          ctx.strokeText(text, px + pw / 2, top + n * lineH);
          ctx.fillStyle = "#ffffff";
          ctx.fillText(text, px + pw / 2, top + n * lineH);
        });
        ctx.restore();
      }
    }
  }

  // Peak marker.
  if (opts.peak) {
    const px = toX(opts.peak.x_mm), py = toY(opts.peak.y_mm);
    ctx.lineWidth = 2; ctx.strokeStyle = "#ffffff";
    ctx.beginPath(); ctx.arc(px, py, 7, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(px - 11, py); ctx.lineTo(px + 11, py); ctx.moveTo(px, py - 11); ctx.lineTo(px, py + 11); ctx.stroke();
    if (!opts.compact && opts.peak.t_c !== undefined) {
      const label = `Peak ${opts.peak.t_c.toFixed(1)} °C`;
      ctx.font = "700 12px Segoe UI, system-ui, sans-serif";
      const tw = ctx.measureText(label).width + 12;
      // Keep the label off the chiplet's own name: sit it just above the
      // chiplet that contains the peak (or below, at the package's top edge).
      const host = opts.chiplets.find((c) => opts.peak.x_mm >= c.x_mm && opts.peak.x_mm <= c.x_mm + c.w_mm
        && opts.peak.y_mm >= c.y_mm && opts.peak.y_mm <= c.y_mm + c.h_mm);
      let lx = px - tw / 2;
      let ly = host ? toY(host.y_mm + host.h_mm) - 26 : py - 30;
      if (ly < y0 + 2) ly = host ? toY(host.y_mm) + 6 : py + 14;
      lx = Math.max(x0 + 2, Math.min(x0 + size - tw - 2, lx));
      ctx.fillStyle = "rgba(255,93,108,0.92)";
      roundRect(ctx, lx, ly, tw, 20, 5); ctx.fill();
      ctx.fillStyle = "#fff"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
      ctx.fillText(label, lx + 6, ly + 10);
    }
  }
  return { ...geo, scale: s, toX, toY, toMmX: (px) => (px - x0) / s, toMmY: (py) => (y0 + size - py) / s };
}

export function formatW(w) {
  return w >= 100 ? w.toFixed(0) : w >= 10 ? w.toFixed(1) : w.toFixed(2);
}

function hexAlpha(hex, alpha) {
  const v = hex.replace("#", "");
  const n = parseInt(v.length === 3 ? v.split("").map((c) => c + c).join("") : v, 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${alpha})`;
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}

export function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  const W = Math.max(1, Math.round(w * dpr)), H = Math.max(1, Math.round(h * dpr));
  if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width: w, height: h };
}
