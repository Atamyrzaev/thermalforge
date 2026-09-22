// RISC3D Studio :: interactive floorplan.
//
// Chiplets are dragged in physical millimetres and snap to the mesh cell, so
// every position the user can reach is one the solver represents exactly.

import { drawPackageMap, setupCanvas } from "./mapdraw.js";

export class FloorplanView {
  constructor(canvas, tooltip, handlers) {
    this.canvas = canvas;
    this.tooltip = tooltip;
    this.handlers = handlers;
    this.model = null;
    this.geo = null;
    this.drag = null;
    canvas.addEventListener("pointerdown", (e) => this.onDown(e));
    canvas.addEventListener("pointermove", (e) => this.onMove(e));
    canvas.addEventListener("pointerup", (e) => this.onUp(e));
    canvas.addEventListener("pointercancel", (e) => this.onUp(e));
    canvas.addEventListener("pointerleave", () => { if (!this.drag) this.hideTip(); });
    new ResizeObserver(() => this.draw()).observe(canvas);
  }

  setModel(model) {
    this.model = model;
    this.draw();
  }

  draw() {
    if (!this.model || !this.canvas.clientWidth || !this.canvas.clientHeight) return;
    const { ctx, width, height } = setupCanvas(this.canvas);
    this.geo = drawPackageMap(ctx, width, height, this.model);
  }

  local(e) {
    const r = this.canvas.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }

  hit(mmX, mmY) {
    const list = this.model.chiplets;
    for (let n = list.length - 1; n >= 0; n--) {
      const c = list[n];
      if (mmX >= c.x_mm && mmX <= c.x_mm + c.w_mm && mmY >= c.y_mm && mmY <= c.y_mm + c.h_mm) return c;
    }
    return null;
  }

  onDown(e) {
    if (!this.geo || !this.model || this.model.locked) return;
    const p = this.local(e);
    const mx = this.geo.toMmX(p.x), my = this.geo.toMmY(p.y);
    const c = this.hit(mx, my);
    this.handlers.onSelect(c ? c.id : null);
    if (!c) return;
    this.drag = { id: c.id, dx: mx - c.x_mm, dy: my - c.y_mm, lastX: c.x_mm, lastY: c.y_mm, moved: false };
    try { this.canvas.setPointerCapture(e.pointerId); } catch (_) { /* capture is a nicety */ }
    this.canvas.style.cursor = "grabbing";
    this.hideTip();
  }

  onMove(e) {
    if (!this.geo || !this.model) return;
    const p = this.local(e);
    const mx = this.geo.toMmX(p.x), my = this.geo.toMmY(p.y);
    if (this.drag) {
      const c = this.model.chiplets.find((q) => q.id === this.drag.id);
      if (!c) return;
      const cell = this.model.cellMm;
      let x = Math.round((mx - this.drag.dx) / cell) * cell;
      let y = Math.round((my - this.drag.dy) / cell) * cell;
      x = +Math.max(0, Math.min(this.model.sizeMm - c.w_mm, x)).toFixed(6);
      y = +Math.max(0, Math.min((this.model.sizeYMm || this.model.sizeMm) - c.h_mm, y)).toFixed(6);
      if (x !== this.drag.lastX || y !== this.drag.lastY) {
        this.drag.lastX = x;
        this.drag.lastY = y;
        this.drag.moved = true;
        this.handlers.onMove(c.id, x, y, true);
      }
      return;
    }
    const c = this.hit(mx, my);
    this.canvas.style.cursor = c && !this.model.locked ? "grab" : "default";
    this.showTip(p, mx, my, c);
  }

  onUp(e) {
    if (!this.drag) return;
    const d = this.drag;
    this.drag = null;
    this.canvas.style.cursor = "grab";
    try { this.canvas.releasePointerCapture(e.pointerId); } catch (_) { /* already released */ }
    if (d.moved) this.handlers.onMove(d.id, d.lastX, d.lastY, false);
  }

  showTip(p, mx, my, c) {
    const m = this.model;
    const sizeY = m.sizeYMm || m.sizeMm;
    if (mx < 0 || my < 0 || mx > m.sizeMm || my > sizeY) { this.hideTip(); return; }
    const lines = [];
    if (m.heat && m.heat.kind === "temp") {
      const i = Math.min(m.heat.nx - 1, Math.floor(mx / (m.sizeMm / m.heat.nx)));
      const j = Math.min(m.heat.ny - 1, Math.floor(my / (sizeY / m.heat.ny)));
      lines.push(`<b>${m.heat.values[i * m.heat.ny + j].toFixed(2)} °C</b> hottest in this column`);
    }
    lines.push(`x ${mx.toFixed(2)} mm · y ${my.toFixed(2)} mm`);
    if (c) lines.push(`${escapeHtml(c.name)} · ${c.w_mm} × ${c.h_mm} mm · ${c.power_w.toFixed(2)} W`);
    this.tooltip.innerHTML = lines.join("<br>");
    const w = this.canvas.clientWidth;
    this.tooltip.style.left = `${Math.min(p.x, w - 230)}px`;
    this.tooltip.style.top = `${p.y}px`;
    this.tooltip.classList.remove("hidden");
  }

  hideTip() {
    this.tooltip.classList.add("hidden");
  }
}

export function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}
