// RISC3D Studio :: 3-D thermal viewer (raw WebGL, no libraries).
//
// The package is drawn voxel by voxel from the solved field: every visible
// face is one mesh cell, coloured by that cell's temperature. Only surface
// faces are emitted, so the picture is exactly the data. Height is
// exaggerated (the stack is 0.24 mm thick on a 20 mm package) and the scale
// is shown in the UI.
//
// Camera: orbit around a target. Left-drag rotates (yaw around the vertical
// axis, pitch clamped so the view never flips), right-drag or Shift-drag pans
// in the screen plane, the wheel zooms toward the target, double-click resets.
// Two-finger touch pans and pinches.

import { deltaColor, tempColor } from "./colormap.js";
import { escapeHtml } from "./floorplan.js";

const VERT = `
attribute vec3 aPos;
attribute vec3 aNormal;
attribute vec3 aColor;
uniform mat4 uMVP;
varying vec3 vColor;
varying vec3 vNormal;
void main() {
  gl_Position = uMVP * vec4(aPos, 1.0);
  vColor = aColor;
  vNormal = aNormal;
}`;

const FRAG = `
precision mediump float;
varying vec3 vColor;
varying vec3 vNormal;
uniform vec3 uLight;
uniform float uFlat;
void main() {
  float d = max(dot(normalize(vNormal), uLight), 0.0);
  float shade = mix(0.62 + 0.38 * d, 1.0, uFlat);
  gl_FragColor = vec4(vColor * shade, 1.0);
}`;

const FOV = 0.62;                      // vertical field of view, radians
const PITCH_MIN = 0.02, PITCH_MAX = 1.5605;

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
}

function lookAt(eye, target, up) {
  const z = norm(sub(eye, target));
  const x = norm(cross(up, z));
  const y = cross(z, x);
  return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1]);
}

function multiply(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      o[c * 4 + r] = s;
    }
  }
  return o;
}

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
  return shader;
}

class Growable {
  constructor(capacity = 1 << 16) { this.data = new Float32Array(capacity); this.length = 0; }
  reserve(n) {
    if (this.length + n <= this.data.length) return;
    let cap = this.data.length;
    while (cap < this.length + n) cap *= 2;
    const next = new Float32Array(cap);
    next.set(this.data.subarray(0, this.length));
    this.data = next;
  }
  vertex(x, y, z, nx, ny, nz, r, g, b) {
    this.reserve(9);
    const d = this.data, o = this.length;
    d[o] = x; d[o + 1] = y; d[o + 2] = z; d[o + 3] = nx; d[o + 4] = ny; d[o + 5] = nz;
    d[o + 6] = r; d[o + 7] = g; d[o + 8] = b;
    this.length += 9;
  }
  view() { return this.data.subarray(0, this.length); }
}

export class Viewer3D {
  constructor(canvas, labelsEl, tooltipEl) {
    this.canvas = canvas;
    this.labelsEl = labelsEl;
    this.tooltip = tooltipEl;
    const gl = canvas.getContext("webgl", { antialias: true }) || canvas.getContext("experimental-webgl");
    if (!gl) throw new Error("WebGL is not available in this browser, so the 3D view cannot be shown.");
    this.gl = gl;

    const program = gl.createProgram();
    gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERT));
    gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
    this.program = program;
    this.loc = {
      aPos: gl.getAttribLocation(program, "aPos"),
      aNormal: gl.getAttribLocation(program, "aNormal"),
      aColor: gl.getAttribLocation(program, "aColor"),
      uMVP: gl.getUniformLocation(program, "uMVP"),
      uLight: gl.getUniformLocation(program, "uLight"),
      uFlat: gl.getUniformLocation(program, "uFlat"),
    };
    this.meshBuffer = gl.createBuffer();
    this.lineBuffer = gl.createBuffer();
    this.meshCount = 0;
    this.lineCount = 0;

    this.data = null;
    this.exag = 25;
    this.peel = 0;
    this.cut = 0;
    this.cam = { yaw: 0.7, pitch: 0.62, dist: 40, target: [0, 1, 0] };
    this.pointers = new Map();
    this.raf = null;
    this.spinning = false;
    this.spinRaf = null;
    this.onSpinChange = null;
    this.mvp = null;

    canvas.addEventListener("contextmenu", (e) => e.preventDefault());
    canvas.addEventListener("pointerdown", (e) => this.onDown(e));
    canvas.addEventListener("pointermove", (e) => this.onMove(e));
    canvas.addEventListener("pointerup", (e) => this.onUp(e));
    canvas.addEventListener("pointercancel", (e) => this.onUp(e));
    canvas.addEventListener("pointerleave", () => { if (!this.pointers.size) this.hideTip(); });
    canvas.addEventListener("wheel", (e) => this.onWheel(e), { passive: false });
    canvas.addEventListener("dblclick", () => { this.resetView(); });
    new ResizeObserver(() => this.requestDraw()).observe(canvas);
  }

  // ---------------------------------------------------------------- data
  setData(data) {
    const regrid = !this.data || this.data.nx !== data.nx || this.data.ny !== data.ny
      || this.data.nz !== data.nz || this.data.cell_mm !== data.cell_mm;
    this.data = data;
    this.peel = Math.min(this.peel, data.nz - 1);
    this.cut = Math.min(this.cut, data.ny - 1);
    this.rebuild();
    if (regrid) this.resetView();
    this.requestDraw();
  }

  setExaggeration(v) { this.exag = v; this.rebuild(); this.requestDraw(); }
  setPeel(v) { this.peel = v; this.rebuild(); this.requestDraw(); }
  setCut(v) { this.cut = v; this.rebuild(); this.requestDraw(); }

  sizeMm() { return this.data ? this.data.nx * this.data.cell_mm : 20; }
  sizeYMm() { return this.data ? this.data.ny * this.data.cell_mm : 20; }
  dzWorld() { return (this.data.dz_um / 1000) * this.exag; }

  rebuild() {
    const d = this.data;
    if (!d) return;
    const gl = this.gl;
    const { nx, ny, nz, cell_mm: c, values } = d;
    const half = (nx * c) / 2;
    const dz = this.dzWorld();
    const kTop = nz - this.peel;
    const jCut = this.cut;
    const colour = d.kind === "delta" ? (v) => deltaColor(v, d.hi) : (v) => tempColor(v, d.lo, d.hi);
    const visible = (i, j, k) => i >= 0 && i < nx && j >= jCut && j < ny && k >= 0 && k < kTop;
    const mesh = new Growable(1 << 18);

    const quad = (p, n, col) => {
      const r = col[0] / 255, g = col[1] / 255, b = col[2] / 255;
      for (const idx of [0, 1, 2, 0, 2, 3]) mesh.vertex(p[idx][0], p[idx][1], p[idx][2], n[0], n[1], n[2], r, g, b);
    };

    for (let i = 0; i < nx; i++) {
      const X0 = i * c - half, X1 = X0 + c;
      for (let j = jCut; j < ny; j++) {
        const Zf = half - j * c, Zb = Zf - c;
        for (let k = 0; k < kTop; k++) {
          const Y0 = k * dz, Y1 = Y0 + dz;
          const col = colour(values[(i * ny + j) * nz + k]);
          if (!visible(i + 1, j, k)) quad([[X1, Y0, Zf], [X1, Y0, Zb], [X1, Y1, Zb], [X1, Y1, Zf]], [1, 0, 0], col);
          if (!visible(i - 1, j, k)) quad([[X0, Y0, Zb], [X0, Y0, Zf], [X0, Y1, Zf], [X0, Y1, Zb]], [-1, 0, 0], col);
          if (!visible(i, j + 1, k)) quad([[X0, Y0, Zb], [X1, Y0, Zb], [X1, Y1, Zb], [X0, Y1, Zb]], [0, 0, -1], col);
          if (!visible(i, j - 1, k)) quad([[X1, Y0, Zf], [X0, Y0, Zf], [X0, Y1, Zf], [X1, Y1, Zf]], [0, 0, 1], col);
          if (!visible(i, j, k + 1)) quad([[X0, Y1, Zf], [X1, Y1, Zf], [X1, Y1, Zb], [X0, Y1, Zb]], [0, 1, 0], col);
          if (!visible(i, j, k - 1)) quad([[X0, Y0, Zb], [X1, Y0, Zb], [X1, Y0, Zf], [X0, Y0, Zf]], [0, -1, 0], col);
        }
      }
    }

    // Chiplet outlines on the top visible surface, while the die is visible.
    const topY = kTop * dz + 0.004 * this.exag;
    if (kTop > d.dieStart && d.chiplets) {
      const w = Math.max(0.05, (nx * c) / 320);
      const yMin = jCut * c;
      for (const ch of d.chiplets) {
        const x0 = ch.x_mm, x1 = ch.x_mm + ch.w_mm;
        const y0 = Math.max(ch.y_mm, yMin), y1 = ch.y_mm + ch.h_mm;
        if (y1 <= y0) continue;
        const strip = (ax, ay, bx, by) => {
          const X0 = ax - half, X1 = bx - half, Z0 = half - ay, Z1 = half - by;
          for (const p of [[X0, Z0], [X1, Z0], [X1, Z1], [X0, Z0], [X1, Z1], [X0, Z1]]) {
            mesh.vertex(p[0], topY, p[1], 0, 1, 0, 1, 1, 1);
          }
        };
        strip(x0, y0, x1, y0 + w);
        strip(x0, y1 - w, x1, y1);
        strip(x0, y0, x0 + w, y1);
        strip(x1 - w, y0, x1, y1);
      }
    }

    // Wireframe: the visible block's edges and the layer interfaces.
    const lines = new Growable(4096);
    const line = (a, b, rgb) => {
      lines.vertex(a[0], a[1], a[2], 0, 1, 0, rgb[0], rgb[1], rgb[2]);
      lines.vertex(b[0], b[1], b[2], 0, 1, 0, rgb[0], rgb[1], rgb[2]);
    };
    const X0 = -half, X1 = half, Zf = half - jCut * c, Zb = -half;
    const H = kTop * dz;
    const edge = [0.55, 0.64, 0.78];
    const rect = (y, rgb) => {
      line([X0, y, Zf], [X1, y, Zf], rgb); line([X1, y, Zf], [X1, y, Zb], rgb);
      line([X1, y, Zb], [X0, y, Zb], rgb); line([X0, y, Zb], [X0, y, Zf], rgb);
    };
    rect(0, edge);
    rect(H, edge);
    for (const [x, z] of [[X0, Zf], [X1, Zf], [X1, Zb], [X0, Zb]]) line([x, 0, z], [x, H, z], edge);
    for (const layer of d.layers.slice(1)) {
      if (layer.k0 < kTop) rect(layer.k0 * dz, [0.08, 0.1, 0.14]);
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, this.meshBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, mesh.view(), gl.STATIC_DRAW);
    this.meshCount = mesh.length / 9;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.lineBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, lines.view(), gl.STATIC_DRAW);
    this.lineCount = lines.length / 9;
  }

  // -------------------------------------------------------------- camera
  // Distance at which the whole block fits the viewport with a margin, from
  // its bounding sphere and the narrower of the two fields of view.
  fitDistance() {
    const S = this.sizeMm();
    const H = this.data ? this.data.nz * this.dzWorld() : 5;
    // Both sides: a 40 x 10 mm package framed as if it were 40 x 40 would
    // sit small in the middle of the view.
    const radius = Math.hypot(S / 2, this.sizeYMm() / 2, H / 2);
    const aspect = Math.max(0.2, this.canvas.clientWidth / Math.max(1, this.canvas.clientHeight));
    const half = Math.min(FOV / 2, Math.atan(Math.tan(FOV / 2) * aspect));
    return (radius / Math.sin(half)) * 1.04;
  }

  resetView() {
    const H = this.data ? this.data.nz * this.dzWorld() : 5;
    this.cam = { yaw: 0.7, pitch: 0.62, dist: this.fitDistance(), target: [0, H / 2, 0] };
    this.requestDraw();
  }

  setView(name) {
    if (name === "iso") { this.resetView(); return; }
    const H = this.data ? this.data.nz * this.dzWorld() : 5;
    const dist = this.fitDistance();
    if (name === "top") this.cam = { yaw: 0, pitch: PITCH_MAX, dist: dist * 0.92, target: [0, H, 0] };
    if (name === "front") this.cam = { yaw: 0, pitch: 0.06, dist: dist * 0.92, target: [0, H / 2, 0] };
    this.requestDraw();
  }

  eye() {
    const { yaw, pitch, dist, target } = this.cam;
    return [target[0] + dist * Math.cos(pitch) * Math.sin(yaw),
      target[1] + dist * Math.sin(pitch),
      target[2] + dist * Math.cos(pitch) * Math.cos(yaw)];
  }

  basis() {
    const eye = this.eye();
    const f = norm(sub(this.cam.target, eye));
    const r = norm(cross(f, [0, 1, 0]));
    const u = cross(r, f);
    return { eye, f, r, u };
  }

  setSpin(on) {
    this.spinning = on;
    if (on && !this.spinRaf) {
      let last = performance.now();
      const step = (now) => {
        if (!this.spinning) { this.spinRaf = null; return; }
        this.cam.yaw -= Math.min(50, now - last) * 0.00035;
        last = now;
        this.draw();
        this.spinRaf = requestAnimationFrame(step);
      };
      this.spinRaf = requestAnimationFrame(step);
    }
    if (this.onSpinChange) this.onSpinChange(on);
  }

  onDown(e) {
    if (this.spinning) this.setSpin(false);
    try { this.canvas.setPointerCapture(e.pointerId); } catch (_) { /* capture is a nicety */ }
    this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.button === 1 || e.shiftKey });
    this.hideTip();
  }

  onMove(e) {
    const prev = this.pointers.get(e.pointerId);
    if (!prev) { this.hover(e); return; }
    if (this.pointers.size >= 2) {
      const others = [...this.pointers.entries()].filter(([id]) => id !== e.pointerId);
      const other = others[0][1];
      const d0 = Math.hypot(prev.x - other.x, prev.y - other.y);
      const d1 = Math.hypot(e.clientX - other.x, e.clientY - other.y);
      const m0 = [(prev.x + other.x) / 2, (prev.y + other.y) / 2];
      const m1 = [(e.clientX + other.x) / 2, (e.clientY + other.y) / 2];
      if (d1 > 1 && d0 > 1) this.zoomBy(d0 / d1);
      this.pan(m1[0] - m0[0], m1[1] - m0[1]);
    } else {
      const dx = e.clientX - prev.x, dy = e.clientY - prev.y;
      if (prev.pan) this.pan(dx, dy); else this.rotate(dx, dy);
    }
    prev.x = e.clientX;
    prev.y = e.clientY;
  }

  onUp(e) {
    this.pointers.delete(e.pointerId);
    try { this.canvas.releasePointerCapture(e.pointerId); } catch (_) { /* released */ }
  }

  onWheel(e) {
    e.preventDefault();
    this.zoomBy(Math.exp(e.deltaY * 0.0012));
  }

  rotate(dx, dy) {
    const k = (2.2 * Math.PI) / Math.max(400, this.canvas.clientWidth);
    this.cam.yaw -= dx * k;
    this.cam.pitch = clamp(this.cam.pitch + dy * k, PITCH_MIN, PITCH_MAX);
    this.requestDraw();
  }

  pan(dx, dy) {
    const { r, u } = this.basis();
    const scale = (2 * this.cam.dist * Math.tan(FOV / 2)) / Math.max(1, this.canvas.clientHeight);
    const t = this.cam.target;
    this.cam.target = [t[0] - r[0] * dx * scale + u[0] * dy * scale,
      t[1] - r[1] * dx * scale + u[1] * dy * scale,
      t[2] - r[2] * dx * scale + u[2] * dy * scale];
    this.requestDraw();
  }

  zoomBy(factor) {
    const S = this.sizeMm();
    this.cam.dist = clamp(this.cam.dist * factor, S * 0.25, S * 8);
    this.requestDraw();
  }

  // ------------------------------------------------------------- picking
  pick(px, py) {
    const d = this.data;
    if (!d) return null;
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    const { eye, f, r, u } = this.basis();
    const th = Math.tan(FOV / 2), aspect = w / h;
    const sx = ((px / w) * 2 - 1) * th * aspect, sy = (1 - (py / h) * 2) * th;
    const dir = norm([f[0] + r[0] * sx + u[0] * sy, f[1] + r[1] * sx + u[1] * sy, f[2] + r[2] * sx + u[2] * sy]);
    const c = d.cell_mm, half = (d.nx * c) / 2, dz = this.dzWorld();
    // Ray in grid coordinates (i, j, k).
    const o = [(eye[0] + half) / c, (half - eye[2]) / c, eye[1] / dz];
    const v = [dir[0] / c, -dir[2] / c, dir[1] / dz];
    const lo = [0, this.cut, 0], hi = [d.nx, d.ny, d.nz - this.peel];
    let t0 = -Infinity, t1 = Infinity;
    for (let a = 0; a < 3; a++) {
      if (Math.abs(v[a]) < 1e-12) {
        if (o[a] < lo[a] || o[a] > hi[a]) return null;
        continue;
      }
      let ta = (lo[a] - o[a]) / v[a], tb = (hi[a] - o[a]) / v[a];
      if (ta > tb) [ta, tb] = [tb, ta];
      t0 = Math.max(t0, ta);
      t1 = Math.min(t1, tb);
    }
    if (t0 > t1 || t1 < 0) return null;
    // The visible region is a box, so the first cell the ray enters is the
    // voxel under the cursor.
    const t = Math.max(t0, 0) + 1e-7;
    const p = [o[0] + v[0] * t, o[1] + v[1] * t, o[2] + v[2] * t];
    return {
      i: clamp(Math.floor(p[0]), 0, d.nx - 1),
      j: clamp(Math.floor(p[1]), this.cut, d.ny - 1),
      k: clamp(Math.floor(p[2]), 0, hi[2] - 1),
    };
  }

  hover(e) {
    const d = this.data;
    if (!d) return;
    const rect = this.canvas.getBoundingClientRect();
    const px = e.clientX - rect.left, py = e.clientY - rect.top;
    const hit = this.pick(px, py);
    if (!hit) { this.hideTip(); return; }
    const { i, j, k } = hit;
    const value = d.values[(i * d.ny + j) * d.nz + k];
    const x = (i + 0.5) * d.cell_mm, y = (j + 0.5) * d.cell_mm, z = (k + 0.5) * d.dz_um;
    const layer = d.layers.find((l) => k >= l.k0 && k < l.k1);
    const lines = [d.kind === "delta"
      ? `<b>${value > 0 ? "+" : ""}${value.toFixed(2)} K</b> ${value < 0 ? "cooler" : value > 0 ? "hotter" : ""} after`
      : `<b>${value.toFixed(2)} °C</b>`];
    lines.push(`${layer ? layer.name : ""} · z ${z.toFixed(0)} µm`);
    lines.push(`x ${x.toFixed(2)} mm · y ${y.toFixed(2)} mm`);
    if (layer && k >= d.dieStart && d.chiplets) {
      const ch = d.chiplets.find((q) => x >= q.x_mm && x <= q.x_mm + q.w_mm && y >= q.y_mm && y <= q.y_mm + q.h_mm);
      if (ch) lines.push(`inside ${escapeHtml(ch.name)}`);
    }
    this.tooltip.innerHTML = lines.join("<br>");
    this.tooltip.style.left = `${Math.min(px, this.canvas.clientWidth - 220)}px`;
    this.tooltip.style.top = `${py}px`;
    this.tooltip.classList.remove("hidden");
  }

  hideTip() { this.tooltip.classList.add("hidden"); }

  // ------------------------------------------------------------- drawing
  requestDraw() {
    if (this.raf) return;
    this.raf = requestAnimationFrame(() => { this.raf = null; this.draw(); });
  }

  project(p, w, h) {
    const m = this.mvp;
    const cx = m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12];
    const cy = m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13];
    const cw = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
    if (cw <= 0) return null;
    return [((cx / cw) * 0.5 + 0.5) * w, (0.5 - (cy / cw) * 0.5) * h];
  }

  draw() {
    const gl = this.gl, canvas = this.canvas;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    const dpr = window.devicePixelRatio || 1;
    const W = Math.round(w * dpr), H = Math.round(h * dpr);
    if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
    gl.viewport(0, 0, W, H);
    gl.clearColor(0.039, 0.059, 0.098, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!this.data || !this.meshCount) { this.labelsEl.innerHTML = ""; return; }

    gl.enable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);
    const S = this.sizeMm();
    const view = lookAt(this.eye(), this.cam.target, [0, 1, 0]);
    const proj = perspective(FOV, w / h, Math.max(0.05, this.cam.dist * 0.02), this.cam.dist * 4 + S * 4);
    this.mvp = multiply(proj, view);

    gl.useProgram(this.program);
    gl.uniformMatrix4fv(this.loc.uMVP, false, this.mvp);
    const light = norm([0.45, 0.85, 0.35]);
    gl.uniform3f(this.loc.uLight, light[0], light[1], light[2]);

    const bind = (buffer) => {
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.enableVertexAttribArray(this.loc.aPos);
      gl.vertexAttribPointer(this.loc.aPos, 3, gl.FLOAT, false, 36, 0);
      gl.enableVertexAttribArray(this.loc.aNormal);
      gl.vertexAttribPointer(this.loc.aNormal, 3, gl.FLOAT, false, 36, 12);
      gl.enableVertexAttribArray(this.loc.aColor);
      gl.vertexAttribPointer(this.loc.aColor, 3, gl.FLOAT, false, 36, 24);
    };
    gl.uniform1f(this.loc.uFlat, 0);
    bind(this.meshBuffer);
    gl.drawArrays(gl.TRIANGLES, 0, this.meshCount);
    gl.uniform1f(this.loc.uFlat, 1);
    bind(this.lineBuffer);
    gl.drawArrays(gl.LINES, 0, this.lineCount);
    this.updateLabels(w, h);
  }

  updateLabels(w, h) {
    const d = this.data;
    const parts = [];
    const kTop = d.nz - this.peel;
    const dz = this.dzWorld();
    const half = (d.nx * d.cell_mm) / 2;
    if (kTop > d.dieStart && d.chiplets) {
      for (const ch of d.chiplets) {
        const cy = ch.y_mm + ch.h_mm / 2;
        if (cy < this.cut * d.cell_mm) continue;
        const p = this.project([ch.x_mm + ch.w_mm / 2 - half, kTop * dz, half - cy], w, h);
        if (p && p[0] > 0 && p[0] < w && p[1] > 0 && p[1] < h) {
          parts.push(`<div class="gl-label" style="left:${p[0]}px;top:${p[1]}px">${escapeHtml(ch.name)}</div>`);
        }
      }
    }
    if (d.peak && d.peak.k < kTop && d.peak.j >= this.cut) {
      const pk = d.peak;
      const p = this.project([(pk.i + 0.5) * d.cell_mm - half, kTop * dz, half - (pk.j + 0.5) * d.cell_mm], w, h);
      if (p && p[0] > 0 && p[0] < w && p[1] > 0 && p[1] < h) {
        parts.push(`<div class="gl-peak" style="left:${p[0]}px;top:${p[1] - 14}px">Peak ${pk.t_c.toFixed(1)} °C</div>`);
      }
    }
    this.labelsEl.innerHTML = parts.join("");
  }
}
