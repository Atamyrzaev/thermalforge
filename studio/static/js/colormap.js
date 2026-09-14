// RISC3D Studio :: colour scales.
//
// Temperature uses the engineering-simulation rainbow (blue cool, red hot). Differences use a symmetric
// blue-white-red scale centred on zero, so "cooler" and "hotter" are never
// confused.

const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);

// The engineering-simulation scale: blue (coolest) -> cyan -> green -> yellow
// -> orange -> red (hottest), the ordering FEA users read at a glance, with
// every step bright enough to read on a dark interface.
const STOPS = [
  [0.0, 42, 72, 214],
  [0.17, 32, 138, 238],
  [0.33, 28, 196, 214],
  [0.5, 58, 204, 112],
  [0.66, 206, 218, 52],
  [0.83, 248, 158, 32],
  [1.0, 232, 44, 44],
];
const LUT_SIZE = 512;
const TURBO = new Uint8ClampedArray(LUT_SIZE * 3);
for (let n = 0; n < LUT_SIZE; n++) {
  const t = n / (LUT_SIZE - 1);
  let s = 1;
  while (s < STOPS.length - 1 && STOPS[s][0] < t) s++;
  const [t0, r0, g0, b0] = STOPS[s - 1];
  const [t1, r1, g1, b1] = STOPS[s];
  const f = t1 > t0 ? clamp01((t - t0) / (t1 - t0)) : 0;
  TURBO[n * 3] = r0 + (r1 - r0) * f;
  TURBO[n * 3 + 1] = g0 + (g1 - g0) * f;
  TURBO[n * 3 + 2] = b0 + (b1 - b0) * f;
}

// value in [lo, hi] -> [r, g, b] in 0..255
export function tempColor(value, lo, hi) {
  const t = hi > lo ? (value - lo) / (hi - lo) : 0.5;
  const n = Math.round(clamp01(t) * (LUT_SIZE - 1)) * 3;
  return [TURBO[n], TURBO[n + 1], TURBO[n + 2]];
}

const COOL = [47, 111, 255], MID = [236, 239, 244], HOT = [255, 77, 77];

// value in [-span, +span] -> [r, g, b] in 0..255
export function deltaColor(value, span) {
  const t = span > 0 ? Math.max(-1, Math.min(1, value / span)) : 0;
  const a = t < 0 ? COOL : HOT;
  const f = Math.abs(t);
  return [MID[0] + (a[0] - MID[0]) * f, MID[1] + (a[1] - MID[1]) * f, MID[2] + (a[2] - MID[2]) * f];
}

export function drawColorbar(canvas, kind) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
  const h = Math.max(1, Math.round(canvas.clientHeight * dpr));
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(w, h);
  for (let x = 0; x < w; x++) {
    const t = x / Math.max(1, w - 1);
    const [r, g, b] = kind === "delta" ? deltaColor(t * 2 - 1, 1) : tempColor(t, 0, 1);
    for (let y = 0; y < h; y++) {
      const o = (y * w + x) * 4;
      img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}
