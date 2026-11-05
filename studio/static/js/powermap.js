// ThermalForge Studio :: power maps from a design team's own file.
//
// The four built-in patterns are guesses about what is inside a chiplet. A
// design team does not have to guess: they have the numbers. This reads the
// two shapes those numbers usually arrive in and turns either into one thing
// the engine understands -- a grid of relative weights over the footprint.
//
//   1. A MATRIX: rows of numbers, one per tile, laid out like the chip.
//        0.4, 0.4, 1.0, 1.0
//        0.4, 2.8, 3.1, 1.0
//      Read exactly as written: the first row is the TOP of the chiplet.
//
//   2. A TABLE of regions, with a header naming the columns:
//        block,x_mm,y_mm,w_mm,h_mm,power_w
//        mac0,0.5,0.5,2.0,2.0,4.2
//      Rasterized onto a fine grid, so any number of regions works and
//      overlapping ones simply add.
//
// Values are relative: the engine normalizes them over the chiplet's power,
// so watts, milliwatts, percentages or activity counts all work. The file's
// own total is reported anyway, because if it IS in watts the user probably
// wants the chiplet set to it.

const MAX_SIDE = 256;
const MAX_TILES = 8192;
const TILE_TARGET_MM = 0.1;      // rasterizing a table: aim for 0.1 mm tiles

const HEADERS = {
  x: ["x", "x_mm", "x_um", "xmm", "xum", "left", "x0"],
  y: ["y", "y_mm", "y_um", "ymm", "yum", "bottom", "y0"],
  w: ["w", "w_mm", "w_um", "width", "width_mm", "width_um", "dx"],
  h: ["h", "h_mm", "h_um", "height", "height_mm", "height_um", "dy"],
  p: ["p", "power", "power_w", "power_mw", "w_power", "watts", "share", "weight", "value", "density"],
};

function splitRows(text) {
  return String(text)
    .replace(/^﻿/, "")                       // a spreadsheet's byte-order mark
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith("#"));
}

function splitCells(line) {
  return line.split(/[,;\t]|\s{2,}|\s+(?=[-+\d.])/).map((cell) => cell.trim()).filter((cell) => cell !== "");
}

const isNumber = (cell) => cell !== "" && Number.isFinite(Number(cell));

function headerRole(cell) {
  const key = cell.toLowerCase().replace(/[^a-z0-9_]/g, "");
  for (const [role, names] of Object.entries(HEADERS)) {
    if (names.includes(key)) return { role, unit: key.endsWith("_um") || key.endsWith("um") ? "um" : "mm", key };
  }
  return null;
}

// ---------------------------------------------------------------- matrix
function parseMatrix(rows) {
  const values = [];
  let cols = null;
  rows.forEach((line, n) => {
    const cells = splitCells(line);
    if (!cells.every(isNumber)) {
      throw new Error(`line ${n + 1} is not all numbers: "${line.slice(0, 40)}"`);
    }
    if (cols === null) cols = cells.length;
    if (cells.length !== cols) {
      throw new Error(`line ${n + 1} has ${cells.length} values but the first row has ${cols}; `
        + "a power matrix must be rectangular");
    }
    for (const cell of cells) {
      const value = Number(cell);
      if (value < 0) throw new Error(`line ${n + 1} has a negative value; a region cannot use less than no power`);
      values.push(value);
    }
  });
  const grid = { cols, rows: rows.length, weights: values };
  checkSize(grid);
  return { grid, total: values.reduce((a, b) => a + b, 0), form: "matrix" };
}

// ----------------------------------------------------------------- table
function parseTable(headerCells, rows, chiplet) {
  const roles = headerCells.map(headerRole);
  const index = {};
  roles.forEach((entry, n) => { if (entry && index[entry.role] === undefined) index[entry.role] = n; });
  for (const role of ["x", "y", "w", "h", "p"]) {
    if (index[role] === undefined) {
      throw new Error("a region table needs columns for x, y, width, height and power "
        + `(got: ${headerCells.join(", ")})`);
    }
  }
  const scale = roles[index.x] && roles[index.x].unit === "um" ? 1e-3 : 1;
  const cols = Math.max(8, Math.min(MAX_SIDE, Math.round(chiplet.w_mm / TILE_TARGET_MM)));
  const rowsN = Math.max(8, Math.min(MAX_SIDE, Math.round(chiplet.h_mm / TILE_TARGET_MM)));
  const weights = new Float64Array(cols * rowsN);
  let total = 0;
  let placed = 0;

  rows.forEach((line, n) => {
    const cells = splitCells(line);
    const read = (role) => Number(cells[index[role]]);
    const x = read("x") * scale, y = read("y") * scale;
    const w = read("w") * scale, h = read("h") * scale;
    const power = read("p");
    if (![x, y, w, h, power].every(Number.isFinite)) {
      throw new Error(`region on line ${n + 2} has a value that is not a number`);
    }
    if (w <= 0 || h <= 0) throw new Error(`region on line ${n + 2} has no area`);
    if (power < 0) throw new Error(`region on line ${n + 2} has negative power`);
    if (x < -1e-9 || y < -1e-9 || x + w > chiplet.w_mm + 1e-6 || y + h > chiplet.h_mm + 1e-6) {
      throw new Error(`region on line ${n + 2} (${x}, ${y}, ${w} × ${h} mm) falls outside the `
        + `${chiplet.w_mm} × ${chiplet.h_mm} mm chiplet`);
    }
    total += power;
    // Tiles whose centre falls in the region share its power. Overlapping
    // regions add, which is what a table of blocks means.
    const covered = [];
    for (let c = 0; c < cols; c++) {
      const cx = (c + 0.5) / cols * chiplet.w_mm;
      if (cx < x || cx > x + w) continue;
      for (let r = 0; r < rowsN; r++) {
        const cy = (1 - (r + 0.5) / rowsN) * chiplet.h_mm;     // first row is the top
        if (cy < y || cy > y + h) continue;
        covered.push(r * cols + c);
      }
    }
    if (!covered.length) {
      // Smaller than a tile: put it in the tile holding its centre, so its
      // power is modelled rather than quietly dropped.
      const c = Math.min(cols - 1, Math.max(0, Math.floor((x + w / 2) / chiplet.w_mm * cols)));
      const r = Math.min(rowsN - 1, Math.max(0, Math.floor((1 - (y + h / 2) / chiplet.h_mm) * rowsN)));
      covered.push(r * cols + c);
    }
    for (const cell of covered) weights[cell] += power / covered.length;
    placed++;
  });

  if (!placed) throw new Error("the table has a header but no regions");
  const grid = { cols, rows: rowsN, weights: Array.from(weights) };
  checkSize(grid);
  return { grid, total, form: "table", regions: placed };
}

function checkSize(grid) {
  if (grid.cols > MAX_SIDE || grid.rows > MAX_SIDE) {
    throw new Error(`a power map may be at most ${MAX_SIDE} tiles on a side`);
  }
  if (grid.cols * grid.rows > MAX_TILES) {
    throw new Error(`a power map may have at most ${MAX_TILES} tiles`);
  }
  if (!grid.weights.some((v) => v > 0)) throw new Error("every tile in the map is zero");
}

/**
 * Reads either form. `chiplet` supplies the footprint a region table is
 * measured against. Returns {grid, total, form} or throws an Error whose
 * message says what to fix.
 */
export function parsePowerMap(text, chiplet) {
  const rows = splitRows(text);
  if (!rows.length) throw new Error("the file is empty");
  const first = splitCells(rows[0]);
  if (!first.length) throw new Error("the first line has no values");
  const looksLikeHeader = !first.every(isNumber);
  if (looksLikeHeader) {
    if (rows.length < 2) throw new Error("the file has a header but no rows under it");
    return parseTable(first, rows.slice(1), chiplet);
  }
  return parseMatrix(rows);
}

// An image is read by BRIGHTNESS: black is no power, white is the most.
// Downsampled by averaging to at most IMAGE_SIDE tiles a side -- far finer
// than any mesh the app builds, and averaging rather than picking pixels so a
// hotspot drawn thinner than a tile is not lost between samples.
const IMAGE_SIDE = 64;
// Above this mean saturation the picture is a COLOUR heatmap, and a colour
// heatmap cannot be read by brightness: in a jet or turbo colour scale yellow
// is brighter than red, so reading luminance would put the hotspot in the
// wrong place with no error. It is refused and the user is asked for a
// greyscale export, which is unambiguous.
const MAX_MEAN_SATURATION = 0.15;

/**
 * Reads a greyscale image as a power map. Returns a Promise of
 * {grid, total: null, form: "image", width, height}.
 */
export function parsePowerImage(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error("not an image the browser can read")); };
    img.onload = () => {
      try {
        const w = img.naturalWidth, h = img.naturalHeight;
        if (!w || !h) throw new Error("the image is empty");
        const cols = Math.max(1, Math.min(IMAGE_SIDE, w));
        const rows = Math.max(1, Math.min(IMAGE_SIDE, h));
        const canvas = document.createElement("canvas");
        canvas.width = w; canvas.height = h;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0);
        const px = ctx.getImageData(0, 0, w, h).data;

        let satSum = 0;
        const sums = new Float64Array(cols * rows), counts = new Float64Array(cols * rows);
        for (let y = 0; y < h; y++) {
          // Image row 0 is the TOP, which is exactly how a power grid is
          // stored, so no flip is needed -- the same convention as a CSV.
          const r = Math.min(rows - 1, Math.floor(y / h * rows));
          for (let x = 0; x < w; x++) {
            const c = Math.min(cols - 1, Math.floor(x / w * cols));
            const i = (y * w + x) * 4;
            const R = px[i], G = px[i + 1], B = px[i + 2], A = px[i + 3] / 255;
            const hi = Math.max(R, G, B), lo = Math.min(R, G, B);
            satSum += hi > 0 ? (hi - lo) / hi : 0;
            // Rec. 709 luminance, weighted by alpha so a transparent pixel is
            // no power rather than white.
            const lum = (0.2126 * R + 0.7152 * G + 0.0722 * B) / 255 * A;
            sums[r * cols + c] += lum;
            counts[r * cols + c] += 1;
          }
        }
        const saturation = satSum / (w * h);
        if (saturation > MAX_MEAN_SATURATION) {
          throw new Error(
            `this looks like a COLOUR heatmap (mean saturation ${(100 * saturation).toFixed(0)} %). ` +
            "Brightness does not track power in a colour scale -- yellow is brighter than red -- " +
            "so it cannot be read without that scale. Export it in greyscale, or as CSV.");
        }
        const weights = Array.from(sums, (v, n) => (counts[n] ? v / counts[n] : 0));
        const grid = { cols, rows, weights };
        checkSize(grid);
        if (!weights.some((v) => v > 0)) throw new Error("the image is black everywhere: no power");
        resolve({ grid, total: null, form: "image", width: w, height: h });
      } catch (err) {
        reject(err);
      } finally {
        URL.revokeObjectURL(url);
      }
    };
    img.src = url;
  });
}

/** Draws a map into a canvas so an upside-down file is visible, not believed. */
export function drawPowerMap(canvas, grid, colour) {
  const ctx = canvas.getContext("2d");
  const { cols, rows, weights } = grid;
  const peak = Math.max(...weights) || 1;
  const off = document.createElement("canvas");
  off.width = cols; off.height = rows;
  const image = off.getContext("2d").createImageData(cols, rows);
  for (let n = 0; n < cols * rows; n++) {
    const [r, g, b] = colour(weights[n] / peak);
    image.data[n * 4] = r; image.data[n * 4 + 1] = g; image.data[n * 4 + 2] = b;
    image.data[n * 4 + 3] = 255;
  }
  off.getContext("2d").putImageData(image, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
}

export function readPowerMapFile(input, chiplet) {
  return new Promise((resolve, reject) => {
    input.value = "";
    input.onchange = () => {
      const file = input.files && input.files[0];
      if (!file) { resolve(null); return; }
      if ((file.type && file.type.startsWith("image/")) || /\.(png|jpe?g|bmp|gif|webp)$/i.test(file.name)) {
        parsePowerImage(file)
          .then((parsed) => resolve({ ...parsed, filename: file.name }))
          .catch((err) => reject(new Error(`${file.name}: ${err.message}`)));
        return;
      }
      const reader = new FileReader();
      reader.onerror = () => reject(new Error(`could not read ${file.name}`));
      reader.onload = () => {
        try {
          const parsed = parsePowerMap(String(reader.result), chiplet);
          resolve({ ...parsed, filename: file.name });
        } catch (err) {
          reject(new Error(`${file.name}: ${err.message}`));
        }
      };
      reader.readAsText(file);
    };
    input.click();
  });
}
