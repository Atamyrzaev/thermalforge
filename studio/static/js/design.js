// RISC3D Studio :: designs on disk, and the undo history.
//
// A design is a plain JSON file: the package, the chiplets and where they sit.
// Files are treated as data, never as instructions -- only known fields are
// read, every number is checked, and anything unreadable is reported with a
// message that says what to fix.

export const DESIGN_FORMAT = "risc3d.design/1";
const MAX_CHIPLETS = 16;          // matches studio/engine.py

export class History {
  constructor(limit = 60) {
    this.limit = limit;
    this.past = [];
    this.future = [];
    this.onChange = null;
  }

  get canUndo() { return this.past.length > 0; }
  get canRedo() { return this.future.length > 0; }

  // Call BEFORE a change, with the state as it is now.
  push(snapshot) {
    this.past.push(snapshot);
    if (this.past.length > this.limit) this.past.shift();
    this.future.length = 0;
    if (this.onChange) this.onChange();
  }

  undo(current) {
    if (!this.canUndo) return null;
    this.future.push(current);
    const snapshot = this.past.pop();
    if (this.onChange) this.onChange();
    return snapshot;
  }

  redo(current) {
    if (!this.canRedo) return null;
    this.past.push(current);
    const snapshot = this.future.pop();
    if (this.onChange) this.onChange();
    return snapshot;
  }

  clear() {
    this.past.length = 0;
    this.future.length = 0;
    if (this.onChange) this.onChange();
  }
}

export function serializeDesign(pkg, chiplets, title) {
  return {
    format: DESIGN_FORMAT,
    saved_utc: new Date().toISOString(),
    title: title || "Custom design",
    package: {
      size_mm: pkg.size_mm, h_w_m2k: pkg.h_w_m2k,
      ambient_c: pkg.ambient_c, pitch_mm: pkg.pitch_mm,
      stack_um: pkg.stack_um || DEFAULT_STACK_UM,
      keepout_um: pkg.keepout_um === undefined ? DEFAULT_KEEPOUT_UM : pkg.keepout_um,
      cooling: pkg.cooling || "direct",
      mesh: pkg.mesh || "standard",
      size_y_mm: pkg.size_y_mm || 0,
    },
    chiplets: chiplets.map((c) => ({
      id: c.id, name: c.name, w_mm: c.w_mm, h_mm: c.h_mm,
      power_w: c.power_w, x_mm: c.x_mm, y_mm: c.y_mm, color: c.color,
      power_pattern: c.power_pattern || "uniform",
      ...(c.blocks && c.blocks.length ? { blocks: c.blocks } : {}),
      ...(c.power_grid ? { power_grid: c.power_grid, map_source: c.map_source || "" } : {}),
    })),
  };
}

const COLOUR = /^#[0-9a-fA-F]{3,8}$/;
const DEFAULT_STACK_UM = [100, 40, 100];
const DEFAULT_KEEPOUT_UM = 100;

// Layer thicknesses, in whole 20 um mesh cells. A file without a stack is a
// file from before the stack was editable: it means the default one.
function parseStack(raw) {
  if (raw === undefined || raw === null) return [...DEFAULT_STACK_UM];
  if (!Array.isArray(raw) || raw.length !== 3) {
    throw new Error("stack_um must be three thicknesses: interposer, bumps, die");
  }
  const names = ["interposer", "bumps", "die"];
  const stack = raw.map((value, n) => {
    if (typeof value !== "number" || !Number.isFinite(value) || value < 20 || value > 400) {
      throw new Error(`the ${names[n]} layer must be between 20 and 400 µm`);
    }
    if (Math.abs(value / 20 - Math.round(value / 20)) > 1e-9) {
      throw new Error(`the ${names[n]} layer (${value} µm) must be a whole number of 20 µm cells`);
    }
    return value;
  });
  if (stack.reduce((a, b) => a + b, 0) > 600) throw new Error("the stack can total at most 600 µm");
  return stack;
}

// Internal power blocks, as fractions of the footprint. The engine checks
// them again; this keeps a bad file from reaching it with a clear message.
function parseBlocks(raw, where) {
  if (!Array.isArray(raw)) throw new Error(`${where}: blocks must be a list`);
  if (raw.length > 8) throw new Error(`${where}: at most 8 internal blocks`);
  let total = 0;
  return raw.map((block, n) => {
    if (!Array.isArray(block) || block.length !== 5
        || block.some((v) => typeof v !== "number" || !Number.isFinite(v))) {
      throw new Error(`${where}: block ${n + 1} must be [x, y, width, height, share]`);
    }
    const [x, y, w, h, share] = block;
    if (x < 0 || y < 0 || w <= 0 || h <= 0 || x + w > 1.000000001 || y + h > 1.000000001) {
      throw new Error(`${where}: block ${n + 1} must sit inside the footprint`);
    }
    total += share;
    if (share < 0 || total > 1.000000001) {
      throw new Error(`${where}: internal blocks claim more power than the chiplet has`);
    }
    return [x, y, w, h, share];
  });
}

// An imported power map: relative weights, row-major, first row at the top.
function parseGrid(raw, where) {
  if (!raw || typeof raw !== "object") throw new Error(`${where}: power_grid must be an object`);
  const { cols, rows, weights } = raw;
  if (!Number.isInteger(cols) || !Number.isInteger(rows) || cols < 1 || rows < 1
      || cols > 256 || rows > 256) {
    throw new Error(`${where}: power_grid cols and rows must be whole numbers, 1 to 256`);
  }
  if (!Array.isArray(weights) || weights.length !== cols * rows) {
    throw new Error(`${where}: power_grid needs ${cols * rows} weights (${cols} × ${rows})`);
  }
  let total = 0;
  for (const value of weights) {
    if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
      throw new Error(`${where}: power_grid values must be numbers, none of them negative`);
    }
    total += value;
  }
  if (total <= 0) throw new Error(`${where}: every tile of the power map is zero`);
  return { cols, rows, weights };
}

function number(source, key, lo, hi, where) {
  const value = source[key];
  if (typeof value !== "number" || !Number.isFinite(value) || value < lo || value > hi) {
    throw new Error(`${where}: ${key} must be a number between ${lo} and ${hi}`);
  }
  return value;
}

// Returns {pkg, chiplets, title} or throws an Error whose message names the problem.
export function parseDesign(text) {
  let raw;
  try {
    raw = JSON.parse(text);
  } catch (err) {
    throw new Error("that file is not JSON, so it is not a RISC3D design");
  }
  if (!raw || typeof raw !== "object" || raw.format !== DESIGN_FORMAT) {
    throw new Error(`that file is not a RISC3D design (expected format "${DESIGN_FORMAT}")`);
  }
  const p = raw.package;
  if (!p || typeof p !== "object") throw new Error("the file has no package");
  const pkg = {
    size_mm: number(p, "size_mm", 5, 40, "package"),
    h_w_m2k: number(p, "h_w_m2k", 10, 1e6, "package"),
    ambient_c: number(p, "ambient_c", -40, 125, "package"),
    pitch_mm: number(p, "pitch_mm", 0.5, 20, "package"),
    stack_um: parseStack(p.stack_um),
    keepout_um: p.keepout_um === undefined || p.keepout_um === null
      ? DEFAULT_KEEPOUT_UM : number(p, "keepout_um", 0, 2000, "package"),
    cooling: typeof p.cooling === "string" ? p.cooling.slice(0, 30) : "direct",
    mesh: p.mesh === "fine" ? "fine" : "standard",
    size_y_mm: Number.isFinite(p.size_y_mm) ? p.size_y_mm : 0,
  };
  if (!Array.isArray(raw.chiplets) || !raw.chiplets.length) throw new Error("the file has no chiplets");
  if (raw.chiplets.length > MAX_CHIPLETS) throw new Error(`at most ${MAX_CHIPLETS} chiplets`);
  const seen = new Set();
  const chiplets = raw.chiplets.map((c, n) => {
    const where = `chiplet ${n + 1}`;
    if (!c || typeof c !== "object") throw new Error(`${where} is not an object`);
    const id = typeof c.id === "string" ? c.id.trim().slice(0, 40) : "";
    if (!id) throw new Error(`${where} has no id`);
    if (seen.has(id)) throw new Error(`two chiplets share the id "${id}"`);
    seen.add(id);
    return {
      id,
      name: (typeof c.name === "string" && c.name.trim() ? c.name.trim() : id).slice(0, 40),
      w_mm: number(c, "w_mm", 0.5, pkg.size_mm, where),
      h_mm: number(c, "h_mm", 0.5, pkg.size_mm, where),
      power_w: number(c, "power_w", 0, 2000, where),
      x_mm: number(c, "x_mm", 0, pkg.size_mm, where),
      y_mm: number(c, "y_mm", 0, pkg.size_mm, where),
      color: typeof c.color === "string" && COLOUR.test(c.color) ? c.color : "#8aa0b8",
      power_pattern: typeof c.power_pattern === "string" ? c.power_pattern.slice(0, 30) : "uniform",
      ...(c.blocks === undefined ? {} : { blocks: parseBlocks(c.blocks, where) }),
      ...(c.power_grid === undefined || c.power_grid === null
        ? {} : { power_grid: parseGrid(c.power_grid, where),
                 map_source: typeof c.map_source === "string" ? c.map_source.slice(0, 80) : "" }),
    };
  });
  const title = typeof raw.title === "string" ? raw.title.slice(0, 80) : "Opened design";
  return { pkg, chiplets, title };
}

function slug(text) {
  return String(text).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 40) || "design";
}

export function saveDesign(pkg, chiplets, title) {
  const design = serializeDesign(pkg, chiplets, title);
  const blob = new Blob([JSON.stringify(design, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${slug(title)}.risc3d.json`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
  return design;
}

// Opens the file picker and resolves with {pkg, chiplets, title}, or null if
// the user picked nothing.
export function openDesign(input) {
  return new Promise((resolve, reject) => {
    input.value = "";
    input.onchange = () => {
      const file = input.files && input.files[0];
      if (!file) { resolve(null); return; }
      const reader = new FileReader();
      reader.onerror = () => reject(new Error(`could not read ${file.name}`));
      reader.onload = () => {
        try {
          resolve(parseDesign(String(reader.result)));
        } catch (err) {
          reject(new Error(`${file.name}: ${err.message}`));
        }
      };
      reader.readAsText(file);
    };
    input.click();
  });
}
