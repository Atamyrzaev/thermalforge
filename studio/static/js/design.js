// RISC3D Studio :: designs on disk, and the undo history.
//
// A design is a plain JSON file: the package, the chiplets and where they sit.
// Files are treated as data, never as instructions -- only known fields are
// read, every number is checked, and anything unreadable is reported with a
// message that says what to fix.

export const DESIGN_FORMAT = "risc3d.design/1";
const MAX_CHIPLETS = 12;

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
    },
    chiplets: chiplets.map((c) => ({
      id: c.id, name: c.name, w_mm: c.w_mm, h_mm: c.h_mm,
      power_w: c.power_w, x_mm: c.x_mm, y_mm: c.y_mm, color: c.color,
    })),
  };
}

const COLOUR = /^#[0-9a-fA-F]{3,8}$/;
const DEFAULT_STACK_UM = [100, 40, 100];

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
