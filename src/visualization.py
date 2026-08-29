"""
visualization.py
RISC3D-AI :: 3D IC / Advanced Packaging Thermal Modeling Framework

ThermalVisualizer consumes the raw temperature array T[x, y, z] produced by
ThermalSolver (Kelvin) and produces:
  - Orthogonal 2D cross-sections (X, Y, Z planes) through the dynamic peak
    hotspot, via Matplotlib (always available, always succeeds).
  - Hotspot extraction: peak coordinate/value and the full list of voxels
    exceeding a configurable threshold, for the EDA "Hotspot Detection
    Vector" requirement (T > 85 degC by default).
  - High-resolution volumetric 3D thermal renders, preferentially via
    PyVista (GPU/software-accelerated volume rendering) with a fully
    automatic, non-fatal fallback to a Matplotlib 3D scatter render if
    PyVista is not installed or fails at runtime (e.g. headless env with
    no working OpenGL context). The caller never has to branch on which
    backend is available.

This module takes only plain NumPy arrays + grid spacing as input, so it
has no hard dependency on GridBuilder/ThermalSolver and can be reused for
post-processing checkpointed results.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless-safe backend; must be set before pyplot import
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("RISC3D-AI.Visualization")


# ---------------------------------------------------------------------------
# Interactive WebGL (Three.js) HTML template.
#
# Single external dependency: three.js r128 from a CDN (loaded by the user's
# browser when the file is opened -- not fetched by this Python process).
# Orbit/zoom/pan camera control is a small hand-written implementation
# (not the three.js examples/OrbitControls.js addon) specifically so this
# file only needs ONE external script tag rather than two, reducing the
# chance of a broken/mismatched-version CDN path.
#
# Placeholders are replaced via str.replace() in render_interactive_webgl(),
# NOT str.format() -- the JS/CSS body below is full of literal `{`/`}`
# characters that would collide with .format()'s substitution syntax.
# ---------------------------------------------------------------------------
_WEBGL_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>__TITLE__</title>
<style>
  html, body { margin: 0; padding: 0; overflow: hidden; background: #0a0a0f; font-family: 'Segoe UI', Arial, sans-serif; }
  #canvas-container { position: absolute; top: 0; left: 0; width: 100%; height: 100%; cursor: grab; }
  #canvas-container:active { cursor: grabbing; }
  .panel {
    position: absolute; background: rgba(15, 15, 25, 0.88); color: #e8e8f0;
    border: 1px solid #3a3a55; border-radius: 8px; padding: 14px 18px;
    font-size: 13px; line-height: 1.6; backdrop-filter: blur(4px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
  }
  #info-panel { top: 16px; left: 16px; max-width: 340px; }
  #info-panel h1 { font-size: 15px; margin: 0 0 8px 0; color: #ff9d5c; font-weight: 600; }
  #info-panel .stat { color: #9ea3b0; }
  #info-panel .stat b { color: #e8e8f0; }
  #selected-voxel { margin-top: 10px; padding-top: 10px; border-top: 1px solid #3a3a55; min-height: 60px; }
  #controls-panel { top: 16px; right: 16px; width: 260px; }
  #controls-panel h2 { font-size: 13px; margin: 0 0 10px 0; color: #ff9d5c; font-weight: 600; }
  #controls-panel label { display: block; margin-top: 10px; color: #9ea3b0; font-size: 12px; }
  #controls-panel input[type="range"] { width: 100%; }
  #controls-panel .range-val { color: #e8e8f0; font-weight: 600; }
  #colorbar-panel { bottom: 16px; left: 16px; width: 280px; }
  #colorbar-gradient {
    height: 18px; border-radius: 3px; margin-top: 6px;
    background: linear-gradient(to right, rgb(0,0,4), rgb(87,16,110), rgb(188,55,84), rgb(249,142,9), rgb(252,255,164));
  }
  #colorbar-labels { display: flex; justify-content: space-between; font-size: 11px; color: #9ea3b0; margin-top: 4px; }
  #help-panel { bottom: 16px; right: 16px; font-size: 11px; color: #8a8fa0; max-width: 260px; }
  #help-panel b { color: #b5b9c8; }
  .btn {
    background: #2a2a40; border: 1px solid #4a4a68; color: #e8e8f0; border-radius: 4px;
    padding: 4px 10px; font-size: 11px; cursor: pointer; margin-top: 8px; margin-right: 6px;
  }
  .btn:hover { background: #3a3a55; }
  #loading { position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%); color: #e8e8f0; font-size: 14px; }
</style>
</head>
<body>
<div id="canvas-container"></div>
<div id="loading">Loading Three.js and building thermal field&hellip;</div>

<div class="panel" id="info-panel">
  <h1>RISC3D-AI Thermal Field</h1>
  <div class="stat">Grid: <b>__NX__ &times; __NY__ &times; __NZ__</b> voxels</div>
  <div class="stat">Temperature range: <b>__T_MIN__&deg;C &ndash; __T_MAX__&deg;C</b></div>
  <div class="stat">Peak: <b>__PEAK_T__&deg;C</b> at index <b>__PEAK_IJK__</b></div>
  <div class="stat">__DOWNSAMPLE_NOTE__</div>
  <div class="stat">Generated __GENERATED__</div>
  <div id="selected-voxel"><i>Click a voxel to inspect its temperature.</i></div>
</div>

<div class="panel" id="controls-panel">
  <h2>Z-Layer Slice</h2>
  <label>Min layer: <span class="range-val" id="zMinLabel">0</span></label>
  <input type="range" id="zMinSlider" min="0" max="1" value="0" step="1">
  <label>Max layer: <span class="range-val" id="zMaxLabel">0</span></label>
  <input type="range" id="zMaxSlider" min="0" max="1" value="1" step="1">
  <label>Point size: <span class="range-val" id="sizeLabel">1.0</span></label>
  <input type="range" id="sizeSlider" min="0.2" max="3.0" value="1.0" step="0.1">
  <div>
    <button class="btn" id="resetViewBtn">Reset View</button>
    <button class="btn" id="resetSliceBtn">Reset Slice</button>
  </div>
</div>

<div class="panel" id="colorbar-panel">
  <div>Temperature</div>
  <div id="colorbar-gradient"></div>
  <div id="colorbar-labels"><span>__T_MIN__&deg;C</span><span>__T_MAX__&deg;C</span></div>
</div>

<div class="panel" id="help-panel">
  <b>Drag</b> to orbit &middot; <b>Scroll</b> to zoom &middot; <b>Shift+drag</b> to pan &middot; <b>Click</b> a voxel to inspect it
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const DATA = __DATA_JSON__;

// ---------------------------------------------------------------------
// Scene setup
// ---------------------------------------------------------------------
const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0f);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100000);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio || 1);
container.appendChild(renderer.domElement);

const extentX = DATA.nx * DATA.dx_um, extentY = DATA.ny * DATA.dy_um, extentZ = DATA.nz * DATA.dz_um;
const center = new THREE.Vector3(extentX / 2, extentZ / 2, extentY / 2);
const maxExtent = Math.max(extentX, extentY, extentZ, 1);

// Faint bounding box for spatial reference.
const boxGeom = new THREE.BoxGeometry(extentX, extentZ, extentY);
const boxEdges = new THREE.EdgesGeometry(boxGeom);
const boxLines = new THREE.LineSegments(boxEdges, new THREE.LineBasicMaterial({ color: 0x3a3a55 }));
boxLines.position.copy(center);
scene.add(boxLines);

const axesHelper = new THREE.AxesHelper(maxExtent * 0.15);
scene.add(axesHelper);

// ---------------------------------------------------------------------
// Custom orbit/zoom/pan camera control (no external OrbitControls dependency)
// ---------------------------------------------------------------------
const controlState = {
  target: center.clone(),
  radius: maxExtent * 2.2,
  theta: Math.PI / 4,   // azimuthal
  phi: Math.PI / 3,     // polar
  isDragging: false, isPanning: false,
  lastX: 0, lastY: 0,
};

function updateCameraFromSpherical() {
  const r = controlState.radius;
  const t = controlState.theta;
  const p = Math.max(0.05, Math.min(Math.PI - 0.05, controlState.phi));
  camera.position.x = controlState.target.x + r * Math.sin(p) * Math.cos(t);
  camera.position.y = controlState.target.y + r * Math.cos(p);
  camera.position.z = controlState.target.z + r * Math.sin(p) * Math.sin(t);
  camera.lookAt(controlState.target);
}
updateCameraFromSpherical();

renderer.domElement.addEventListener('mousedown', (e) => {
  if (e.shiftKey) { controlState.isPanning = true; } else { controlState.isDragging = true; }
  controlState.lastX = e.clientX; controlState.lastY = e.clientY;
});
window.addEventListener('mouseup', () => { controlState.isDragging = false; controlState.isPanning = false; });
window.addEventListener('mousemove', (e) => {
  const dx = e.clientX - controlState.lastX, dy = e.clientY - controlState.lastY;
  if (controlState.isDragging) {
    controlState.theta -= dx * 0.006;
    controlState.phi -= dy * 0.006;
    updateCameraFromSpherical();
  } else if (controlState.isPanning) {
    const panSpeed = controlState.radius * 0.0015;
    const right = new THREE.Vector3(); camera.getWorldDirection(right);
    const camRight = new THREE.Vector3().crossVectors(camera.up, right).normalize();
    const camUp = camera.up.clone();
    controlState.target.addScaledVector(camRight, dx * panSpeed);
    controlState.target.addScaledVector(camUp, dy * panSpeed);
    updateCameraFromSpherical();
  }
  controlState.lastX = e.clientX; controlState.lastY = e.clientY;
});
renderer.domElement.addEventListener('wheel', (e) => {
  e.preventDefault();
  controlState.radius *= (1 + Math.sign(e.deltaY) * 0.1);
  controlState.radius = Math.max(maxExtent * 0.1, Math.min(maxExtent * 8, controlState.radius));
  updateCameraFromSpherical();
}, { passive: false });

document.getElementById('resetViewBtn').addEventListener('click', () => {
  controlState.target.copy(center);
  controlState.radius = maxExtent * 2.2;
  controlState.theta = Math.PI / 4;
  controlState.phi = Math.PI / 3;
  updateCameraFromSpherical();
});

// ---------------------------------------------------------------------
// Colormap (inferno-like 5-stop gradient)
// ---------------------------------------------------------------------
const COLOR_STOPS = [
  [0.00, 0, 0, 4], [0.25, 87, 16, 110], [0.50, 188, 55, 84], [0.75, 249, 142, 9], [1.00, 252, 255, 164]
];
function colormap(v) {
  v = Math.max(0, Math.min(1, v));
  for (let i = 0; i < COLOR_STOPS.length - 1; i++) {
    const [t0, r0, g0, b0] = COLOR_STOPS[i];
    const [t1, r1, g1, b1] = COLOR_STOPS[i + 1];
    if (v >= t0 && v <= t1) {
      const f = (t1 > t0) ? (v - t0) / (t1 - t0) : 0;
      return [(r0 + (r1 - r0) * f) / 255, (g0 + (g1 - g0) * f) / 255, (b0 + (b1 - b0) * f) / 255];
    }
  }
  return [252 / 255, 255 / 255, 164 / 255];
}

// ---------------------------------------------------------------------
// Build full voxel position/color/temperature arrays once (i-major,
// j, k nested order -- matches the flat index used by the embedded data).
// ---------------------------------------------------------------------
const N = DATA.nx * DATA.ny * DATA.nz;
const allPositions = new Float32Array(N * 3);
const allColors = new Float32Array(N * 3);
const allK = new Int32Array(N);

let n = 0;
for (let i = 0; i < DATA.nx; i++) {
  for (let j = 0; j < DATA.ny; j++) {
    for (let k = 0; k < DATA.nz; k++) {
      const temp = DATA.temps_c[n];
      const v = (DATA.t_max > DATA.t_min) ? (temp - DATA.t_min) / (DATA.t_max - DATA.t_min) : 0.5;
      const [r, g, b] = colormap(v);
      allPositions[n * 3 + 0] = i * DATA.dx_um;
      allPositions[n * 3 + 1] = k * DATA.dz_um;   // Z (layer) mapped to screen-up for intuitive slicing
      allPositions[n * 3 + 2] = j * DATA.dy_um;
      allColors[n * 3 + 0] = r; allColors[n * 3 + 1] = g; allColors[n * 3 + 2] = b;
      allK[n] = k;
      n++;
    }
  }
}

const geometry = new THREE.BufferGeometry();
const material = new THREE.PointsMaterial({ size: Math.max(1.0, maxExtent * 0.006), vertexColors: true, sizeAttenuation: true });
const pointsObject = new THREE.Points(geometry, material);
scene.add(pointsObject);

let visibleIndices = [];  // maps buffer-position index -> original flat index n

function rebuildVisibleSubset(zMin, zMax) {
  visibleIndices = [];
  const positions = [], colors = [];
  for (let idx = 0; idx < N; idx++) {
    const k = allK[idx];
    if (k >= zMin && k <= zMax) {
      positions.push(allPositions[idx * 3], allPositions[idx * 3 + 1], allPositions[idx * 3 + 2]);
      colors.push(allColors[idx * 3], allColors[idx * 3 + 1], allColors[idx * 3 + 2]);
      visibleIndices.push(idx);
    }
  }
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geometry.computeBoundingSphere();
}
rebuildVisibleSubset(0, DATA.nz - 1);

// ---------------------------------------------------------------------
// Z-slice slider wiring
// ---------------------------------------------------------------------
const zMinSlider = document.getElementById('zMinSlider');
const zMaxSlider = document.getElementById('zMaxSlider');
const zMinLabel = document.getElementById('zMinLabel');
const zMaxLabel = document.getElementById('zMaxLabel');
zMinSlider.max = DATA.nz - 1; zMaxSlider.max = DATA.nz - 1; zMaxSlider.value = DATA.nz - 1;
zMinLabel.textContent = '0'; zMaxLabel.textContent = String(DATA.nz - 1);

function onSliceChange() {
  let zMin = parseInt(zMinSlider.value, 10), zMax = parseInt(zMaxSlider.value, 10);
  if (zMin > zMax) { [zMin, zMax] = [zMax, zMin]; }
  zMinLabel.textContent = String(zMin); zMaxLabel.textContent = String(zMax);
  rebuildVisibleSubset(zMin, zMax);
}
zMinSlider.addEventListener('input', onSliceChange);
zMaxSlider.addEventListener('input', onSliceChange);
document.getElementById('resetSliceBtn').addEventListener('click', () => {
  zMinSlider.value = 0; zMaxSlider.value = DATA.nz - 1; onSliceChange();
});

const sizeSlider = document.getElementById('sizeSlider');
const sizeLabel = document.getElementById('sizeLabel');
sizeSlider.addEventListener('input', () => {
  material.size = Math.max(0.1, maxExtent * 0.006 * parseFloat(sizeSlider.value));
  sizeLabel.textContent = parseFloat(sizeSlider.value).toFixed(1);
});

// ---------------------------------------------------------------------
// Click-to-inspect via raycasting
// ---------------------------------------------------------------------
const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = Math.max(1.0, maxExtent * 0.01);
const mouseNdc = new THREE.Vector2();
const selectedVoxelDiv = document.getElementById('selected-voxel');

renderer.domElement.addEventListener('click', (e) => {
  const dx = e.clientX - controlState.lastX, dy = e.clientY - controlState.lastY;
  if (Math.abs(dx) > 3 || Math.abs(dy) > 3) return;  // ignore click-after-drag

  const rect = renderer.domElement.getBoundingClientRect();
  mouseNdc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  mouseNdc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouseNdc, camera);
  const intersects = raycaster.intersectObject(pointsObject);

  if (intersects.length > 0) {
    const bufferIdx = intersects[0].index;
    const origIdx = visibleIndices[bufferIdx];
    const i = Math.floor(origIdx / (DATA.ny * DATA.nz));
    const j = Math.floor((origIdx % (DATA.ny * DATA.nz)) / DATA.nz);
    const k = origIdx % DATA.nz;
    const temp = DATA.temps_c[origIdx];
    selectedVoxelDiv.innerHTML =
      '<b>Selected voxel</b><br>' +
      'Index (i,j,k): (' + i + ', ' + j + ', ' + k + ')<br>' +
      'Position: (' + (i * DATA.dx_um).toFixed(1) + ', ' + (j * DATA.dy_um).toFixed(1) + ', ' + (k * DATA.dz_um).toFixed(1) + ') &mu;m<br>' +
      'Temperature: <b>' + temp.toFixed(2) + ' &deg;C</b>';
  }
});

// ---------------------------------------------------------------------
// Resize + render loop
// ---------------------------------------------------------------------
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

document.getElementById('loading').style.display = 'none';

function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}
animate();
</script>
</body>
</html>
"""





class VisualizationError(Exception):
    """Raised on invalid input arrays or unrecoverable rendering failures
    (i.e. failures in the Matplotlib fallback path itself, which has no
    further fallback)."""


@dataclass
class HotspotReport:
    """Structured hotspot-detection result."""
    threshold_c: float
    count: int
    peak_temperature_c: float
    peak_location_ijk: Tuple[int, int, int]
    peak_location_um: Tuple[float, float, float]
    coordinates_ijk: np.ndarray   # shape (N, 3)
    temperatures_c: np.ndarray    # shape (N,)


class ThermalVisualizer:
    """
    Post-processing and visualization layer for a single T[x,y,z] snapshot
    (Kelvin) produced by ThermalSolver.
    """

    def __init__(
        self,
        temperature_k: np.ndarray,
        dx: float,
        dy: float,
        dz: float,
        output_dir: str = "outputs",
    ):
        self._validate_input(temperature_k, dx, dy, dz)
        self.T_k = np.asarray(temperature_k, dtype=np.float64)
        self.T_c = self.T_k - 273.15
        self.dx, self.dy, self.dz = float(dx), float(dy), float(dz)
        self.nx, self.ny, self.nz = self.T_k.shape
        self.output_dir = output_dir

        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except OSError as exc:
            raise VisualizationError(f"Could not create output_dir '{self.output_dir}': {exc}") from exc

        # Physical-unit coordinate axes (micrometers), for plot extents/labels.
        self.x_um = np.arange(self.nx) * self.dx * 1e6
        self.y_um = np.arange(self.ny) * self.dy * 1e6
        self.z_um = np.arange(self.nz) * self.dz * 1e6

        logger.info(
            "ThermalVisualizer initialized: grid=%s, T range=[%.2f, %.2f] C, output_dir='%s'",
            self.T_k.shape, float(np.min(self.T_c)), float(np.max(self.T_c)), self.output_dir,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_input(temperature_k: np.ndarray, dx: float, dy: float, dz: float) -> None:
        arr = np.asarray(temperature_k)
        if arr.ndim != 3:
            raise VisualizationError(f"temperature_k must be a 3D array [x,y,z], got ndim={arr.ndim}")
        if arr.size == 0 or min(arr.shape) < 1:
            raise VisualizationError(f"temperature_k has an empty/degenerate shape {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise VisualizationError("temperature_k contains NaN/Inf values; cannot visualize a diverged field.")
        if dx <= 0 or dy <= 0 or dz <= 0:
            raise VisualizationError(f"Grid spacing must be positive, got dx={dx}, dy={dy}, dz={dz}")

    # ------------------------------------------------------------------
    # Hotspot extraction
    # ------------------------------------------------------------------
    def get_peak_hotspot(self) -> Dict[str, object]:
        """Returns the single global peak temperature location/value."""
        idx = np.unravel_index(np.argmax(self.T_c), self.T_c.shape)
        i, j, k = (int(v) for v in idx)
        return {
            "index_ijk": (i, j, k),
            "location_um": (float(self.x_um[i]), float(self.y_um[j]), float(self.z_um[k])),
            "temperature_c": float(self.T_c[i, j, k]),
        }

    def get_hotspot_report(self, threshold_c: float = 85.0) -> HotspotReport:
        """
        Full hotspot scan: every voxel with T > threshold_c, per the EDA
        'Hotspot Detection Vector' requirement (T[x,y,z] > 85 degC).
        """
        mask = self.T_c > threshold_c
        coords = np.argwhere(mask)
        temps = self.T_c[mask]

        # Sort descending by temperature so the hottest cells lead the report.
        if coords.shape[0] > 0:
            order = np.argsort(-temps)
            coords = coords[order]
            temps = temps[order]

        peak = self.get_peak_hotspot()
        if coords.shape[0] == 0:
            logger.info("No voxels exceed threshold_c=%.2f C (peak=%.2f C). Grid is within safe operating range.",
                        threshold_c, peak["temperature_c"])

        return HotspotReport(
            threshold_c=threshold_c,
            count=int(coords.shape[0]),
            peak_temperature_c=peak["temperature_c"],
            peak_location_ijk=peak["index_ijk"],
            peak_location_um=peak["location_um"],
            coordinates_ijk=coords,
            temperatures_c=temps,
        )

    def export_hotspot_csv(self, threshold_c: float = 85.0, filename: str = "hotspot_report.csv") -> str:
        """Writes the full hotspot list to CSV: i,j,k,x_um,y_um,z_um,temperature_c."""
        report = self.get_hotspot_report(threshold_c=threshold_c)
        filepath = os.path.join(self.output_dir, filename)
        try:
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["i", "j", "k", "x_um", "y_um", "z_um", "temperature_c"])
                for (i, j, k), t in zip(report.coordinates_ijk, report.temperatures_c):
                    writer.writerow([
                        int(i), int(j), int(k),
                        f"{self.x_um[i]:.3f}", f"{self.y_um[j]:.3f}", f"{self.z_um[k]:.3f}",
                        f"{t:.4f}",
                    ])
            logger.info("Wrote %d hotspot rows to %s", report.count, filepath)
            return filepath
        except OSError as exc:
            raise VisualizationError(f"Failed to write hotspot CSV to '{filepath}': {exc}") from exc

    # ------------------------------------------------------------------
    # 2D cross-sections
    # ------------------------------------------------------------------
    def _slice_indices_through_peak(self) -> Tuple[int, int, int]:
        return self.get_peak_hotspot()["index_ijk"]

    def plot_orthogonal_slices(
        self,
        i: Optional[int] = None,
        j: Optional[int] = None,
        k: Optional[int] = None,
        filename: str = "cross_sections.png",
        cmap: str = "inferno",
        dpi: int = 200,
    ) -> str:
        """
        Renders 3 side-by-side 2D cross-sections (YZ at fixed x=i, XZ at
        fixed y=j, XY at fixed z=k), defaulting to planes through the
        dynamic peak hotspot when indices are not supplied. Shared color
        scale across all 3 panels for direct visual comparison.
        """
        peak_i, peak_j, peak_k = self._slice_indices_through_peak()
        i = peak_i if i is None else i
        j = peak_j if j is None else j
        k = peak_k if k is None else k
        for name, val, bound in (("i", i, self.nx), ("j", j, self.ny), ("k", k, self.nz)):
            if not (0 <= val < bound):
                raise VisualizationError(f"Slice index {name}={val} out of bounds [0, {bound})")

        vmin, vmax = float(np.min(self.T_c)), float(np.max(self.T_c))

        try:
            fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

            # YZ plane at fixed x=i
            im0 = axes[0].imshow(
                self.T_c[i, :, :].T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                extent=[self.y_um[0], self.y_um[-1], self.z_um[0], self.z_um[-1]], aspect="auto",
            )
            axes[0].set_title(f"YZ plane @ x={self.x_um[i]:.1f} \u03bcm (i={i})")
            axes[0].set_xlabel("Y (\u03bcm)"); axes[0].set_ylabel("Z (\u03bcm)")
            axes[0].plot(self.y_um[j], self.z_um[k], marker="x", color="cyan", markersize=10, markeredgewidth=2)

            # XZ plane at fixed y=j
            im1 = axes[1].imshow(
                self.T_c[:, j, :].T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                extent=[self.x_um[0], self.x_um[-1], self.z_um[0], self.z_um[-1]], aspect="auto",
            )
            axes[1].set_title(f"XZ plane @ y={self.y_um[j]:.1f} \u03bcm (j={j})")
            axes[1].set_xlabel("X (\u03bcm)"); axes[1].set_ylabel("Z (\u03bcm)")
            axes[1].plot(self.x_um[i], self.z_um[k], marker="x", color="cyan", markersize=10, markeredgewidth=2)

            # XY plane at fixed z=k
            im2 = axes[2].imshow(
                self.T_c[:, :, k].T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                extent=[self.x_um[0], self.x_um[-1], self.y_um[0], self.y_um[-1]], aspect="auto",
            )
            axes[2].set_title(f"XY plane @ z={self.z_um[k]:.1f} \u03bcm (k={k})")
            axes[2].set_xlabel("X (\u03bcm)"); axes[2].set_ylabel("Y (\u03bcm)")
            axes[2].plot(self.x_um[i], self.y_um[j], marker="x", color="cyan", markersize=10, markeredgewidth=2)

            cbar = fig.colorbar(im2, ax=axes, shrink=0.85, pad=0.02)
            cbar.set_label("Temperature (\u00b0C)")
            fig.suptitle(
                f"RISC3D-AI Thermal Cross-Sections \u2014 Peak = {self.T_c[peak_i, peak_j, peak_k]:.2f}\u00b0C "
                f"@ (i={peak_i}, j={peak_j}, k={peak_k})",
                fontsize=13,
            )

            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=dpi)
            plt.close(fig)
            logger.info("Saved orthogonal cross-sections to %s", filepath)
            return filepath
        except Exception as exc:
            plt.close("all")
            raise VisualizationError(f"Failed to render orthogonal slices: {exc}") from exc

    def plot_temperature_histogram(
        self, filename: str = "temperature_histogram.png", bins: int = 60, dpi: int = 200,
    ) -> str:
        """Distribution of all voxel temperatures -- quick sanity check for
        bimodal hotspot separation vs. bulk substrate temperature."""
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(self.T_c.ravel(), bins=bins, color="firebrick", alpha=0.85)
            ax.axvline(float(np.max(self.T_c)), color="black", linestyle="--", label="Peak")
            ax.axvline(float(np.mean(self.T_c)), color="blue", linestyle=":", label="Mean")
            ax.set_xlabel("Temperature (\u00b0C)")
            ax.set_ylabel("Voxel count")
            ax.set_title("RISC3D-AI Grid Temperature Distribution")
            ax.legend()
            fig.tight_layout()

            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=dpi)
            plt.close(fig)
            logger.info("Saved temperature histogram to %s", filepath)
            return filepath
        except Exception as exc:
            plt.close("all")
            raise VisualizationError(f"Failed to render temperature histogram: {exc}") from exc

    # ------------------------------------------------------------------
    # 3D volumetric rendering (PyVista preferred, Matplotlib fallback)
    # ------------------------------------------------------------------
    def plot_volume(
        self,
        filename: str = "volumetric_thermal.png",
        cmap: str = "inferno",
        prefer_pyvista: bool = True,
        window_size: Tuple[int, int] = (1600, 1200),
        max_scatter_points: int = 60000,
        scatter_percentile_floor: float = 40.0,
    ) -> str:
        """
        Renders a high-resolution volumetric 3D thermal map. Tries PyVista
        first (true volume rendering with transfer-function opacity); if
        PyVista is not installed or the render fails at runtime (common in
        headless/no-GPU CI environments), automatically falls back to a
        Matplotlib 3D scatter render of the upper-percentile-temperature
        voxels, so this method always returns a valid image path.

        Parameters
        ----------
        scatter_percentile_floor : float
            In the Matplotlib fallback, only voxels at or above this
            percentile of the temperature distribution are plotted (keeps
            the scatter from being dominated by cold bulk-substrate voxels
            and keeps point counts tractable).
        """
        filepath = os.path.join(self.output_dir, filename)

        if prefer_pyvista:
            try:
                rendered_path = self._render_volume_pyvista(filepath, cmap, window_size)
                logger.info("Saved PyVista volumetric render to %s", rendered_path)
                return rendered_path
            except ImportError:
                logger.warning(
                    "PyVista not installed; falling back to Matplotlib 3D scatter rendering. "
                    "Install `pyvista` for GPU/software-accelerated true volume rendering."
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "PyVista volumetric render failed at runtime (%s); falling back to "
                    "Matplotlib 3D scatter rendering.", exc,
                )

        return self._render_volume_matplotlib(filepath, cmap, max_scatter_points, scatter_percentile_floor)

    def _render_volume_pyvista(self, filepath: str, cmap: str, window_size: Tuple[int, int]) -> str:
        import pyvista as pv  # deferred import: optional dependency

        pv.OFF_SCREEN = True
        grid = pv.ImageData(dimensions=self.T_c.shape)
        grid.spacing = (self.dx * 1e6, self.dy * 1e6, self.dz * 1e6)  # micrometers
        grid.origin = (0.0, 0.0, 0.0)
        grid.point_data["temperature_C"] = self.T_c.flatten(order="F")

        plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
        plotter.add_volume(
            grid, scalars="temperature_C", cmap=cmap, opacity="sigmoid",
            clim=[float(np.min(self.T_c)), float(np.max(self.T_c))],
            show_scalar_bar=False,
        )
        plotter.add_scalar_bar(title="Temperature (\u00b0C)", n_labels=5, fmt="%.1f")
        plotter.add_axes()
        plotter.reset_camera()
        plotter.camera.azimuth = 35
        plotter.camera.elevation = 20
        plotter.camera.zoom(1.6)
        plotter.set_background("white")
        plotter.screenshot(filepath)
        plotter.close()
        return filepath

    def _render_volume_matplotlib(
        self, filepath: str, cmap: str, max_points: int, percentile_floor: float,
    ) -> str:
        try:
            threshold_val = float(np.percentile(self.T_c, percentile_floor))
            mask = self.T_c >= threshold_val
            coords = np.argwhere(mask)
            temps = self.T_c[mask]

            if coords.shape[0] == 0:
                # Should not happen given percentile_floor < 100, but guard anyway.
                coords = np.argwhere(np.ones_like(self.T_c, dtype=bool))
                temps = self.T_c.ravel()

            if coords.shape[0] > max_points:
                rng = np.random.default_rng(seed=42)
                sel = rng.choice(coords.shape[0], size=max_points, replace=False)
                coords = coords[sel]
                temps = temps[sel]

            xs = self.x_um[coords[:, 0]]
            ys = self.y_um[coords[:, 1]]
            zs = self.z_um[coords[:, 2]]

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            sc = ax.scatter(xs, ys, zs, c=temps, cmap=cmap, s=6, alpha=0.6, edgecolors="none")
            ax.set_xlabel("X (\u03bcm)"); ax.set_ylabel("Y (\u03bcm)"); ax.set_zlabel("Z (\u03bcm)")
            ax.set_title(
                f"RISC3D-AI Volumetric Thermal Map (Matplotlib fallback)\n"
                f"Showing top {100 - percentile_floor:.0f}% hottest voxels "
                f"(n={coords.shape[0]})"
            )
            cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.1)
            cbar.set_label("Temperature (\u00b0C)")
            fig.tight_layout()
            fig.savefig(filepath, dpi=200)
            plt.close(fig)
            return filepath
        except Exception as exc:
            plt.close("all")
            raise VisualizationError(
                f"Matplotlib volumetric fallback rendering also failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Convenience: full report generation
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Interactive 3D WebGL (Three.js) export
    # ------------------------------------------------------------------
    def render_interactive_webgl(
        self,
        filename: str = "thermal_3d.html",
        max_voxels: int = 250_000,
        title: str = "RISC3D-AI :: Interactive 3D Thermal Field",
    ) -> str:
        """
        Generates a single, self-contained (except for one Three.js CDN
        script tag) interactive HTML file: an orbit/zoom/pan-able WebGL
        point-cloud render of the full 3D temperature field, with a Z-range
        slice slider and click-to-inspect voxel readout. The complete
        temperature array is embedded directly in the HTML as JSON (no
        server, no separate data file -- the .html is the whole deliverable).

        If the grid exceeds max_voxels, it is uniformly strided down (not
        cropped) so the embedded dataset and browser rendering stay
        responsive; the HTML clearly labels the render as downsampled in
        that case.

        Does NOT require PyVista/Matplotlib at all -- pure Python stdlib
        (json) generates the data payload; WebGL rendering happens entirely
        client-side in the user's browser via Three.js (loaded from a CDN
        the first time the file is opened with internet access).

        Returns the path to the written HTML file. Does not open a browser
        itself -- callers (e.g. main.py) that want auto-launch should call
        webbrowser.open() on the returned path.
        """
        try:
            nx, ny, nz = self.nx, self.ny, self.nz
            T_c = self.T_c
            dx_um, dy_um, dz_um = self.dx * 1e6, self.dy * 1e6, self.dz * 1e6

            total_voxels = nx * ny * nz
            stride = 1
            if total_voxels > max_voxels:
                stride = int(np.ceil((total_voxels / max_voxels) ** (1.0 / 3.0)))
                stride = max(stride, 1)

            T_sub = T_c[::stride, ::stride, ::stride]
            sub_nx, sub_ny, sub_nz = T_sub.shape
            downsampled = stride > 1

            temps_flat = T_sub.reshape(-1).astype(np.float64)
            t_min = float(np.min(temps_flat))
            t_max = float(np.max(temps_flat))

            peak = self.get_peak_hotspot()

            payload = {
                "nx": sub_nx, "ny": sub_ny, "nz": sub_nz,
                "dx_um": dx_um * stride, "dy_um": dy_um * stride, "dz_um": dz_um * stride,
                "temps_c": [round(float(v), 3) for v in temps_flat],
                "t_min": round(t_min, 3), "t_max": round(t_max, 3),
            }

            data_json = json.dumps(payload, separators=(",", ":"))

            downsample_note = (
                f"Downsampled {stride}x per axis ({total_voxels:,} &rarr; {sub_nx*sub_ny*sub_nz:,} voxels) "
                f"for browser performance."
                if downsampled else
                f"Full resolution ({total_voxels:,} voxels)."
            )

            html = _WEBGL_TEMPLATE
            html = html.replace("__TITLE__", title)
            html = html.replace("__DATA_JSON__", data_json)
            html = html.replace("__T_MIN__", f"{t_min:.2f}")
            html = html.replace("__T_MAX__", f"{t_max:.2f}")
            html = html.replace("__PEAK_T__", f"{peak['temperature_c']:.2f}")
            html = html.replace("__PEAK_IJK__", f"({peak['index_ijk'][0]}, {peak['index_ijk'][1]}, {peak['index_ijk'][2]})")
            html = html.replace("__NX__", str(nx)); html = html.replace("__NY__", str(ny)); html = html.replace("__NZ__", str(nz))
            html = html.replace("__DOWNSAMPLE_NOTE__", downsample_note)
            html = html.replace("__GENERATED__", time.strftime("%Y-%m-%d %H:%M:%S"))

            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html)

            logger.info("Wrote interactive WebGL 3D thermal view to %s (%d voxels embedded)",
                        filepath, sub_nx * sub_ny * sub_nz)
            return filepath

        except Exception as exc:  # noqa: BLE001
            raise VisualizationError(f"Failed to render interactive WebGL output: {exc}") from exc

    # ------------------------------------------------------------------
    def generate_full_report(self, hotspot_threshold_c: float = 85.0) -> Dict[str, object]:
        """
        Runs the complete visualization + analytics suite and returns a
        dict of generated artifact paths plus the hotspot summary, for
        main.py to log/print after a simulation run.
        """
        report = self.get_hotspot_report(threshold_c=hotspot_threshold_c)
        artifacts = {
            "cross_sections_png": self.plot_orthogonal_slices(),
            "histogram_png": self.plot_temperature_histogram(),
            "volumetric_png": self.plot_volume(),
            "hotspot_csv": self.export_hotspot_csv(threshold_c=hotspot_threshold_c),
        }
        return {
            "artifacts": artifacts,
            "hotspot_count": report.count,
            "peak_temperature_c": report.peak_temperature_c,
            "peak_location_um": report.peak_location_um,
            "peak_location_ijk": report.peak_location_ijk,
        }


if __name__ == "__main__":
    # Minimal smoke test with a synthetic Gaussian hotspot field.
    nx, ny, nz = 40, 40, 30
    xs, ys, zs = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    cx, cy, cz = nx * 0.6, ny * 0.4, nz * 0.75
    r2 = ((xs - cx) / 6.0) ** 2 + ((ys - cy) / 6.0) ** 2 + ((zs - cz) / 4.0) ** 2
    T_c = 45.0 + 60.0 * np.exp(-r2)  # ambient ~45C, peak ~105C
    T_k = T_c + 273.15

    viz = ThermalVisualizer(T_k, dx=25e-6, dy=25e-6, dz=10e-6, output_dir="viz_smoke_test")
    result = viz.generate_full_report(hotspot_threshold_c=85.0)
    print("Generated artifacts:", result["artifacts"])
    print(f"Peak: {result['peak_temperature_c']:.2f} C at {result['peak_location_um']} um "
          f"(index {result['peak_location_ijk']})")
    print(f"Hotspot voxel count (>85C): {result['hotspot_count']}")