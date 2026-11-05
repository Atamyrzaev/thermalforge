// ThermalForge Studio :: talking to the local engine.

async function parse(response) {
  let data;
  try {
    data = await response.json();
  } catch (err) {
    throw new Error(`server returned ${response.status} without JSON`);
  }
  if (!response.ok && !data.error) data.error = `server returned ${response.status}`;
  return data;
}

export async function getJSON(url) {
  return parse(await fetch(url, { cache: "no-store" }));
}

export async function postJSON(url, body) {
  return parse(await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  }));
}

// Temperatures arrive as little-endian float32, C-order [i][j][k].
export function decodeField(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

export const cellIndex = (i, j, k, ny, nz) => (i * ny + j) * nz + k;
