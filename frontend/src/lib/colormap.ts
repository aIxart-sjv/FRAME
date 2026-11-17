/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Small colour helpers for rendering a decoded raster band to a canvas -- a display transform only, exactly
 * the same category as the backend's own visualisation normalisation. No scientific value is computed here.
 */

export type RGB = [number, number, number];

/** The app's own accent gradient (already used decoratively in the legend chips before this was wired to real data). */
const NDVI_LOW: RGB = [185, 28, 28]; // #b91c1c
const NDVI_MID: RGB = [216, 184, 76]; // #D8B84C
const NDVI_HIGH: RGB = [22, 163, 74]; // #16a34a

const SEQUENTIAL_LOW: RGB = [15, 22, 32]; // near-background
const SEQUENTIAL_HIGH: RGB = [216, 184, 76]; // #D8B84C

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function lerpColor(a: RGB, b: RGB, t: number): RGB {
  return [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
}

/** value in [0, 1] -> a single-hue ramp from the dark background to the gold accent (uncertainty / |NDVI diff| maps). */
export function sequentialColor(t: number): RGB {
  const clamped = Math.min(1, Math.max(0, t));
  return lerpColor(SEQUENTIAL_LOW, SEQUENTIAL_HIGH, clamped);
}

/** value in [-1, 1] -> the app's red / gold / green NDVI ramp. */
export function ndviColor(value: number): RGB {
  const clamped = Math.min(1, Math.max(-1, value));
  if (clamped <= 0) return lerpColor(NDVI_LOW, NDVI_MID, (clamped + 1));
  return lerpColor(NDVI_MID, NDVI_HIGH, clamped);
}

/** Robust display range for a band: the [lowPct, highPct] percentile of its finite values, ignoring NaN (masked pixels). */
export function percentileRange(band: Float32Array, lowPct = 2, highPct = 98): [number, number] {
  const finite: number[] = [];
  for (let i = 0; i < band.length; i += 1) {
    const v = band[i];
    if (Number.isFinite(v)) finite.push(v);
  }
  if (finite.length === 0) return [0, 1];
  finite.sort((a, b) => a - b);
  const at = (p: number) => finite[Math.min(finite.length - 1, Math.max(0, Math.round((p / 100) * (finite.length - 1))))];
  const lo = at(lowPct);
  const hi = at(highPct);
  return hi > lo ? [lo, hi] : [lo, lo + 1];
}

export function normalize(value: number, lo: number, hi: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, (value - lo) / (hi - lo)));
}
