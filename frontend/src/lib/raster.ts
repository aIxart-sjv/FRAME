/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Canvas rendering of a decoded raster (src/lib/geotiff.ts) -- a percentile-stretched RGB composite, a single-band
 * sequential heatmap (stability, |NDVI difference|), and an NDVI diverging map. Display only; no analysis.
 */

import { normalize, ndviColor, percentileRange, sequentialColor } from './colormap';
import type { DecodedRaster } from './geotiff';
import { BAND_INDEX } from './geotiff';

function setCanvasSize(canvas: HTMLCanvasElement, width: number, height: number): CanvasRenderingContext2D {
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2D canvas context is not available.');
  return ctx;
}

/** A natural-colour composite from the raster's B04/B03/B02 bands, each independently percentile-stretched.
 * `bandIndices` says which decoded band is which -- it defaults to the canonical RGBN order the backend always
 * writes for its own outputs, but a user's own uploaded file can store its bands in any order (the API is
 * explicitly "order-independent"), so the caller must pass the real mapping (from the upload response's
 * `band_names`) when rendering that file directly in the browser. */
export function renderRgbComposite(canvas: HTMLCanvasElement, raster: DecodedRaster, bandIndices: { red: number; green: number; blue: number } = { red: BAND_INDEX.RED, green: BAND_INDEX.GREEN, blue: BAND_INDEX.BLUE }): void {
  const { width, height } = raster;
  const ctx = setCanvasSize(canvas, width, height);
  const image = ctx.createImageData(width, height);
  const red = raster.bands[bandIndices.red];
  const green = raster.bands[bandIndices.green];
  const blue = raster.bands[bandIndices.blue];
  const [rLo, rHi] = percentileRange(red);
  const [gLo, gHi] = percentileRange(green);
  const [bLo, bHi] = percentileRange(blue);
  for (let i = 0; i < width * height; i += 1) {
    const alpha = Number.isFinite(red[i]) && Number.isFinite(green[i]) && Number.isFinite(blue[i]) ? 255 : 0;
    image.data[i * 4] = Math.round(normalize(red[i], rLo, rHi) * 255);
    image.data[i * 4 + 1] = Math.round(normalize(green[i], gLo, gHi) * 255);
    image.data[i * 4 + 2] = Math.round(normalize(blue[i], bLo, bHi) * 255);
    image.data[i * 4 + 3] = alpha;
  }
  ctx.putImageData(image, 0, 0);
}

/** A single band rendered as a sequential heatmap (dark -> gold), percentile-stretched. Returns the [lo, hi] range used, for a legend. */
export function renderSequentialBand(canvas: HTMLCanvasElement, band: Float32Array, width: number, height: number): [number, number] {
  const ctx = setCanvasSize(canvas, width, height);
  const image = ctx.createImageData(width, height);
  const [lo, hi] = percentileRange(band);
  for (let i = 0; i < width * height; i += 1) {
    const finite = Number.isFinite(band[i]);
    const [r, g, b] = sequentialColor(normalize(band[i], lo, hi));
    image.data[i * 4] = r;
    image.data[i * 4 + 1] = g;
    image.data[i * 4 + 2] = b;
    image.data[i * 4 + 3] = finite ? 255 : 0;
  }
  ctx.putImageData(image, 0, 0);
  return [lo, hi];
}

/** A single NDVI band ([-1, 1]) rendered with the app's red/gold/green ramp. */
export function renderNdviBand(canvas: HTMLCanvasElement, band: Float32Array, width: number, height: number): void {
  const ctx = setCanvasSize(canvas, width, height);
  const image = ctx.createImageData(width, height);
  for (let i = 0; i < width * height; i += 1) {
    const finite = Number.isFinite(band[i]);
    const [r, g, b] = ndviColor(finite ? band[i] : 0);
    image.data[i * 4] = r;
    image.data[i * 4 + 1] = g;
    image.data[i * 4 + 2] = b;
    image.data[i * 4 + 3] = finite ? 255 : 0;
  }
  ctx.putImageData(image, 0, 0);
}

/** The real reflectance value of every band at a canvas-space click, mapped back to raster pixel coordinates (canvas is drawn 1:1 with the raster's own width/height, scaled by CSS only). */
export function samplePixel(raster: DecodedRaster, canvas: HTMLCanvasElement, clientX: number, clientY: number): { x: number; y: number; values: number[] } | null {
  const rect = canvas.getBoundingClientRect();
  const x = Math.floor(((clientX - rect.left) / rect.width) * raster.width);
  const y = Math.floor(((clientY - rect.top) / rect.height) * raster.height);
  if (x < 0 || y < 0 || x >= raster.width || y >= raster.height) return null;
  const idx = y * raster.width + x;
  return { x, y, values: raster.bands.map((band) => band[idx]) };
}

/** NDVI = (NIR - Red) / (NIR + Red), the exact formula the backend uses (frame.consistency / frame.analysis). `null` where the denominator is not usably away from zero. */
export function ndviAt(values: number[]): number | null {
  const red = values[BAND_INDEX.RED];
  const nir = values[BAND_INDEX.NIR];
  if (!Number.isFinite(red) || !Number.isFinite(nir)) return null;
  const denom = nir + red;
  if (Math.abs(denom) < 1e-6) return null;
  return (nir - red) / denom;
}
