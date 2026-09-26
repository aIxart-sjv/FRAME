/**
 * Canvas drawing routines. Each function fills a 2D rendering context with
 * pixels derived from already-decoded band arrays (src/lib/geotiff.ts) via
 * a display-only color transform (src/lib/colormap.ts). No geospatial
 * reprojection, tiling, or basemap -- see frame/api/README.md's "Scope"
 * note; this is a raster preview, not a GIS viewer.
 */

import { NDVI_DIFF_HIGH, NDVI_DIFF_LOW, UNCERTAINTY_HIGH, UNCERTAINTY_LOW, ndviColor, normalize, percentileRange, sequentialColor, type RGB } from './colormap'

function ensureCanvasSize(canvas: HTMLCanvasElement, width: number, height: number): void {
  if (canvas.width !== width) canvas.width = width
  if (canvas.height !== height) canvas.height = height
}

export interface RgbBands {
  red: Float32Array
  green: Float32Array
  blue: Float32Array
}

/** Percentile-stretched true-color composite -- the same 2/98 stretch
 * convention `experiments/*`'s own preview scripts use, reimplemented here
 * only as a rendering step (raw reflectance floats have no display range
 * of their own). */
export function renderRgbComposite(canvas: HTMLCanvasElement, width: number, height: number, bands: RgbBands): void {
  ensureCanvasSize(canvas, width, height)
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const [rLo, rHi] = percentileRange(bands.red)
  const [gLo, gHi] = percentileRange(bands.green)
  const [bLo, bHi] = percentileRange(bands.blue)

  const image = ctx.createImageData(width, height)
  for (let i = 0; i < width * height; i++) {
    const r = normalize(bands.red[i], rLo, rHi)
    const g = normalize(bands.green[i], gLo, gHi)
    const b = normalize(bands.blue[i], bLo, bHi)
    image.data[i * 4] = Math.round(r * 255)
    image.data[i * 4 + 1] = Math.round(g * 255)
    image.data[i * 4 + 2] = Math.round(b * 255)
    image.data[i * 4 + 3] = 255
  }
  ctx.putImageData(image, 0, 0)
}

export type SequentialKind = 'uncertainty' | 'ndvi-diff'

const SEQUENTIAL_RAMPS: Record<SequentialKind, { low: RGB; high: RGB }> = {
  uncertainty: { low: UNCERTAINTY_LOW, high: UNCERTAINTY_HIGH },
  'ndvi-diff': { low: NDVI_DIFF_LOW, high: NDVI_DIFF_HIGH },
}

/** Single-hue magnitude heatmap (uncertainty, or |NDVI difference|),
 * percentile-scaled so the ramp uses the full range this particular scene
 * actually produced. Returns the [lo, hi] range used, so a legend can
 * display it. */
export function renderSequentialHeatmap(canvas: HTMLCanvasElement, width: number, height: number, band: Float32Array, kind: SequentialKind): [number, number] {
  ensureCanvasSize(canvas, width, height)
  const ctx = canvas.getContext('2d')
  const [lo, hi] = percentileRange(band, 2, 98)
  if (!ctx) return [lo, hi]

  const ramp = SEQUENTIAL_RAMPS[kind]
  const image = ctx.createImageData(width, height)
  for (let i = 0; i < width * height; i++) {
    const t = normalize(band[i], lo, hi)
    const [r, g, b] = sequentialColor(t, ramp.low, ramp.high)
    image.data[i * 4] = Math.round(r)
    image.data[i * 4 + 1] = Math.round(g)
    image.data[i * 4 + 2] = Math.round(b)
    image.data[i * 4 + 3] = Number.isFinite(band[i]) ? 255 : 0
  }
  ctx.putImageData(image, 0, 0)
  return [lo, hi]
}

/** Fixed-domain NDVI vegetation ramp -- see colormap.ts for why the domain
 * is fixed rather than per-image stretched. */
export function renderNdvi(canvas: HTMLCanvasElement, width: number, height: number, band: Float32Array): void {
  ensureCanvasSize(canvas, width, height)
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const image = ctx.createImageData(width, height)
  for (let i = 0; i < width * height; i++) {
    const value = band[i]
    const [r, g, b] = ndviColor(value)
    image.data[i * 4] = Math.round(r)
    image.data[i * 4 + 1] = Math.round(g)
    image.data[i * 4 + 2] = Math.round(b)
    image.data[i * 4 + 3] = Number.isFinite(value) ? 255 : 0
  }
  ctx.putImageData(image, 0, 0)
}

/** RGB base image with a semi-transparent uncertainty overlay -- alpha is
 * modulated per-pixel by the normalized uncertainty value itself (low
 * uncertainty -> the base image shows through clean; high uncertainty ->
 * the amber overlay dominates). This is what makes "uncertainty is
 * spatially localized" visually obvious, without a binary mask. */
export function renderUncertaintyOverlay(canvas: HTMLCanvasElement, width: number, height: number, bands: RgbBands, uncertaintyBand: Float32Array, maxOverlayAlpha = 0.85): [number, number] {
  ensureCanvasSize(canvas, width, height)
  const ctx = canvas.getContext('2d')
  const [uLo, uHi] = percentileRange(uncertaintyBand, 2, 98)
  if (!ctx) return [uLo, uHi]

  const [rLo, rHi] = percentileRange(bands.red)
  const [gLo, gHi] = percentileRange(bands.green)
  const [bLo, bHi] = percentileRange(bands.blue)

  const image = ctx.createImageData(width, height)
  for (let i = 0; i < width * height; i++) {
    const baseR = normalize(bands.red[i], rLo, rHi) * 255
    const baseG = normalize(bands.green[i], gLo, gHi) * 255
    const baseB = normalize(bands.blue[i], bLo, bHi) * 255

    const t = normalize(uncertaintyBand[i], uLo, uHi)
    const alpha = t * maxOverlayAlpha
    const [or_, og, ob] = sequentialColor(t, UNCERTAINTY_LOW, UNCERTAINTY_HIGH)

    image.data[i * 4] = Math.round(baseR * (1 - alpha) + or_ * alpha)
    image.data[i * 4 + 1] = Math.round(baseG * (1 - alpha) + og * alpha)
    image.data[i * 4 + 2] = Math.round(baseB * (1 - alpha) + ob * alpha)
    image.data[i * 4 + 3] = 255
  }
  ctx.putImageData(image, 0, 0)
  return [uLo, uHi]
}
