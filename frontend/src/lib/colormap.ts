/**
 * Display-only color mapping. Every function here maps an already-computed
 * numeric value to a pixel color for rendering -- it never derives a new
 * scientific quantity. This mirrors `frame.uncertainty.statistics.
 * normalize_for_visualization`'s own stated boundary ("display only, never
 * scientific"): percentile clipping and rescaling for legibility on screen
 * is a rendering technique, not a measurement.
 */

export type RGB = [number, number, number]

/** Percentile range of finite values in `data`, ignoring NaN -- the same
 * 2nd/98th-percentile convention `experiments/*` already use for their own
 * matplotlib previews (see e.g. experiments/baseline/run_baseline.py). */
export function percentileRange(data: Float32Array, lowPct = 2, highPct = 98): [number, number] {
  const finite: number[] = []
  for (let i = 0; i < data.length; i++) {
    const v = data[i]
    if (Number.isFinite(v)) finite.push(v)
  }
  if (finite.length === 0) return [0, 1]
  finite.sort((a, b) => a - b)
  const lo = finite[Math.max(0, Math.floor((lowPct / 100) * (finite.length - 1)))]
  const hi = finite[Math.min(finite.length - 1, Math.ceil((highPct / 100) * (finite.length - 1)))]
  return hi > lo ? [lo, hi] : [lo, lo + 1]
}

export function normalize(value: number, lo: number, hi: number): number {
  if (!Number.isFinite(value)) return 0
  return Math.min(1, Math.max(0, (value - lo) / (hi - lo)))
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

export function lerpColor(c0: RGB, c1: RGB, t: number): RGB {
  return [lerp(c0[0], c1[0], t), lerp(c0[1], c1[1], t), lerp(c0[2], c1[2], t)]
}

/** Two-stop sequential ramp: `low` at t=0 -> `high` at t=1. Used for the
 * uncertainty map and the NDVI absolute-difference map -- both are
 * magnitude-only (>= 0) quantities with no natural diverging midpoint. */
export function sequentialColor(t: number, low: RGB, high: RGB): RGB {
  return lerpColor(low, high, Math.min(1, Math.max(0, t)))
}

export const UNCERTAINTY_LOW: RGB = [10, 13, 18]
export const UNCERTAINTY_HIGH: RGB = [224, 164, 88] // --uncertainty

export const NDVI_DIFF_LOW: RGB = [10, 13, 18]
export const NDVI_DIFF_HIGH: RGB = [199, 107, 209] // violet, distinct from the amber uncertainty ramp

/** Fixed-domain vegetation-index ramp (display only). NDVI has an absolute,
 * physically meaningful scale, so -- unlike raw reflectance RGB, which is
 * per-image percentile-stretched for legibility -- the SAME fixed domain is
 * used for both the native and SR-derived NDVI renders. Without this they
 * would not be visually comparable. Domain narrowed to [-0.3, 0.9], the
 * range that actually carries visual information for vegetated/bare/water
 * scenes; values outside it are clamped, not rescaled. */
const NDVI_STOPS: { value: number; color: RGB }[] = [
  { value: -0.3, color: [61, 46, 34] }, // bare soil / water
  { value: 0.0, color: [176, 158, 84] }, // sparse / senescent
  { value: 0.3, color: [156, 189, 74] }, // moderate vegetation
  { value: 0.6, color: [88, 154, 63] }, // dense vegetation
  { value: 0.9, color: [27, 94, 32] }, // very dense vegetation
]

export function ndviColor(value: number): RGB {
  if (!Number.isFinite(value)) return [30, 34, 40]
  const clamped = Math.min(NDVI_STOPS[NDVI_STOPS.length - 1].value, Math.max(NDVI_STOPS[0].value, value))
  for (let i = 0; i < NDVI_STOPS.length - 1; i++) {
    const a = NDVI_STOPS[i]
    const b = NDVI_STOPS[i + 1]
    if (clamped >= a.value && clamped <= b.value) {
      const t = (clamped - a.value) / (b.value - a.value)
      return lerpColor(a.color, b.color, t)
    }
  }
  return NDVI_STOPS[NDVI_STOPS.length - 1].color
}
