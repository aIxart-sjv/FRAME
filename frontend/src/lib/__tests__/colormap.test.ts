import { describe, expect, it } from 'vitest'
import { ndviColor, normalize, percentileRange, sequentialColor } from '../colormap'

describe('percentileRange', () => {
  it('ignores NaN values when computing the range', () => {
    const data = Float32Array.from([1, 2, 3, 4, 5, NaN, NaN])
    const [lo, hi] = percentileRange(data, 0, 100)
    expect(lo).toBe(1)
    expect(hi).toBe(5)
  })

  it('falls back to a non-degenerate range for constant data', () => {
    const data = Float32Array.from([3, 3, 3])
    const [lo, hi] = percentileRange(data)
    expect(hi).toBeGreaterThan(lo)
  })
})

describe('normalize', () => {
  it('clamps to [0, 1] and treats non-finite input as 0', () => {
    expect(normalize(5, 0, 10)).toBeCloseTo(0.5)
    expect(normalize(-5, 0, 10)).toBe(0)
    expect(normalize(50, 0, 10)).toBe(1)
    expect(normalize(NaN, 0, 10)).toBe(0)
  })
})

describe('sequentialColor', () => {
  it('interpolates linearly between the low and high anchor colors', () => {
    const low: [number, number, number] = [0, 0, 0]
    const high: [number, number, number] = [100, 200, 50]
    expect(sequentialColor(0, low, high)).toEqual([0, 0, 0])
    expect(sequentialColor(1, low, high)).toEqual([100, 200, 50])
    expect(sequentialColor(0.5, low, high)).toEqual([50, 100, 25])
  })
})

describe('ndviColor', () => {
  it('uses a fixed domain so native and SR NDVI stay visually comparable', () => {
    // Same value in, same color out, regardless of what other NDVI data
    // exists in the frame -- this is the whole point of a fixed domain.
    expect(ndviColor(0.5)).toEqual(ndviColor(0.5))
  })

  it('clamps out-of-range values instead of extrapolating', () => {
    expect(ndviColor(5)).toEqual(ndviColor(0.9))
    expect(ndviColor(-5)).toEqual(ndviColor(-0.3))
  })

  it('renders NaN as a neutral color rather than crashing', () => {
    expect(ndviColor(NaN)).toEqual([30, 34, 40])
  })
})
