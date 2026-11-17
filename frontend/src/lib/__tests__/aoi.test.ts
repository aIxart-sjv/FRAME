import { describe, expect, it } from 'vitest'
import {
  approximateAreaKm2,
  bboxFromCorners,
  bboxFromPoints,
  formatBbox,
  isValidBbox,
  polygonToGeojson,
  rectangleToGeojson,
  safeRetrievalResolutionM,
} from '../aoi'

describe('bboxFromCorners', () => {
  it('is order-independent', () => {
    expect(bboxFromCorners([80.32, 13.15], [80.12, 12.85])).toEqual([80.12, 12.85, 80.32, 13.15])
    expect(bboxFromCorners([80.12, 12.85], [80.32, 13.15])).toEqual([80.12, 12.85, 80.32, 13.15])
  })
})

describe('bboxFromPoints', () => {
  it('encloses an arbitrary point set', () => {
    expect(bboxFromPoints([[80.2, 13.0], [80.1, 13.2], [80.3, 12.9]])).toEqual([80.1, 12.9, 80.3, 13.2])
  })

  it('throws on an empty point list rather than returning a fabricated box', () => {
    expect(() => bboxFromPoints([])).toThrow()
  })
})

describe('isValidBbox', () => {
  it('accepts a real, ordered, in-range box', () => {
    expect(isValidBbox([80.12, 12.85, 80.32, 13.15])).toBe(true)
  })

  it('rejects a degenerate box from a single click with no drag', () => {
    expect(isValidBbox([80.2, 13.0, 80.2, 13.0])).toBe(false)
  })

  it('rejects an inverted box', () => {
    expect(isValidBbox([80.32, 12.85, 80.12, 13.15])).toBe(false)
  })

  it('rejects an out-of-range coordinate', () => {
    expect(isValidBbox([80.12, 12.85, 200, 13.15])).toBe(false)
  })

  it('rejects NaN from a failed parse', () => {
    expect(isValidBbox([NaN, 12.85, 80.32, 13.15])).toBe(false)
  })
})

describe('rectangleToGeojson', () => {
  it('produces a closed 5-point ring matching the bbox corners', () => {
    const geojson = rectangleToGeojson([80.12, 12.85, 80.32, 13.15])
    expect(geojson.type).toBe('Polygon')
    const ring = geojson.coordinates[0]
    expect(ring).toHaveLength(5)
    expect(ring[0]).toEqual(ring[4]) // closed
    expect(ring).toEqual([
      [80.12, 12.85],
      [80.32, 12.85],
      [80.32, 13.15],
      [80.12, 13.15],
      [80.12, 12.85],
    ])
  })
})

describe('polygonToGeojson', () => {
  it('closes an open ring', () => {
    const geojson = polygonToGeojson([[80.1, 13.0], [80.2, 13.1], [80.3, 12.9]])
    const ring = geojson.coordinates[0]
    expect(ring).toHaveLength(4)
    expect(ring[0]).toEqual(ring[3])
  })

  it('does not duplicate the closing point if the caller already closed the ring', () => {
    const geojson = polygonToGeojson([[80.1, 13.0], [80.2, 13.1], [80.3, 12.9], [80.1, 13.0]])
    expect(geojson.coordinates[0]).toHaveLength(4)
  })

  it('rejects fewer than 3 points', () => {
    expect(() => polygonToGeojson([[80.1, 13.0], [80.2, 13.1]])).toThrow()
  })
})

describe('approximateAreaKm2', () => {
  it('is positive and roughly matches a known small AOI (~20km x 33km near the equator)', () => {
    const area = approximateAreaKm2([80.12, 12.85, 80.32, 13.15])
    expect(area).toBeGreaterThan(400)
    expect(area).toBeLessThan(900)
  })
})

describe('formatBbox', () => {
  it('formats to the requested precision', () => {
    expect(formatBbox([80.123456, 12.85, 80.32, 13.15], 2)).toBe('[80.12, 12.85, 80.32, 13.15]')
  })
})

describe('safeRetrievalResolutionM', () => {
  it('stays at the minimum (10 m) for a small AOI', () => {
    // ~1.1km x 1.1km AOI (0.01 degree) -- well within 256 px at 10 m.
    expect(safeRetrievalResolutionM([80.2, 13.0, 80.21, 13.01])).toBe(10)
  })

  it('coarsens the resolution for a large, real-world-sized drawn AOI (regression: this exact AOI hung a live retrieve request against the real CDSE API)', () => {
    const resolution = safeRetrievalResolutionM([79.9013, 12.8224, 80.6264, 13.5258])
    expect(resolution).toBeGreaterThan(10)
    // At the returned resolution, the longest side must be well under 1000 px -- the actual
    // failure mode was an ~8000 px request.
    const widthM = (80.6264 - 79.9013) * 111_320 * Math.cos((13.17 * Math.PI) / 180)
    expect(widthM / resolution).toBeLessThan(1000)
  })

  it('never returns below the minimum resolution regardless of target pixels', () => {
    expect(safeRetrievalResolutionM([80.2, 13.0, 80.21, 13.01], 1)).toBeGreaterThanOrEqual(10)
  })

  it('never exceeds the backend-accepted maximum of 100 m, even for a very large AOI (regression: a 310 m result was rejected by the real API with a 422)', () => {
    const resolution = safeRetrievalResolutionM([79.9013, 12.8224, 80.6264, 13.5258])
    expect(resolution).toBeLessThanOrEqual(100)
  })
})
