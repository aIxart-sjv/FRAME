/**
 * GeoTIFF decoding for in-browser display. This reads pixel arrays out of
 * a GeoTIFF file (via the `geotiff` npm package -- a standard TIFF/BigTIFF
 * parser, the client-side equivalent of the `rasterio` reads the backend
 * already does) so the already-computed rasters the API returns can be
 * drawn on a <canvas>.
 *
 * This module does NOT compute anything scientific: no NDVI, no
 * uncertainty statistic, no resampling for analysis purposes. It only
 * extracts the raw per-band float arrays the backend already produced --
 * every number displayed comes from frame.* via the API, never recomputed
 * here (see src/lib/colormap.ts and src/lib/raster.ts for the *rendering*
 * transforms, which are display-only, the same category as the backend's
 * own `normalize_for_visualization`).
 */

import { fromArrayBuffer } from 'geotiff'

export interface DecodedRaster {
  width: number
  height: number
  /** One Float32Array per band, row-major, length width*height. */
  bands: Float32Array[]
}

export async function decodeGeoTiff(source: Blob): Promise<DecodedRaster> {
  const buffer = await source.arrayBuffer()
  const tiff = await fromArrayBuffer(buffer)
  const image = await tiff.getImage()
  const width = image.getWidth()
  const height = image.getHeight()
  const rasters = await image.readRasters({ interleave: false })
  const bandArrays = (Array.isArray(rasters) ? rasters : [rasters]) as ArrayLike<number>[]
  const bands = bandArrays.map((band) => Float32Array.from(band))
  return { width, height, bands }
}

/** Band index lookup for the RGBN order this prototype always uses (see
 * frame.preprocessing.RGBN_BANDS): B04 (Red), B03 (Green), B02 (Blue), B08 (NIR). */
export const BAND_INDEX = { RED: 0, GREEN: 1, BLUE: 2, NIR: 3 } as const
