/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Client-side GeoTIFF decoding (the browser counterpart of the backend's own `rasterio` reads) -- purely a display
 * concern: no NDVI, uncertainty statistic, or resampling is computed here. All of that comes from the backend response.
 */

import { fromArrayBuffer, type GeoTIFFImage } from 'geotiff';

/** The FRAME RGBN band order the backend always writes: B04 (red), B03 (green), B02 (blue), B08 (NIR). */
export const BAND_INDEX = { RED: 0, GREEN: 1, BLUE: 2, NIR: 3 } as const;

export interface DecodedRaster {
  width: number;
  height: number;
  /** One Float32Array per band, in file order. */
  bands: Float32Array[];
  bandCount: number;
  /** Bounding box in the raster's own CRS units, [minX, minY, maxX, maxY], if the file carries georeferencing. */
  bbox: [number, number, number, number] | null;
}

async function decodeImage(image: GeoTIFFImage): Promise<DecodedRaster> {
  const width = image.getWidth();
  const height = image.getHeight();
  const raster = await image.readRasters();
  const bandCount = Array.isArray(raster) ? raster.length : 1;
  const bands: Float32Array[] = [];
  for (let i = 0; i < bandCount; i += 1) {
    const band = Array.isArray(raster) ? raster[i] : raster;
    bands.push(band instanceof Float32Array ? band : Float32Array.from(band as ArrayLike<number>));
  }
  let bbox: [number, number, number, number] | null = null;
  try {
    const box = image.getBoundingBox();
    bbox = [box[0], box[1], box[2], box[3]];
  } catch {
    bbox = null;
  }
  return { width, height, bands, bandCount, bbox };
}

/** Decode a GeoTIFF `Blob` (as returned by the FRAME API's download endpoints) into per-band Float32Arrays. */
export async function decodeGeotiffBlob(blob: Blob): Promise<DecodedRaster> {
  const buffer = await blob.arrayBuffer();
  const tiff = await fromArrayBuffer(buffer);
  const image = await tiff.getImage();
  return decodeImage(image);
}

/** Decode a GeoTIFF `File` (a user's own upload, before it is sent to the backend) the same way. */
export async function decodeGeotiffFile(file: File): Promise<DecodedRaster> {
  const buffer = await file.arrayBuffer();
  const tiff = await fromArrayBuffer(buffer);
  const image = await tiff.getImage();
  return decodeImage(image);
}

/** The real value of one band at one pixel, or `null` if the coordinate is outside the raster. */
export function pixelValue(raster: DecodedRaster, bandIndex: number, x: number, y: number): number | null {
  if (x < 0 || y < 0 || x >= raster.width || y >= raster.height || bandIndex >= raster.bandCount) return null;
  return raster.bands[bandIndex][y * raster.width + x];
}
