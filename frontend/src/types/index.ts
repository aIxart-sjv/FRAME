/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * FRAME frontend — application-level types.
 *
 * These describe UI state only. The data contract with the backend lives in `src/api/types.ts` and matches
 * `frame/api/schemas.py` field for field; nothing here invents a shape the API does not actually return.
 */

export type ViewerMode = 'original' | 'super-resolved' | 'ndvi' | 'evidence';

export type ValidationSection = 'spectral' | 'spatial' | 'detail' | 'stability' | 'reference' | 'generalisation';

/** A rectangle the analyst dragged on the viewer, in raster pixel coordinates of the layer it was drawn on. */
export interface PixelRegion {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  /** metres per pixel of the layer this region was measured on (2.5 for the SR grid, the input's own resolution for the original). */
  pixelSizeMeters: number;
}

/** Real, computed-on-the-spot statistics of a region — every field is derived from the decoded raster in the browser, at display time; nothing here is looked up from a table. */
export interface RegionStats {
  region: PixelRegion;
  widthMeters: number;
  heightMeters: number;
  areaHectares: number;
  pixelCount: number;
  /** mean/min/max of the band(s) actually on screen when the region was drawn (RGBN reflectance, the stability value, or NDVI). */
  bandStats: Array<{ label: string; mean: number; min: number; max: number }>;
}

export type WorkspaceTab = 'inspect' | 'analyze' | 'validate';
