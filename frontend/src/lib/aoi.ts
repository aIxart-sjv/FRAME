/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Pure AOI (area-of-interest) geometry helpers -- no map, no React, no network. Kept separate from the MapLibre
 * component specifically so this logic can be unit-tested directly (jsdom has no WebGL context, so a mounted
 * MapLibre map cannot be exercised in a unit test -- see src/test/setup.ts).
 */

export type LngLat = [number, number]; // [lon, lat], EPSG:4326
export type Bbox = [number, number, number, number]; // [min_lon, min_lat, max_lon, max_lat], EPSG:4326

export interface AoiPolygonGeojson {
  type: 'Polygon';
  coordinates: number[][][];
}

/** The bbox spanning two opposite corners of a dragged rectangle -- order-independent. */
export function bboxFromCorners(a: LngLat, b: LngLat): Bbox {
  return [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])];
}

/** The bbox enclosing an arbitrary set of points (e.g. a drawn polygon's vertices). */
export function bboxFromPoints(points: LngLat[]): Bbox {
  if (points.length === 0) throw new Error('At least one point is required to compute a bounding box.');
  const lons = points.map((p) => p[0]);
  const lats = points.map((p) => p[1]);
  return [Math.min(...lons), Math.min(...lats), Math.max(...lons), Math.max(...lats)];
}

/** Whether `bbox` is a real, orderable, in-range box the backend's AcquisitionQuery would accept
 * (frame/acquisition/contract.py's own validation) -- checked client-side so a degenerate AOI
 * (a single click with no drag, a self-crossing polygon collapsed to a line) never reaches the API. */
export function isValidBbox(bbox: Bbox): boolean {
  const [minLon, minLat, maxLon, maxLat] = bbox;
  return (
    [minLon, minLat, maxLon, maxLat].every(Number.isFinite) &&
    minLon < maxLon &&
    minLat < maxLat &&
    minLon >= -180 &&
    maxLon <= 180 &&
    minLat >= -90 &&
    maxLat <= 90
  );
}

/** A rectangle AOI as a closed GeoJSON Polygon ring. */
export function rectangleToGeojson(bbox: Bbox): AoiPolygonGeojson {
  const [minLon, minLat, maxLon, maxLat] = bbox;
  return {
    type: 'Polygon',
    coordinates: [
      [
        [minLon, minLat],
        [maxLon, minLat],
        [maxLon, maxLat],
        [minLon, maxLat],
        [minLon, minLat],
      ],
    ],
  };
}

/** A hand-drawn polygon AOI (>= 3 vertices, open ring) as a closed GeoJSON Polygon. */
export function polygonToGeojson(points: LngLat[]): AoiPolygonGeojson {
  if (points.length < 3) throw new Error('A polygon AOI needs at least 3 points.');
  const first = points[0];
  const last = points[points.length - 1];
  const closed = first[0] === last[0] && first[1] === last[1] ? points : [...points, first];
  return { type: 'Polygon', coordinates: [closed.map((p) => [p[0], p[1]])] };
}

/** Approximate AOI area in km^2 for on-screen display only (equirectangular approximation, not
 * geodesic) -- never sent to the backend, which does its own real UTM-projected area math
 * (frame.acquisition.geometry) when it matters. */
export function approximateAreaKm2(bbox: Bbox): number {
  const [minLon, minLat, maxLon, maxLat] = bbox;
  const midLatRad = ((minLat + maxLat) / 2) * (Math.PI / 180);
  const kmPerDegLat = 110.574;
  const kmPerDegLon = 111.32 * Math.cos(midLatRad);
  const widthKm = (maxLon - minLon) * kmPerDegLon;
  const heightKm = (maxLat - minLat) * kmPerDegLat;
  return Math.abs(widthKm * heightKm);
}

export function formatBbox(bbox: Bbox, digits = 4): string {
  return `[${bbox.map((v) => v.toFixed(digits)).join(', ')}]`;
}

/** The backend's own hard cap (frame/api/schemas.py AcquisitionRetrieveRequest.resolution_m, `le=100`) --
 * duplicated here, not imported, since the frontend has no build-time access to the backend schema; a
 * value above this is rejected with a 422 regardless of what this function computes. */
const BACKEND_MAX_RESOLUTION_M = 100;

/** A resolution (metres/pixel) that keeps a real pixel-retrieval request's output roughly bounded
 * to `targetPixels` per side, regardless of how large an AOI the analyst drew. Retrieving a whole
 * drawn AOI at a fixed native resolution (e.g. always 10 m) is unsafe against a real third-party
 * API: a large AOI at 10 m can mean tens of millions of pixels in one real, costed request (this
 * exact failure mode hung a real CDSE Process API request during live testing of this feature).
 * Uses the same equirectangular approximation as `approximateAreaKm2` -- adequate for choosing a
 * safe resolution, not for the request's actual science (the backend does its own real UTM
 * projection). Clamped to the backend's own accepted range: for an AOI so large that even the
 * maximum allowed resolution would exceed `targetPixels`, the request is still valid (never
 * rejected), just larger than the pixel target -- there is no larger resolution left to ask for. */
export function safeRetrievalResolutionM(bbox: Bbox, targetPixels = 256, minResolutionM = 10): number {
  const [minLon, minLat, maxLon, maxLat] = bbox;
  const midLatRad = ((minLat + maxLat) / 2) * (Math.PI / 180);
  const widthM = (maxLon - minLon) * 111_320 * Math.cos(midLatRad);
  const heightM = (maxLat - minLat) * 110_574;
  const longestSideM = Math.max(Math.abs(widthM), Math.abs(heightM));
  const target = Math.max(minResolutionM, Math.ceil(longestSideM / targetPixels / 10) * 10);
  return Math.min(target, BACKEND_MAX_RESOLUTION_M);
}
