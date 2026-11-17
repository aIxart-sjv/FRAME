/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useEffect, useRef } from 'react';
// maplibre-gl v6 is ESM-only with no default export (unlike v4/v5) -- a namespace import
// is the version-agnostic way to reach its named exports (Map, NavigationControl, ...).
import * as maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import type { AoiPolygonGeojson, Bbox, LngLat } from '../../lib/aoi';
import { bboxFromCorners, polygonToGeojson, rectangleToGeojson } from '../../lib/aoi';
import type { SceneCandidate } from '../../api/types';

/**
 * MapLibre's own public demo style (https://maplibre.org/maplibre-gl-js/docs/examples/): a real, free,
 * key-free vector basemap MapLibre itself publishes for exactly this kind of use (getting started, demos,
 * testing). Not a decorative image -- this is a genuine interactive map (pan/zoom/tiles), just a minimal
 * basemap style rather than a fully-styled commercial one.
 */
const BASEMAP_STYLE_URL = 'https://demotiles.maplibre.org/style.json';

const AOI_SOURCE_ID = 'frame-aoi';
const AOI_DRAFT_SOURCE_ID = 'frame-aoi-draft';
const FOOTPRINTS_SOURCE_ID = 'frame-scene-footprints';

const emptyFeatureCollection = (): GeoJSON.FeatureCollection => ({ type: 'FeatureCollection', features: [] });

function polygonFeature(geojson: AoiPolygonGeojson): GeoJSON.Feature {
  return { type: 'Feature', properties: {}, geometry: geojson as unknown as GeoJSON.Geometry };
}

export type DrawMode = 'rectangle' | 'polygon' | null;

interface AcquisitionMapProps {
  className?: string;
  /** The confirmed AOI to display (drawn or search-derived), or null for none. */
  aoiGeojson: AoiPolygonGeojson | null;
  /** Real scene footprints returned by a search, with the currently-selected one highlighted. */
  scenes?: SceneCandidate[];
  selectedSceneId?: string | null;
  onSceneClick?: (sceneId: string) => void;
  /** Which AOI tool is active. Rectangle: drag a box. Polygon: click to add vertices, double-click to finish. */
  drawMode: DrawMode;
  onRectangleDrawn?: (bbox: Bbox) => void;
  onPolygonDrawn?: (points: LngLat[]) => void;
  initialCenter?: LngLat;
  initialZoom?: number;
}

/**
 * A real, interactive MapLibre GL map: pan/zoom, rectangle/polygon AOI drawing, and real scene-footprint
 * display. Shared by SelectArea (draw the AOI) and SelectScene (show real search results) so both pages use
 * exactly one map implementation. Never renders a decorative/mock map -- if MapLibre fails to initialise
 * (e.g. no WebGL), that failure is surfaced, not silently swapped for a fake graphic.
 */
export const AcquisitionMap: React.FC<AcquisitionMapProps> = ({
  className,
  aoiGeojson,
  scenes = [],
  selectedSceneId = null,
  onSceneClick,
  drawMode,
  onRectangleDrawn,
  onPolygonDrawn,
  initialCenter = [80.2144, 13.0125],
  initialZoom = 8,
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const [ready, setReady] = React.useState(false);
  const [initError, setInitError] = React.useState<string | null>(null);

  const drawModeRef = useRef(drawMode);
  drawModeRef.current = drawMode;
  const onRectangleDrawnRef = useRef(onRectangleDrawn);
  onRectangleDrawnRef.current = onRectangleDrawn;
  const onPolygonDrawnRef = useRef(onPolygonDrawn);
  onPolygonDrawnRef.current = onPolygonDrawn;
  const onSceneClickRef = useRef(onSceneClick);
  onSceneClickRef.current = onSceneClick;

  // Map init/teardown -- once per mount.
  useEffect(() => {
    if (!containerRef.current) return;
    let map: maplibregl.Map;
    try {
      map = new maplibregl.Map({
        container: containerRef.current,
        style: BASEMAP_STYLE_URL,
        center: initialCenter,
        zoom: initialZoom,
      });
    } catch (error) {
      setInitError(error instanceof Error ? error.message : 'The map could not be created (no WebGL support?).');
      return;
    }
    mapRef.current = map;
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right');

    map.on('error', (event) => {
      // A tile/style network error should not look like a fabricated blank map -- surface it.
      setInitError((event.error && event.error.message) || 'The map failed to load a resource.');
    });

    map.on('load', () => {
      map.addSource(AOI_SOURCE_ID, { type: 'geojson', data: emptyFeatureCollection() });
      map.addLayer({ id: `${AOI_SOURCE_ID}-fill`, type: 'fill', source: AOI_SOURCE_ID, paint: { 'fill-color': '#D8B84C', 'fill-opacity': 0.12 } });
      map.addLayer({ id: `${AOI_SOURCE_ID}-line`, type: 'line', source: AOI_SOURCE_ID, paint: { 'line-color': '#D8B84C', 'line-width': 2 } });

      map.addSource(AOI_DRAFT_SOURCE_ID, { type: 'geojson', data: emptyFeatureCollection() });
      map.addLayer({
        id: `${AOI_DRAFT_SOURCE_ID}-fill`,
        type: 'fill',
        source: AOI_DRAFT_SOURCE_ID,
        paint: { 'fill-color': '#D8B84C', 'fill-opacity': 0.08 },
      });
      map.addLayer({
        id: `${AOI_DRAFT_SOURCE_ID}-line`,
        type: 'line',
        source: AOI_DRAFT_SOURCE_ID,
        paint: { 'line-color': '#D8B84C', 'line-width': 1.5, 'line-dasharray': [2, 2] },
      });

      map.addSource(FOOTPRINTS_SOURCE_ID, { type: 'geojson', data: emptyFeatureCollection() });
      map.addLayer({
        id: `${FOOTPRINTS_SOURCE_ID}-fill`,
        type: 'fill',
        source: FOOTPRINTS_SOURCE_ID,
        paint: {
          'fill-color': ['case', ['get', 'selected'], '#35C98A', '#2563eb'],
          'fill-opacity': ['case', ['get', 'selected'], 0.25, 0.08],
        },
      });
      map.addLayer({
        id: `${FOOTPRINTS_SOURCE_ID}-line`,
        type: 'line',
        source: FOOTPRINTS_SOURCE_ID,
        paint: {
          'line-color': ['case', ['get', 'selected'], '#35C98A', '#2563eb'],
          'line-width': ['case', ['get', 'selected'], 2.5, 1.2],
        },
      });

      map.on('click', `${FOOTPRINTS_SOURCE_ID}-fill`, (e) => {
        const sceneId = e.features?.[0]?.properties?.id;
        if (sceneId && onSceneClickRef.current) onSceneClickRef.current(sceneId as string);
      });
      map.on('mouseenter', `${FOOTPRINTS_SOURCE_ID}-fill`, () => {
        map.getCanvas().style.cursor = 'pointer';
      });
      map.on('mouseleave', `${FOOTPRINTS_SOURCE_ID}-fill`, () => {
        map.getCanvas().style.cursor = '';
      });

      setReady(true);
    });

    // -- Rectangle / polygon drawing --------------------------------------------------------
    let dragStart: LngLat | null = null;
    let polygonPoints: LngLat[] = [];

    const updateDraft = (geojson: AoiPolygonGeojson | null) => {
      const source = map.getSource(AOI_DRAFT_SOURCE_ID) as maplibregl.GeoJSONSource | undefined;
      source?.setData(geojson ? { type: 'FeatureCollection', features: [polygonFeature(geojson)] } : emptyFeatureCollection());
    };

    const handleMouseDown = (e: maplibregl.MapMouseEvent) => {
      if (drawModeRef.current !== 'rectangle') return;
      dragStart = [e.lngLat.lng, e.lngLat.lat];
      map.dragPan.disable();
    };
    const handleMouseMove = (e: maplibregl.MapMouseEvent) => {
      if (drawModeRef.current !== 'rectangle' || !dragStart) return;
      const current: LngLat = [e.lngLat.lng, e.lngLat.lat];
      updateDraft(rectangleToGeojson(bboxFromCorners(dragStart, current)));
    };
    const handleMouseUp = (e: maplibregl.MapMouseEvent) => {
      if (drawModeRef.current !== 'rectangle' || !dragStart) return;
      const current: LngLat = [e.lngLat.lng, e.lngLat.lat];
      const bbox = bboxFromCorners(dragStart, current);
      dragStart = null;
      map.dragPan.enable();
      updateDraft(null);
      onRectangleDrawnRef.current?.(bbox);
    };
    const handleClick = (e: maplibregl.MapMouseEvent) => {
      if (drawModeRef.current !== 'polygon') return;
      polygonPoints = [...polygonPoints, [e.lngLat.lng, e.lngLat.lat]];
      if (polygonPoints.length >= 2) {
        try {
          updateDraft(polygonToGeojson(polygonPoints));
        } catch {
          // fewer than 3 points -- nothing to preview yet
        }
      }
    };
    const handleDoubleClick = (e: maplibregl.MapMouseEvent) => {
      if (drawModeRef.current !== 'polygon' || polygonPoints.length < 2) return;
      e.preventDefault();
      const finished = polygonPoints;
      polygonPoints = [];
      updateDraft(null);
      onPolygonDrawnRef.current?.(finished);
    };

    map.on('mousedown', handleMouseDown);
    map.on('mousemove', handleMouseMove);
    map.on('mouseup', handleMouseUp);
    map.on('click', handleClick);
    map.on('dblclick', handleDoubleClick);

    return () => {
      map.remove();
      mapRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Disable double-click zoom while polygon-drawing (it would otherwise zoom the map on finish).
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    if (drawMode === 'polygon') map.doubleClickZoom.disable();
    else map.doubleClickZoom.enable();
  }, [drawMode, ready]);

  // Keep the confirmed AOI layer in sync.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    const source = map.getSource(AOI_SOURCE_ID) as maplibregl.GeoJSONSource | undefined;
    source?.setData(aoiGeojson ? { type: 'FeatureCollection', features: [polygonFeature(aoiGeojson)] } : emptyFeatureCollection());
  }, [aoiGeojson, ready]);

  // Keep real scene footprints in sync, and fit the view to them once they arrive.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    const source = map.getSource(FOOTPRINTS_SOURCE_ID) as maplibregl.GeoJSONSource | undefined;
    const features: GeoJSON.Feature[] = scenes
      .filter((scene) => scene.geometry)
      .map((scene) => ({
        type: 'Feature',
        properties: { id: scene.id, selected: scene.id === selectedSceneId },
        geometry: scene.geometry as unknown as GeoJSON.Geometry,
      }));
    source?.setData({ type: 'FeatureCollection', features });

    if (features.length > 0) {
      const bounds = new maplibregl.LngLatBounds();
      for (const feature of features) {
        const geom = feature.geometry as GeoJSON.Polygon;
        for (const ring of geom.coordinates) {
          for (const [lon, lat] of ring) bounds.extend([lon, lat]);
        }
      }
      if (!bounds.isEmpty()) map.fitBounds(bounds, { padding: 32, maxZoom: 13, duration: 500 });
    }
  }, [scenes, selectedSceneId, ready]);

  return (
    <div className={className} style={{ position: 'relative', width: '100%', height: '100%' }}>
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
      {initError && (
        <div className="absolute inset-x-0 top-0 m-2 rounded bg-[#D94B4B]/90 px-3 py-2 text-xs text-white">
          Map error: {initError}
        </div>
      )}
    </div>
  );
};
