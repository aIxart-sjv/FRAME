/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { AlertTriangle, ArrowRight, Loader2, MousePointerClick, Square, Trash2, Upload } from 'lucide-react';
import { AcquisitionMap, type DrawMode } from './AcquisitionMap';
import type { AoiPolygonGeojson, Bbox, LngLat } from '../../lib/aoi';
import { approximateAreaKm2, bboxFromPoints, formatBbox, isValidBbox, polygonToGeojson, rectangleToGeojson } from '../../lib/aoi';
import type { AsyncStatus } from '../../state/useFrameSession';

interface SelectAreaProps {
  aoiBbox: Bbox | null;
  aoiGeojson: AoiPolygonGeojson | null;
  onSetAoi: (bbox: Bbox | null, geojson: AoiPolygonGeojson | null) => void;
  startDate: string;
  onChangeStartDate: (value: string) => void;
  endDate: string;
  onChangeEndDate: (value: string) => void;
  maxCloudCover: number;
  onChangeMaxCloudCover: (value: number) => void;
  searchStatus: AsyncStatus;
  searchError: string | null;
  onSearch: () => void;
  onContinue: () => void;
  onCancel: () => void;
  onUseManualUpload: () => void;
}

/**
 * Step 1 of the real map-first acquisition flow: draw a real AOI on a real MapLibre map, choose a date range
 * and cloud-cover threshold, then run a real `POST /acquisition/search`. Manual GeoTIFF upload remains available
 * as an explicit fallback (see UploadScene.tsx) for when no real scene covers what the analyst needs, or the
 * backend's acquisition providers are unavailable.
 */
export const SelectArea: React.FC<SelectAreaProps> = ({
  aoiBbox,
  aoiGeojson,
  onSetAoi,
  startDate,
  onChangeStartDate,
  endDate,
  onChangeEndDate,
  maxCloudCover,
  onChangeMaxCloudCover,
  searchStatus,
  searchError,
  onSearch,
  onContinue,
  onCancel,
  onUseManualUpload,
}) => {
  const [drawMode, setDrawMode] = useState<DrawMode>('rectangle');
  const [polygonPoints, setPolygonPoints] = useState<LngLat[]>([]);

  const aoiValid = aoiBbox !== null && isValidBbox(aoiBbox);
  const dateRangeValid = Boolean(startDate) && Boolean(endDate) && startDate < endDate;
  const canSearch = aoiValid && dateRangeValid && searchStatus !== 'pending';

  const handleRectangleDrawn = (bbox: Bbox) => {
    setPolygonPoints([]);
    if (isValidBbox(bbox)) onSetAoi(bbox, null);
  };

  const handlePolygonDrawn = (points: LngLat[]) => {
    setPolygonPoints(points);
    try {
      const geojson = polygonToGeojson(points);
      const bbox = bboxFromPoints(points);
      if (isValidBbox(bbox)) onSetAoi(bbox, geojson);
    } catch {
      // fewer than 3 points -- ignore, nothing to select yet
    }
  };

  const handleClear = () => {
    setPolygonPoints([]);
    onSetAoi(null, null);
  };

  const handleContinue = () => {
    if (!canSearch) return;
    onSearch();
    onContinue();
  };

  return (
    <div className="w-full min-h-[calc(100vh-3.5rem)] bg-[#0B0E12] text-white flex flex-col">
      <div className="border-b border-[#1D242C] px-6 py-6 bg-[#151A20]">
        <div className="max-w-6xl mx-auto flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <p className="text-xs font-mono uppercase tracking-wider text-[#D8B84C] mb-1">Step 1 of 2</p>
            <h1 className="text-2xl font-bold tracking-tight text-white">Where should we look?</h1>
            <p className="text-xs text-[#9AA3AD] mt-0.5 max-w-xl">
              Draw an area of interest on the map, choose a date range and cloud-cover ceiling, then search the
              real Copernicus Data Space catalogue (frame.acquisition — POST /acquisition/search).
            </p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={onCancel}
              className="px-4 py-2 text-xs font-medium text-[#9AA3AD] hover:text-white bg-[#1D242C] hover:bg-[#343C45] rounded transition-colors cursor-pointer"
            >
              Cancel
            </button>
            <button
              disabled={!canSearch}
              onClick={handleContinue}
              className={`flex items-center gap-2 px-5 py-2 text-xs font-semibold rounded transition-all cursor-pointer ${
                canSearch ? 'bg-[#D8B84C] hover:bg-[#E4C96A] text-[#151A20] shadow-md' : 'bg-[#1D242C] text-[#9AA3AD] cursor-not-allowed'
              }`}
            >
              {searchStatus === 'pending' ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <span>Search &amp; continue</span>}
              {searchStatus !== 'pending' && <ArrowRight className="w-3.5 h-3.5" />}
            </button>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto w-full flex-1 p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left column: draw tools + search parameters */}
        <div className="lg:col-span-5 flex flex-col gap-5">
          <div className="p-4 bg-[#151A20] border border-[#1D242C] rounded-lg space-y-3">
            <span className="text-xs font-medium text-[#9AA3AD] block">Draw tool</span>
            <div className="grid grid-cols-3 gap-1 p-1 bg-[#0B0E12] border border-[#1D242C] rounded-lg">
              <button
                onClick={() => setDrawMode('rectangle')}
                className={`flex flex-col items-center justify-center py-2 px-1 text-[11px] font-medium rounded transition-colors cursor-pointer ${
                  drawMode === 'rectangle' ? 'bg-[#1D242C] text-[#D8B84C]' : 'text-[#9AA3AD] hover:text-white'
                }`}
              >
                <Square className="w-3.5 h-3.5 mb-1" />
                <span>Rectangle</span>
              </button>
              <button
                onClick={() => setDrawMode('polygon')}
                className={`flex flex-col items-center justify-center py-2 px-1 text-[11px] font-medium rounded transition-colors cursor-pointer ${
                  drawMode === 'polygon' ? 'bg-[#1D242C] text-[#D8B84C]' : 'text-[#9AA3AD] hover:text-white'
                }`}
              >
                <MousePointerClick className="w-3.5 h-3.5 mb-1" />
                <span>Polygon</span>
              </button>
              <button
                onClick={handleClear}
                className="flex flex-col items-center justify-center py-2 px-1 text-[11px] font-medium rounded transition-colors cursor-pointer text-[#9AA3AD] hover:text-[#D94B4B]"
              >
                <Trash2 className="w-3.5 h-3.5 mb-1" />
                <span>Clear</span>
              </button>
            </div>
            <p className="text-[11px] text-[#9AA3AD] leading-relaxed">
              {drawMode === 'rectangle'
                ? 'Click and drag on the map to draw a rectangular AOI.'
                : 'Click to add vertices; double-click to finish the polygon.'}
            </p>
            {polygonPoints.length > 0 && polygonPoints.length < 3 && (
              <p className="text-[11px] text-[#D8B84C] font-mono">{polygonPoints.length} point(s) placed — at least 3 needed.</p>
            )}
            <div className="p-2.5 bg-[#0B0E12] rounded border border-[#1D242C] font-mono text-[11px] text-[#D8B84C]">
              {aoiValid ? `AOI: ${formatBbox(aoiBbox!)} · ~${approximateAreaKm2(aoiBbox!).toFixed(1)} km²` : 'No area of interest selected yet.'}
            </div>
            <p className="text-[11px] text-[#9AA3AD] leading-relaxed">
              A polygon AOI is currently reduced to its bounding box for search — the backend does not yet filter
              by the polygon's true shape (frame/acquisition, docs/ACQUISITION.md §2).
            </p>
          </div>

          <div className="p-4 bg-[#151A20] border border-[#1D242C] rounded-lg space-y-3">
            <span className="text-xs font-medium text-[#9AA3AD] block">Date range</span>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-[11px] text-[#9AA3AD] block mb-1">Start</label>
                <input
                  type="date"
                  value={startDate}
                  onChange={(e) => onChangeStartDate(e.target.value)}
                  className="w-full bg-[#0B0E12] border border-[#343C45] rounded px-2.5 py-1.5 text-xs font-mono text-white focus:outline-none focus:border-[#D8B84C]"
                />
              </div>
              <div>
                <label className="text-[11px] text-[#9AA3AD] block mb-1">End</label>
                <input
                  type="date"
                  value={endDate}
                  onChange={(e) => onChangeEndDate(e.target.value)}
                  className="w-full bg-[#0B0E12] border border-[#343C45] rounded px-2.5 py-1.5 text-xs font-mono text-white focus:outline-none focus:border-[#D8B84C]"
                />
              </div>
            </div>
            {startDate && endDate && !dateRangeValid && (
              <p className="text-[11px] text-[#D94B4B]">Start date must be before end date.</p>
            )}

            <div>
              <div className="flex justify-between text-[11px] mb-1">
                <span className="text-[#9AA3AD]">Max cloud cover</span>
                <span className="font-mono text-white">{maxCloudCover}%</span>
              </div>
              <input
                type="range"
                min={1}
                max={100}
                value={maxCloudCover}
                onChange={(e) => onChangeMaxCloudCover(Number(e.target.value))}
                className="w-full accent-[#D8B84C] bg-[#1D242C] h-1.5 rounded cursor-pointer"
              />
            </div>
          </div>

          {searchStatus === 'error' && searchError && (
            <div className="p-3 bg-[#151A20] border border-[#D94B4B]/40 rounded-lg flex items-start gap-2.5 text-xs">
              <AlertTriangle className="w-3.5 h-3.5 text-[#D94B4B] shrink-0 mt-0.5" />
              <span className="text-[#9AA3AD] leading-relaxed">{searchError}</span>
            </div>
          )}

          <button
            onClick={onUseManualUpload}
            className="flex items-center justify-center gap-2 px-4 py-2.5 text-xs font-medium text-[#9AA3AD] hover:text-white bg-[#151A20] hover:bg-[#1D242C] border border-[#1D242C] rounded-lg transition-colors cursor-pointer"
          >
            <Upload className="w-3.5 h-3.5" />
            <span>Or upload a GeoTIFF directly (fallback)</span>
          </button>
        </div>

        {/* Right column: real interactive map */}
        <div className="lg:col-span-7 flex flex-col bg-[#151A20] border border-[#1D242C] rounded-lg overflow-hidden">
          <div className="px-4 py-2.5 bg-[#151A20] border-b border-[#1D242C] flex items-center justify-between text-xs">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-[#35C98A]" />
              <span className="text-[#9AA3AD] font-medium">Interactive map (MapLibre GL)</span>
            </div>
            {aoiValid && <span className="font-mono text-[11px] text-[#9AA3AD]">{formatBbox(aoiBbox!, 2)}</span>}
          </div>
          <div className="relative flex-1 min-h-[420px]">
            <AcquisitionMap
              aoiGeojson={aoiGeojson ?? (aoiValid ? rectangleToGeojson(aoiBbox!) : null)}
              drawMode={drawMode}
              onRectangleDrawn={handleRectangleDrawn}
              onPolygonDrawn={handlePolygonDrawn}
            />
          </div>
        </div>
      </div>
    </div>
  );
};
