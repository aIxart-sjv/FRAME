/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Crosshair, Eye, Info, Loader2, Ruler, Split, Square, X } from 'lucide-react';
import * as api from '../../api/client';
import { useDecodedFile, useRemoteRaster } from '../../hooks/useRaster';
import { BAND_INDEX, type DecodedRaster } from '../../lib/geotiff';
import { renderNdviBand, renderRgbComposite, renderSequentialBand, samplePixel } from '../../lib/raster';
import type { NDVIAnalysisResponse, SRResultResponse, UploadResponse } from '../../api/types';
import type { RegionStats, ViewerMode } from '../../types';

interface InspectViewerProps {
  file: File | null;
  /** The upload response for `file`, if still known this session -- gives the file's OWN band order, which the API
   * accepts in any order but does not rewrite (only the backend's own SR/NDVI/stability outputs are canonical
   * B04/B03/B02/B08; this file may not be). `null` when opened from a past session entry with no `file` either. */
  uploadResult: UploadResponse | null;
  jobResult: SRResultResponse;
  ndviAnalysis: NDVIAnalysisResponse | null;
  onNavigateToAnalyze: () => void;
  onNavigateToValidate: () => void;
}

const OVERALL_STD_BAND_INDEX = 4; // frame/api/services/pipeline.py: [B04_std, B03_std, B02_std, B08_std, overall_std]
const CANONICAL_ORDER = ['B04', 'B03', 'B02', 'B08']; // what every backend-written raster (SR, stability, NDVI) uses

type Tool = 'pan' | 'roi' | 'pixel-probe' | 'measure';

/** Real band names in the order this raster's own bands are actually stored -- the file's own order for the
 * original upload (order-independent by contract), the fixed canonical order for anything the backend wrote. */
function bandOrderFor(mode: ViewerMode, uploadResult: UploadResponse | null): string[] {
  if (mode === 'original' && uploadResult) return [...uploadResult.band_names];
  return CANONICAL_ORDER;
}

function rgbIndices(order: string[]): { red: number; green: number; blue: number } {
  const at = (name: string, fallback: number) => {
    const i = order.indexOf(name);
    return i >= 0 ? i : fallback;
  };
  return { red: at('B04', BAND_INDEX.RED), green: at('B03', BAND_INDEX.GREEN), blue: at('B02', BAND_INDEX.BLUE) };
}

/** NDVI = (NIR - Red) / (NIR + Red) at whichever indices `order` says B08 and B04 actually are. */
function ndviFromOrder(values: number[], order: string[]): number | null {
  const redIdx = order.indexOf('B04');
  const nirIdx = order.indexOf('B08');
  if (redIdx < 0 || nirIdx < 0) return null;
  const red = values[redIdx];
  const nir = values[nirIdx];
  if (!Number.isFinite(red) || !Number.isFinite(nir)) return null;
  const denom = nir + red;
  return Math.abs(denom) < 1e-6 ? null : (nir - red) / denom;
}

function pixelSizeFor(mode: ViewerMode, jobResult: SRResultResponse, uploadResolutionM: number | null): number {
  if (mode === 'original') return uploadResolutionM ?? jobResult.resolution.native_resolution_m;
  return jobResult.resolution.sr_resolution_m;
}

/**
 * Renders the REAL rasters the backend produced (and the user's own uploaded file), fetched and decoded in the
 * browser — no illustrative placeholder imagery. Four modes: the original upload, the SR output, the NDVI
 * demonstration (only once it has actually been run), and the TTA stability diagnostic. The pixel probe, region
 * tool, and ruler all read real decoded pixel values; nothing here is a fixed sample or a canned number.
 */
export const InspectViewer: React.FC<InspectViewerProps> = ({ file, uploadResult, jobResult, ndviAnalysis, onNavigateToAnalyze, onNavigateToValidate }) => {
  const [viewerMode, setViewerMode] = useState<ViewerMode>('super-resolved');
  const [ndviLayer, setNdviLayer] = useState<'native' | 'sr' | 'diff'>('sr');
  const [isSplitCompare, setIsSplitCompare] = useState<boolean>(false);
  const [sliderPosition, setSliderPosition] = useState<number>(50);
  const [activeTool, setActiveTool] = useState<Tool>('pan');
  const [showMetadataDrawer, setShowMetadataDrawer] = useState<boolean>(false);
  const [probe, setProbe] = useState<{ x: number; y: number; values: number[]; ndvi: number | null } | null>(null);
  const [measurePoints, setMeasurePoints] = useState<Array<{ x: number; y: number }>>([]);
  const [measureMeters, setMeasureMeters] = useState<number | null>(null);
  const [roiDragStart, setRoiDragStart] = useState<{ x: number; y: number } | null>(null);
  const [roiStats, setRoiStats] = useState<RegionStats | null>(null);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const isDraggingSlider = useRef(false);

  const original = useDecodedFile(file);
  const sr = useRemoteRaster(() => api.fetchSrGeotiff(jobResult.job_id), jobResult.job_id);
  const evidence = useRemoteRaster(() => api.fetchUncertaintyGeotiff(jobResult.job_id), jobResult.job_id);
  const ndviLayerKey = ndviAnalysis ? `${ndviAnalysis.analysis_id}:${ndviLayer}` : null;
  const ndviLayerPath = ndviLayer === 'native' ? 'native-ndvi' : ndviLayer === 'sr' ? 'sr-ndvi' : 'ndvi-diff';
  const ndviRaster = useRemoteRaster(() => api.fetchNdviGeotiff(ndviAnalysis!.analysis_id, ndviLayerPath), ndviLayerKey);

  const activeRaster: DecodedRaster | null = viewerMode === 'original' ? original.raster : viewerMode === 'evidence' ? evidence.raster : viewerMode === 'ndvi' ? ndviRaster.raster : sr.raster;
  const isLoading = viewerMode === 'original' ? original.status === 'pending' : viewerMode === 'evidence' ? evidence.status === 'pending' : viewerMode === 'ndvi' ? ndviRaster.status === 'pending' : sr.status === 'pending';
  const loadError = viewerMode === 'original' ? original.error : viewerMode === 'evidence' ? evidence.error : viewerMode === 'ndvi' ? ndviRaster.error : sr.error;

  const pixelSize = pixelSizeFor(viewerMode, jobResult, null);
  const [legendRange, setLegendRange] = useState<[number, number] | null>(null);

  const showSplit = isSplitCompare && viewerMode === 'super-resolved';
  const activeBandOrder = bandOrderFor(viewerMode, uploadResult);
  const originalBandOrder = bandOrderFor('original', uploadResult);

  // Render whichever raster is active into the one on-screen canvas. Split-compare draws original (nearest-
  // upsampled) and SR into the SAME canvas with a clip, rather than overlaying a second canvas: an absolutely
  // positioned sibling with no in-flow content collapses its `aspect-ratio` parent to 0x0 in every tested browser,
  // since the parent then has no normal-flow child to size itself from.
  useEffect(() => {
    if (!canvasRef.current || !activeRaster) return;
    if (viewerMode === 'evidence') {
      const [lo, hi] = renderSequentialBand(canvasRef.current, activeRaster.bands[OVERALL_STD_BAND_INDEX], activeRaster.width, activeRaster.height);
      setLegendRange([lo, hi]);
    } else if (viewerMode === 'ndvi') {
      renderNdviBand(canvasRef.current, activeRaster.bands[0], activeRaster.width, activeRaster.height);
      setLegendRange(null);
    } else if (showSplit && original.raster) {
      const canvas = canvasRef.current;
      const { width, height } = activeRaster;
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      const originalCanvas = document.createElement('canvas');
      renderRgbComposite(originalCanvas, original.raster, rgbIndices(originalBandOrder));
      const srCanvas = document.createElement('canvas');
      renderRgbComposite(srCanvas, activeRaster);
      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, width, height);
      ctx.drawImage(originalCanvas, 0, 0, originalCanvas.width, originalCanvas.height, 0, 0, width, height);
      const clipWidth = (sliderPosition / 100) * width;
      ctx.save();
      ctx.beginPath();
      ctx.rect(clipWidth, 0, width - clipWidth, height);
      ctx.clip();
      ctx.drawImage(srCanvas, 0, 0);
      ctx.restore();
      setLegendRange(null);
    } else {
      renderRgbComposite(canvasRef.current, activeRaster, rgbIndices(activeBandOrder));
      setLegendRange(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeRaster, viewerMode, showSplit, original.raster, sliderPosition, uploadResult]);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDraggingSlider.current || !canvasRef.current) return;
      const rect = canvasRef.current.getBoundingClientRect();
      const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
      setSliderPosition((x / rect.width) * 100);
    };
    const handleMouseUp = () => {
      isDraggingSlider.current = false;
    };
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, []);

  const activeCanvas = canvasRef.current;

  const handleCanvasMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (activeTool !== 'roi' || !activeCanvas || !activeRaster) return;
    const sample = samplePixel(activeRaster, activeCanvas, e.clientX, e.clientY);
    if (sample) setRoiDragStart({ x: sample.x, y: sample.y });
  };

  const handleCanvasMouseUp = (e: React.MouseEvent<HTMLDivElement>) => {
    if (activeTool !== 'roi' || !roiDragStart || !activeCanvas || !activeRaster) return;
    const sample = samplePixel(activeRaster, activeCanvas, e.clientX, e.clientY);
    setRoiDragStart(null);
    if (!sample) return;
    const minX = Math.min(roiDragStart.x, sample.x);
    const maxX = Math.max(roiDragStart.x, sample.x);
    const minY = Math.min(roiDragStart.y, sample.y);
    const maxY = Math.max(roiDragStart.y, sample.y);
    if (maxX - minX < 2 || maxY - minY < 2) return; // too small to be a deliberate drag
    const bandCount = activeRaster.bandCount;
    const bandStats: RegionStats['bandStats'] = [];
    const labels = viewerMode === 'evidence' ? ['stability (overall)'] : viewerMode === 'ndvi' ? ['NDVI'] : activeBandOrder;
    const bandsToUse = viewerMode === 'evidence' ? [OVERALL_STD_BAND_INDEX] : Array.from({ length: bandCount }, (_, i) => i);
    for (let b = 0; b < bandsToUse.length; b += 1) {
      const band = activeRaster.bands[bandsToUse[b]];
      let sum = 0;
      let count = 0;
      let min = Infinity;
      let max = -Infinity;
      for (let y = minY; y < maxY; y += 1) {
        for (let x = minX; x < maxX; x += 1) {
          const v = band[y * activeRaster.width + x];
          if (!Number.isFinite(v)) continue;
          sum += v;
          count += 1;
          if (v < min) min = v;
          if (v > max) max = v;
        }
      }
      bandStats.push({ label: labels[b] ?? `band ${bandsToUse[b]}`, mean: count ? sum / count : NaN, min: count ? min : NaN, max: count ? max : NaN });
    }
    const size = pixelSizeFor(viewerMode, jobResult, null);
    const widthMeters = (maxX - minX) * size;
    const heightMeters = (maxY - minY) * size;
    setRoiStats({
      region: { minX, minY, maxX, maxY, pixelSizeMeters: size },
      widthMeters,
      heightMeters,
      areaHectares: (widthMeters * heightMeters) / 10000,
      pixelCount: (maxX - minX) * (maxY - minY),
      bandStats,
    });
  };

  const handleCanvasClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!activeCanvas || !activeRaster) return;
    if (activeTool === 'pixel-probe') {
      const sample = samplePixel(activeRaster, activeCanvas, e.clientX, e.clientY);
      if (sample) setProbe({ ...sample, ndvi: viewerMode === 'ndvi' ? sample.values[0] : viewerMode === 'evidence' ? null : ndviFromOrder(sample.values, activeBandOrder) });
    } else if (activeTool === 'measure') {
      const sample = samplePixel(activeRaster, activeCanvas, e.clientX, e.clientY);
      if (!sample) return;
      if (measurePoints.length >= 2) {
        setMeasurePoints([{ x: sample.x, y: sample.y }]);
        setMeasureMeters(null);
      } else {
        const points = [...measurePoints, { x: sample.x, y: sample.y }];
        setMeasurePoints(points);
        if (points.length === 2) {
          const dx = (points[1].x - points[0].x) * pixelSize;
          const dy = (points[1].y - points[0].y) * pixelSize;
          setMeasureMeters(Math.sqrt(dx * dx + dy * dy));
        }
      }
    }
  };

  const geo = jobResult.metadata.output_geospatial;

  const legendLabel = useMemo(() => {
    if (viewerMode === 'evidence') return 'TTA stability (overall, per-pixel std)';
    if (viewerMode === 'ndvi') return `NDVI (${ndviLayer === 'diff' ? '|SR − native| difference' : ndviLayer === 'native' ? 'native, 10 m grid' : 'SR-derived, 2.5 m grid'})`;
    return null;
  }, [viewerMode, ndviLayer]);

  return (
    <div className="w-full flex-1 flex flex-col bg-[#0B0E12] text-white overflow-hidden relative">
      {/* Top Workspace Command Bar */}
      <div className="h-12 bg-[#151A20] border-b border-[#1D242C] px-6 flex items-center justify-between z-20 flex-wrap gap-2">
        <div className="flex items-center gap-1 bg-[#0B0E12] p-0.5 rounded border border-[#1D242C]">
          <button onClick={() => setViewerMode('original')} className={`px-3 py-1 text-xs font-medium rounded transition-colors ${viewerMode === 'original' ? 'bg-[#1D242C] text-[#D8B84C] font-semibold' : 'text-[#9AA3AD] hover:text-white'}`}>
            1. Original
          </button>
          <button onClick={() => setViewerMode('super-resolved')} className={`px-3 py-1 text-xs font-medium rounded transition-colors ${viewerMode === 'super-resolved' ? 'bg-[#1D242C] text-[#D8B84C] font-semibold' : 'text-[#9AA3AD] hover:text-white'}`}>
            2. Super-resolved
          </button>
          <button onClick={() => setViewerMode('ndvi')} className={`px-3 py-1 text-xs font-medium rounded transition-colors ${viewerMode === 'ndvi' ? 'bg-[#1D242C] text-[#D8B84C] font-semibold' : 'text-[#9AA3AD] hover:text-white'}`}>
            3. NDVI
          </button>
          <button onClick={() => setViewerMode('evidence')} className={`px-3 py-1 text-xs font-medium rounded transition-colors ${viewerMode === 'evidence' ? 'bg-[#1D242C] text-[#D8B84C] font-semibold' : 'text-[#9AA3AD] hover:text-white'}`}>
            4. Evidence
          </button>
        </div>

        {viewerMode === 'ndvi' && (
          <div className="flex items-center gap-1.5 text-xs">
            <span className="text-[#9AA3AD] text-[11px] font-mono">LAYER:</span>
            {(['native', 'sr', 'diff'] as const).map((layer) => (
              <button key={layer} onClick={() => setNdviLayer(layer)} className={`px-2.5 py-0.5 uppercase text-[11px] font-mono rounded ${ndviLayer === layer ? 'bg-[#D8B84C] text-[#0B0E12] font-bold' : 'text-[#9AA3AD] hover:text-white bg-[#151A20]'}`}>
                {layer}
              </button>
            ))}
          </div>
        )}

        <div className="flex items-center gap-2">
          {viewerMode === 'super-resolved' && (
            <button
              onClick={() => setIsSplitCompare(!isSplitCompare)}
              className={`flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded border cursor-pointer transition-colors ${isSplitCompare ? 'bg-[#151A20] border-[#D8B84C] text-[#D8B84C]' : 'bg-[#151A20] border-[#1D242C] text-[#9AA3AD] hover:text-white'}`}
              title="Compare against the original input"
            >
              <Split className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Split compare</span>
            </button>
          )}

          <div className="flex items-center bg-[#0B0E12] p-0.5 rounded border border-[#1D242C]">
            <button onClick={() => setActiveTool('pan')} className={`p-1.5 rounded transition-colors cursor-pointer ${activeTool === 'pan' ? 'bg-[#1D242C] text-white' : 'text-[#9AA3AD] hover:text-white'}`} title="Pan">
              <Eye className="w-3.5 h-3.5" />
            </button>
            <button onClick={() => setActiveTool('roi')} className={`p-1.5 rounded transition-colors cursor-pointer ${activeTool === 'roi' ? 'bg-[#1D242C] text-[#D8B84C]' : 'text-[#9AA3AD] hover:text-white'}`} title="Drag a region for real area and band statistics">
              <Square className="w-3.5 h-3.5" />
            </button>
            <button onClick={() => setActiveTool('pixel-probe')} className={`p-1.5 rounded transition-colors cursor-pointer ${activeTool === 'pixel-probe' ? 'bg-[#1D242C] text-[#D8B84C]' : 'text-[#9AA3AD] hover:text-white'}`} title="Read the real pixel value under the cursor">
              <Crosshair className="w-3.5 h-3.5" />
            </button>
            <button onClick={() => setActiveTool('measure')} className={`p-1.5 rounded transition-colors cursor-pointer ${activeTool === 'measure' ? 'bg-[#1D242C] text-[#D8B84C]' : 'text-[#9AA3AD] hover:text-white'}`} title="Measure a real ground distance">
              <Ruler className="w-3.5 h-3.5" />
            </button>
          </div>

          <button onClick={() => setShowMetadataDrawer(!showMetadataDrawer)} className="flex items-center gap-1 px-2 py-1 text-xs text-[#9AA3AD] hover:text-white bg-[#151A20] hover:bg-[#1D242C] border border-[#1D242C] rounded transition-colors cursor-pointer">
            <Info className="w-3.5 h-3.5 text-[#D8B84C]" />
            <span className="hidden md:inline">Metadata</span>
          </button>
        </div>
      </div>

      {/* Main Imagery Viewer Stage */}
      <div
        onMouseDown={handleCanvasMouseDown}
        onMouseUp={handleCanvasMouseUp}
        onClick={handleCanvasClick}
        className={`relative flex-1 bg-[#0b0e14] overflow-hidden select-none flex items-center justify-center ${
          activeTool === 'pixel-probe' || activeTool === 'measure' ? 'cursor-crosshair' : activeTool === 'roi' ? 'cursor-cell' : 'cursor-grab'
        }`}
      >
        {viewerMode === 'original' && !file && (
          <p className="text-xs text-[#9AA3AD]">No original file is available for this reconstruction (opened from a previous session entry).</p>
        )}
        {viewerMode === 'ndvi' && !ndviAnalysis && (
          <div className="text-center max-w-sm">
            <p className="text-xs text-[#9AA3AD] mb-3">NDVI has not been run for this reconstruction yet.</p>
            <button onClick={onNavigateToAnalyze} className="px-3 py-1.5 bg-[#D8B84C] text-[#0B0E12] font-semibold text-xs rounded hover:bg-[#E4C96A] transition-colors cursor-pointer">
              Run it in Analyze
            </button>
          </div>
        )}
        {isLoading && <Loader2 className="w-6 h-6 text-[#D8B84C] animate-spin" />}
        {loadError && <p className="text-xs text-[#D94B4B] max-w-sm text-center">{loadError}</p>}

        {activeRaster && !isLoading && (
          <div className="relative max-w-full max-h-full" style={{ aspectRatio: `${activeRaster.width} / ${activeRaster.height}` }}>
            <canvas ref={canvasRef} className="w-full h-full object-contain block" />

            {showSplit && (
              <>
                <div style={{ left: `${sliderPosition}%` }} className="absolute top-0 bottom-0 w-0.5 bg-[#D8B84C] z-10 pointer-events-none">
                  <div
                    onMouseDown={(e) => {
                      e.stopPropagation();
                      isDraggingSlider.current = true;
                    }}
                    className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-8 h-8 rounded-full bg-[#151A20] border-2 border-[#D8B84C] shadow-xl flex items-center justify-center cursor-ew-resize pointer-events-auto"
                  >
                    <Split className="w-3.5 h-3.5 text-[#D8B84C]" />
                  </div>
                </div>
                <div className="absolute top-3 left-3 bg-[#0B0E12]/90 border border-[#1D242C] px-2 py-0.5 rounded text-[11px] font-mono text-[#9AA3AD] pointer-events-none whitespace-nowrap z-10">
                  Original input
                </div>
                <div className="absolute top-3 right-3 bg-[#0B0E12]/90 border border-[#1D242C] px-2 py-0.5 rounded text-[11px] font-mono text-white pointer-events-none whitespace-nowrap z-10">
                  SR ({jobResult.resolution.sr_resolution_m} m)
                </div>
              </>
            )}

            {measurePoints.map((pt, i) => (
              <div key={i} className="absolute w-2 h-2 rounded-full bg-[#D8B84C] -translate-x-1/2 -translate-y-1/2 pointer-events-none" style={{ left: `${(pt.x / activeRaster.width) * 100}%`, top: `${(pt.y / activeRaster.height) * 100}%` }} />
            ))}

            {roiStats && (
              <div
                className="absolute border-2 border-[#D8B84C] bg-[#D8B84C]/10 pointer-events-none"
                style={{
                  left: `${(roiStats.region.minX / activeRaster.width) * 100}%`,
                  top: `${(roiStats.region.minY / activeRaster.height) * 100}%`,
                  width: `${((roiStats.region.maxX - roiStats.region.minX) / activeRaster.width) * 100}%`,
                  height: `${((roiStats.region.maxY - roiStats.region.minY) / activeRaster.height) * 100}%`,
                }}
              />
            )}
          </div>
        )}

        {/* Floating Bottom Left Telemetry HUD */}
        <div className="absolute bottom-3 left-3 bg-[#0B0E12]/95 border border-[#1D242C] px-3 py-1.5 rounded text-xs font-mono text-[#9AA3AD] space-y-0.5 pointer-events-none z-10 shadow-lg">
          <div className="flex items-center gap-3">
            <span className="text-white font-semibold">{jobResult.model_name}</span>
            <span>·</span>
            <span>CRS: {geo.crs ?? '—'}</span>
            <span>·</span>
            <span className="text-[#D8B84C]">{pixelSize} m / px</span>
          </div>
        </div>

        {legendLabel && legendRange && (
          <div className="absolute bottom-3 right-3 bg-[#0B0E12]/95 border border-[#1D242C] px-3 py-2 rounded text-xs z-10 shadow-lg">
            <span className="text-[11px] font-mono text-[#D8B84C] uppercase block mb-1">{legendLabel}</span>
            <div className="flex items-center gap-2 text-[10px] font-mono text-[#9AA3AD]">
              <div className="w-12 h-2 bg-gradient-to-r from-[#0f1620] to-[#D8B84C] rounded" />
              <span>{legendRange[0].toExponential(2)} → {legendRange[1].toExponential(2)} (2nd–98th percentile)</span>
            </div>
          </div>
        )}
        {viewerMode === 'ndvi' && legendLabel && (
          <div className="absolute bottom-3 right-3 bg-[#0B0E12]/95 border border-[#1D242C] px-3 py-2 rounded text-xs z-10 shadow-lg">
            <span className="text-[11px] font-mono text-[#D8B84C] uppercase block mb-1">{legendLabel}</span>
            <div className="flex items-center gap-2 text-[10px] font-mono text-[#9AA3AD]">
              <div className="w-12 h-2 bg-gradient-to-r from-[#b91c1c] via-[#D8B84C] to-[#16a34a] rounded" />
              <span>-1.0 → +1.0</span>
            </div>
          </div>
        )}
      </div>

      {/* Bottom Panel: Region & Pixel Probe results */}
      <div className="border-t border-[#1D242C] bg-[#151A20] p-4 z-20">
        <div className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-12 gap-4 items-start">
          <div className="md:col-span-5 p-3 bg-[#0B0E12] rounded border border-[#1D242C]">
            <div className="flex items-center gap-2 mb-1">
              <Square className="w-3.5 h-3.5 text-[#D8B84C]" />
              <span className="text-xs font-semibold text-white">{roiStats ? `${roiStats.pixelCount.toLocaleString()} px region` : measureMeters !== null ? 'Measured distance' : 'No region drawn'}</span>
            </div>
            {roiStats ? (
              <div className="text-[11px] font-mono text-[#9AA3AD] space-y-1">
                <div>{roiStats.widthMeters.toFixed(1)} m × {roiStats.heightMeters.toFixed(1)} m — {roiStats.areaHectares.toFixed(3)} ha</div>
                {roiStats.bandStats.map((b) => (
                  <div key={b.label}>{b.label}: mean {b.mean.toFixed(4)} (min {b.min.toFixed(4)}, max {b.max.toFixed(4)})</div>
                ))}
              </div>
            ) : measureMeters !== null ? (
              <span className="text-[11px] font-mono text-[#D8B84C]">{measureMeters.toFixed(2)} m</span>
            ) : (
              <span className="text-[11px] text-[#9AA3AD]">Use the region tool to drag a box, or the ruler to click two points.</span>
            )}
          </div>

          <div className="md:col-span-7 p-3 bg-[#0B0E12] rounded border border-[#1D242C]">
            <div className="flex items-center justify-between mb-1.5 text-xs">
              <div className="flex items-center gap-2">
                <Crosshair className="w-3.5 h-3.5 text-[#D8B84C]" />
                <span className="font-semibold text-white">{probe ? `Pixel (${probe.x}, ${probe.y})` : 'Use the probe tool and click the image'}</span>
              </div>
              {probe?.ndvi !== null && probe?.ndvi !== undefined && (
                <span className="text-[11px] font-mono text-[#D8B84C]">NDVI: {probe.ndvi.toFixed(3)}</span>
              )}
            </div>
            {probe && (
              <div className="flex flex-wrap gap-3 text-[11px] font-mono text-[#9AA3AD]">
                {probe.values.map((v, i) => (
                  <span key={i}>
                    {viewerMode === 'evidence' ? `band ${i}` : viewerMode === 'ndvi' ? 'NDVI' : activeBandOrder[i] ?? `band ${i}`}: {Number.isFinite(v) ? v.toFixed(5) : '—'}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Geospatial Metadata Drawer */}
      {showMetadataDrawer && (
        <div className="absolute right-0 top-12 bottom-0 w-80 bg-[#151A20] border-l border-[#1D242C] p-5 z-30 shadow-2xl flex flex-col justify-between overflow-y-auto">
          <div>
            <div className="flex items-center justify-between pb-3 border-b border-[#1D242C] mb-4">
              <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-white">Geospatial metadata</h3>
              <button onClick={() => setShowMetadataDrawer(false)} className="text-[#9AA3AD] hover:text-white p-1 rounded hover:bg-[#1D242C] cursor-pointer">
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="space-y-3 text-xs">
              <div>
                <span className="text-[11px] font-mono text-[#9AA3AD] block">Output pixel grid</span>
                <span className="font-mono text-[#D8B84C] font-semibold">{jobResult.resolution.sr_resolution_m} m / px</span>
                <span className="text-[10px] text-[#9AA3AD] block">Input: {jobResult.resolution.native_resolution_m} m / px</span>
              </div>
              <div>
                <span className="text-[11px] font-mono text-[#9AA3AD] block">CRS</span>
                <span className="font-mono text-white">{geo.crs ?? '—'}</span>
              </div>
              <div>
                <span className="text-[11px] font-mono text-[#9AA3AD] block">Affine transform</span>
                <pre className="font-mono text-[10px] text-[#9AA3AD] bg-[#0B0E12] p-2 rounded border border-[#1D242C] mt-1 whitespace-pre-wrap break-all">
                  {geo.transform.map((v) => v.toFixed(4)).join(', ')}
                </pre>
              </div>
              <div>
                <span className="text-[11px] font-mono text-[#9AA3AD] block">Output dimensions</span>
                <span className="text-white font-mono">{geo.width} × {geo.height} px</span>
              </div>
              <div>
                <span className="text-[11px] font-mono text-[#9AA3AD] block">Bands</span>
                <span className="text-white">{jobResult.bands.join(', ')}</span>
              </div>
              <div>
                <span className="text-[11px] font-mono text-[#9AA3AD] block">Model</span>
                <span className="text-white">{jobResult.model_name}</span>
              </div>
              <div>
                <span className="text-[11px] font-mono text-[#9AA3AD] block">Valid-pixel coverage</span>
                <span className="text-white">{(jobResult.metadata.preprocessing_mask_coverage * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>
          <div className="pt-4 border-t border-[#1D242C]">
            <p className="text-[10px] text-[#9AA3AD] leading-relaxed">
              These pixel boundaries represent a super-resolved mathematical reconstruction, not a directly measured Sentinel-2 ground detection.
            </p>
            <button onClick={onNavigateToValidate} className="mt-2 text-[10px] text-[#D8B84C] hover:underline">See the full validation evidence →</button>
          </div>
        </div>
      )}
    </div>
  );
};
