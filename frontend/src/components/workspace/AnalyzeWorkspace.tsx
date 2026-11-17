/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useEffect, useRef } from 'react';
import { Info, Loader2 } from 'lucide-react';
import * as api from '../../api/client';
import { useRemoteRaster } from '../../hooks/useRaster';
import { renderNdviBand, renderSequentialBand } from '../../lib/raster';
import type { AsyncStatus } from '../../state/useFrameSession';
import type { NDVIAnalysisResponse } from '../../api/types';

interface AnalyzeWorkspaceProps {
  analysisStatus: AsyncStatus;
  analysisResult: NDVIAnalysisResponse | null;
  analysisError: string | null;
  onRunAnalysis: () => void;
}

function NdviPanel({ title, cacheKey, fetcher, kind }: { title: string; cacheKey: string; fetcher: () => Promise<Blob>; kind: 'ndvi' | 'diff' }) {
  const raster = useRemoteRaster(fetcher, cacheKey);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (!canvasRef.current || !raster.raster) return;
    if (kind === 'ndvi') renderNdviBand(canvasRef.current, raster.raster.bands[0], raster.raster.width, raster.raster.height);
    else renderSequentialBand(canvasRef.current, raster.raster.bands[0], raster.raster.width, raster.raster.height);
  }, [raster.raster, kind]);

  return (
    <div className="bg-[#151A20] border border-[#1D242C] rounded-lg overflow-hidden flex flex-col">
      <div className="px-3 py-2 border-b border-[#1D242C] text-[11px] font-mono text-[#9AA3AD]">{title}</div>
      <div className="relative aspect-square bg-[#0c1017] flex items-center justify-center">
        {raster.status === 'pending' && <Loader2 className="w-5 h-5 text-[#D8B84C] animate-spin" />}
        {raster.status === 'error' && <span className="text-[11px] text-[#D94B4B] p-3 text-center">{raster.error}</span>}
        {raster.raster && <canvas ref={canvasRef} className="w-full h-full object-contain" />}
      </div>
    </div>
  );
}

/**
 * The only downstream analysis this backend actually computes: an NDVI demonstration comparing the SR product's
 * NDVI with the NDVI of its own low-resolution input (`POST /analysis/ndvi`). There is no land-cover, NDBI/NDWI/
 * BSI, or multi-temporal change detection here — the backend has no such endpoint, and none of FRAME's own
 * research measured them for an arbitrary uploaded scene (docs/CLAIMS.md in the repository).
 */
export const AnalyzeWorkspace: React.FC<AnalyzeWorkspaceProps> = ({ analysisStatus, analysisResult, analysisError, onRunAnalysis }) => {
  return (
    <div className="w-full flex-1 flex flex-col bg-[#0B0E12] text-white overflow-y-auto">
      <div className="bg-[#151A20] border-b border-[#1D242C] px-6 py-4">
        <p className="text-xs font-mono uppercase tracking-wider text-[#D8B84C] mb-0.5">Downstream Utility Demonstration</p>
        <h1 className="text-xl font-bold tracking-tight text-white">Does the reconstruction change an NDVI-derived reading?</h1>
        <p className="text-xs text-[#9AA3AD] max-w-2xl">
          Compares NDVI computed from the SR product with NDVI computed from its own low-resolution input, on this one scene. This is a demonstration of a downstream computation, not a reference-based accuracy test —
          FRAME's own reference-based tests found the effect small and dataset-dependent, with no consistent advantage established (see the Validate tab).
        </p>
      </div>

      <div className="max-w-5xl w-full mx-auto p-6 flex-1 flex flex-col gap-6">
        {!analysisResult && (
          <div className="p-8 text-center bg-[#151A20] border border-[#1D242C] rounded-lg">
            <p className="text-xs text-[#9AA3AD] mb-4">Run the NDVI demonstration against this reconstruction.</p>
            <button
              disabled={analysisStatus === 'pending'}
              onClick={onRunAnalysis}
              className="px-5 py-2.5 bg-[#D8B84C] hover:bg-[#E4C96A] text-[#151A20] font-bold text-xs rounded transition-all inline-flex items-center gap-2 cursor-pointer disabled:opacity-60"
            >
              {analysisStatus === 'pending' && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
              <span>{analysisStatus === 'pending' ? 'Running…' : 'Run NDVI analysis'}</span>
            </button>
            {analysisStatus === 'error' && analysisError && (
              <p className="text-[11px] text-[#D94B4B] mt-3 max-w-md mx-auto">{analysisError}</p>
            )}
          </div>
        )}

        {analysisResult && (
          <>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <NdviPanel title="Native NDVI (10 m grid)" cacheKey={`${analysisResult.analysis_id}:native`} fetcher={() => api.fetchNdviGeotiff(analysisResult.analysis_id, 'native-ndvi')} kind="ndvi" />
              <NdviPanel title="SR-derived NDVI (2.5 m grid)" cacheKey={`${analysisResult.analysis_id}:sr`} fetcher={() => api.fetchNdviGeotiff(analysisResult.analysis_id, 'sr-ndvi')} kind="ndvi" />
              <NdviPanel title="Absolute difference" cacheKey={`${analysisResult.analysis_id}:diff`} fetcher={() => api.fetchNdviGeotiff(analysisResult.analysis_id, 'ndvi-diff')} kind="diff" />
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              {[
                ['Valid pixels', analysisResult.comparison.valid_pixel_count.toLocaleString()],
                ['Mean abs. difference', analysisResult.comparison.mean_abs_difference?.toFixed(4) ?? '—'],
                ['RMSE', analysisResult.comparison.rmse?.toFixed(4) ?? '—'],
                ['Max abs. difference', analysisResult.comparison.max_abs_difference?.toFixed(4) ?? '—'],
                ['Resampling', analysisResult.comparison.resampling_method],
                ['Stability ↔ disagreement corr.', analysisResult.uncertainty_weighted_summary.correlation_uncertainty_vs_abs_diff?.toFixed(3) ?? 'not computable'],
                ['Stability-weighted mean abs. diff.', analysisResult.uncertainty_weighted_summary.uncertainty_weighted_mean_abs_diff?.toFixed(4) ?? '—'],
                ['Unweighted mean abs. diff.', analysisResult.uncertainty_weighted_summary.unweighted_mean_abs_diff?.toFixed(4) ?? '—'],
              ].map(([label, value]) => (
                <div key={label} className="p-3 bg-[#151A20] border border-[#1D242C] rounded">
                  <span className="text-[11px] font-mono text-[#9AA3AD] block">{label}</span>
                  <span className="text-sm font-mono text-white">{value}</span>
                </div>
              ))}
            </div>

            <div className="p-4 bg-[#151A20] border border-[#343C45] rounded-lg flex items-start gap-3">
              <Info className="w-4 h-4 text-[#D8B84C] shrink-0 mt-0.5" />
              <ul className="text-[11px] text-[#9AA3AD] space-y-1 leading-relaxed">
                {analysisResult.scientific_caveats.map((c) => (
                  <li key={c}>{c}</li>
                ))}
              </ul>
            </div>
          </>
        )}
      </div>
    </div>
  );
};
