/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { CheckCircle2, Download, FileJson, Layers, Loader2, X } from 'lucide-react';
import * as api from '../../api/client';
import { downloadBlob, downloadJson } from '../../lib/download';
import type { NDVIAnalysisResponse, SRResultResponse } from '../../api/types';

interface ExportDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  jobResult: SRResultResponse | null;
  ndviAnalysis: NDVIAnalysisResponse | null;
}

type DownloadKey = 'sr' | 'uncertainty' | 'result-json' | 'ndvi-native' | 'ndvi-sr' | 'ndvi-diff' | 'ndvi-json';

/**
 * Every option here is a real file this backend can actually serve — the SR and stability GeoTIFFs, the raw JSON
 * result, and (once run) the three NDVI GeoTIFFs and the NDVI JSON. There is no bundling, no STAC record, no
 * land-cover raster, and no PDF report: the backend produces none of those, so none is offered.
 */
export const ExportDrawer: React.FC<ExportDrawerProps> = ({ isOpen, onClose, jobResult, ndviAnalysis }) => {
  const [pending, setPending] = useState<DownloadKey | null>(null);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen || !jobResult) return null;

  const run = async (key: DownloadKey, action: () => Promise<void>) => {
    setPending(key);
    setError(null);
    try {
      await action();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Download failed.');
    } finally {
      setPending(null);
    }
  };

  const items: Array<{ key: DownloadKey; label: string; format: string; description: string; action: () => Promise<void> }> = [
    {
      key: 'sr',
      label: 'SR-derived GeoTIFF',
      format: 'GeoTIFF, float32',
      description: `${jobResult.bands.join(', ')} — ${jobResult.output_shape.join(' × ')}, ${jobResult.resolution.sr_resolution_m} m pixel grid, CRS ${jobResult.crs ?? '—'}.`,
      action: async () => downloadBlob(await api.fetchSrGeotiff(jobResult.job_id), `sr_${jobResult.job_id}.tif`),
    },
    {
      key: 'uncertainty',
      label: 'TTA stability GeoTIFF',
      format: 'GeoTIFF, float32, 5 bands',
      description: 'Per-band standard deviation plus the overall stability band, on the same grid as the SR output.',
      action: async () => downloadBlob(await api.fetchUncertaintyGeotiff(jobResult.job_id), `stability_${jobResult.job_id}.tif`),
    },
    {
      key: 'result-json',
      label: 'Reconstruction result (JSON)',
      format: 'JSON',
      description: 'The exact response of POST /sr/run for this job — model, timing, geospatial metadata, stability summary, caveats.',
      action: async () => downloadJson(jobResult, `result_${jobResult.job_id}.json`),
    },
  ];

  if (ndviAnalysis) {
    items.push(
      { key: 'ndvi-native', label: 'Native NDVI GeoTIFF', format: 'GeoTIFF, float32, 1 band', description: '10 m grid.', action: async () => downloadBlob(await api.fetchNdviGeotiff(ndviAnalysis.analysis_id, 'native-ndvi'), `ndvi_native_${ndviAnalysis.analysis_id}.tif`) },
      { key: 'ndvi-sr', label: 'SR-derived NDVI GeoTIFF', format: 'GeoTIFF, float32, 1 band', description: `${jobResult.resolution.sr_resolution_m} m grid.`, action: async () => downloadBlob(await api.fetchNdviGeotiff(ndviAnalysis.analysis_id, 'sr-ndvi'), `ndvi_sr_${ndviAnalysis.analysis_id}.tif`) },
      { key: 'ndvi-diff', label: 'NDVI absolute-difference GeoTIFF', format: 'GeoTIFF, float32, 1 band', description: '10 m grid.', action: async () => downloadBlob(await api.fetchNdviGeotiff(ndviAnalysis.analysis_id, 'ndvi-diff'), `ndvi_diff_${ndviAnalysis.analysis_id}.tif`) },
      { key: 'ndvi-json', label: 'NDVI analysis result (JSON)', format: 'JSON', description: 'The exact response of POST /analysis/ndvi for this job.', action: async () => downloadJson(ndviAnalysis, `ndvi_${ndviAnalysis.analysis_id}.json`) },
    );
  }

  return (
    <div className="fixed inset-0 z-50 overflow-hidden flex justify-end">
      <div onClick={onClose} className="fixed inset-0 bg-black/60 backdrop-blur-xs transition-opacity" />
      <div className="relative w-full max-w-md bg-[#151A20] border-l border-[#1D242C] text-white flex flex-col z-10 shadow-2xl">
        <div className="p-6 border-b border-[#1D242C] flex items-center justify-between">
          <div>
            <h2 className="text-base font-bold tracking-tight text-white">Download outputs</h2>
            <p className="text-xs text-[#9AA3AD] mt-0.5 font-mono">{jobResult.model_name} · {jobResult.job_id.slice(0, 8)}…</p>
          </div>
          <button onClick={onClose} className="p-1 rounded text-[#9AA3AD] hover:text-white hover:bg-[#1D242C] transition-colors cursor-pointer">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-6 flex-1 overflow-y-auto space-y-3">
          {!ndviAnalysis && (
            <p className="text-[11px] text-[#9AA3AD] flex items-start gap-2">
              <Layers className="w-3.5 h-3.5 text-[#D8B84C] shrink-0 mt-0.5" />
              <span>Run the NDVI analysis in the Analyze tab to also export its three rasters and its JSON result.</span>
            </p>
          )}
          {items.map((item) => (
            <div key={item.key} className="p-3.5 bg-[#0B0E12] border border-[#1D242C] hover:border-[#343C45] rounded-lg">
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs font-semibold text-white flex items-center gap-1.5">
                  {item.key.includes('json') ? <FileJson className="w-3.5 h-3.5 text-[#D8B84C]" /> : <Layers className="w-3.5 h-3.5 text-[#D8B84C]" />}
                  {item.label}
                </span>
                <span className="text-[10px] font-mono px-1.5 py-0.2 bg-[#1D242C] rounded text-[#9AA3AD] shrink-0">{item.format}</span>
              </div>
              <p className="text-[11px] text-[#9AA3AD] mt-1 leading-normal">{item.description}</p>
              <button
                disabled={pending !== null}
                onClick={() => run(item.key, item.action)}
                className="mt-2 flex items-center gap-1.5 px-3 py-1.5 bg-[#1D242C] hover:bg-[#343C45] text-xs font-medium text-white rounded transition-colors cursor-pointer disabled:opacity-50"
              >
                {pending === item.key ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Download className="w-3.5 h-3.5" />}
                <span>{pending === item.key ? 'Downloading…' : 'Download'}</span>
              </button>
            </div>
          ))}
          {error && <p className="text-[11px] text-[#D94B4B]">{error}</p>}
        </div>

        <div className="p-4 border-t border-[#1D242C] text-[10px] text-[#9AA3AD] flex items-center gap-1.5">
          <CheckCircle2 className="w-3 h-3 text-[#35C98A]" />
          <span>Each file streams directly from the backend — nothing is bundled or re-encoded here.</span>
        </div>
      </div>
    </div>
  );
};
