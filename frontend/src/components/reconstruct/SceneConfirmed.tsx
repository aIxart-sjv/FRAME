/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { AlertTriangle, ArrowRight, CheckCircle2, Cpu, Download, Loader2, RotateCcw } from 'lucide-react';
import type { AsyncStatus } from '../../state/useFrameSession';
import type { AcquisitionRetrieveResponse, SceneCandidate } from '../../api/types';
import { acquisitionDownloadUrl } from '../../api/client';

interface SceneConfirmedProps {
  scene: SceneCandidate;
  retrieveStatus: AsyncStatus;
  retrieveResult: AcquisitionRetrieveResponse | null;
  retrieveError: string | null;
  retrieveResolutionM: number;
  onRetrieve: () => void;
  onCancelRetrieve: () => void;
  prepareSrStatus: AsyncStatus;
  prepareSrError: string | null;
  onRunSr: () => void;
  onCancelPrepareSr: () => void;
  onContinueToConfigure: () => void;
  onStartOver: () => void;
}

/**
 * A real selected scene: retrieve real pixels (POST /acquisition/retrieve), then hand them off into the
 * existing, unmodified SR pipeline (POST /acquisition/retrieve/{id}/prepare-sr-input -> the same POST /sr/run
 * a manual upload uses). The backend is synchronous end to end -- these buttons show that honestly (a spinner
 * while a real request is in flight, a real error if it fails), never a fabricated queue or progress bar.
 */
export const SceneConfirmed: React.FC<SceneConfirmedProps> = ({
  scene,
  retrieveStatus,
  retrieveResult,
  retrieveError,
  retrieveResolutionM,
  onRetrieve,
  onCancelRetrieve,
  prepareSrStatus,
  prepareSrError,
  onRunSr,
  onCancelPrepareSr,
  onContinueToConfigure,
  onStartOver,
}) => {
  return (
    <div className="w-full min-h-[calc(100vh-3.5rem)] bg-[#0B0E12] text-white flex flex-col items-center p-6">
      <div className="max-w-2xl w-full bg-[#151A20] border border-[#1D242C] rounded-xl overflow-hidden shadow-2xl mt-10">
        <div className="p-6 border-b border-[#1D242C] flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2 text-xs text-[#35C98A] font-semibold mb-1">
              <CheckCircle2 className="w-4 h-4" />
              <span>Scene selected</span>
            </div>
            <h2 className="text-lg font-bold tracking-tight text-white">{scene.mgrs_tile ?? scene.id}</h2>
            <p className="text-xs font-mono text-[#9AA3AD] mt-0.5 break-all">{scene.id}</p>
          </div>
          <button
            onClick={onStartOver}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-[#9AA3AD] hover:text-white bg-[#1D242C] hover:bg-[#343C45] rounded transition-colors cursor-pointer shrink-0"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Start over</span>
          </button>
        </div>

        <div className="p-5 grid grid-cols-2 gap-3 text-xs border-b border-[#1D242C]">
          <div>
            <span className="text-[11px] font-mono text-[#9AA3AD] block">Provider</span>
            <span className="font-mono text-white">{scene.provider}</span>
          </div>
          <div>
            <span className="text-[11px] font-mono text-[#9AA3AD] block">Acquisition datetime</span>
            <span className="font-mono text-white">{scene.datetime}</span>
          </div>
          <div>
            <span className="text-[11px] font-mono text-[#9AA3AD] block">Cloud cover</span>
            <span className="font-mono text-white">{scene.cloud_cover !== null ? `${scene.cloud_cover.toFixed(1)}%` : '—'}</span>
          </div>
          <div>
            <span className="text-[11px] font-mono text-[#9AA3AD] block">Processing baseline</span>
            <span className="font-mono text-white">{scene.processing_baseline ?? '—'}</span>
          </div>
        </div>

        <div className="p-5 space-y-4">
          <p className="text-xs text-[#9AA3AD] leading-relaxed">
            Retrieve real pixels for this scene (POST /acquisition/retrieve — CDSE Process API, with Planetary
            Computer as a disclosed fallback), at a resolution automatically coarsened to keep the request's real
            size bounded (~{retrieveResolutionM} m/pixel for this AOI — never the full native resolution for an
            arbitrarily large drawn area). The four retrieved bands (B04/B03/B02/B08) can then be run through the
            existing RGBN super-resolution pipeline unchanged — the same one a manually-uploaded GeoTIFF uses.
          </p>

          {retrieveStatus !== 'success' && (
            <div className="flex gap-2">
              <button
                onClick={onRetrieve}
                disabled={retrieveStatus === 'pending'}
                className="flex items-center justify-center gap-2 flex-1 px-5 py-2.5 text-xs font-semibold rounded transition-all cursor-pointer bg-[#D8B84C] hover:bg-[#E4C96A] text-[#151A20] shadow-md disabled:opacity-60 disabled:cursor-not-allowed"
              >
                {retrieveStatus === 'pending' ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Download className="w-3.5 h-3.5" />}
                <span>{retrieveStatus === 'pending' ? 'Retrieving real pixels…' : 'Retrieve real pixels for this scene'}</span>
              </button>
              {retrieveStatus === 'pending' && (
                <button
                  onClick={onCancelRetrieve}
                  className="px-4 py-2.5 text-xs font-medium text-[#9AA3AD] hover:text-white bg-[#1D242C] hover:bg-[#343C45] rounded transition-colors cursor-pointer"
                >
                  Cancel
                </button>
              )}
            </div>
          )}

          {retrieveStatus === 'error' && retrieveError && (
            <div className="p-3 bg-[#0B0E12] border border-[#D94B4B]/40 rounded-lg flex items-start gap-2.5 text-xs">
              <AlertTriangle className="w-3.5 h-3.5 text-[#D94B4B] shrink-0 mt-0.5" />
              <span className="text-[#9AA3AD] leading-relaxed">{retrieveError}</span>
            </div>
          )}

          {retrieveStatus === 'success' && retrieveResult && (
            <div className="p-4 bg-[#0B0E12] border border-[#35C98A]/30 rounded-lg space-y-3 text-xs">
              <div className="flex items-center gap-2 text-[#35C98A] font-semibold">
                <CheckCircle2 className="w-4 h-4" />
                <span>Real pixels retrieved</span>
              </div>
              <div className="grid grid-cols-2 gap-3 font-mono text-[11px] text-[#9AA3AD]">
                <div>Bands: <span className="text-white">{retrieveResult.bands.join(', ')}</span></div>
                <div>Grid: <span className="text-white">{retrieveResult.width} × {retrieveResult.height} @ {retrieveResult.resolution_m} m</span></div>
                <div>CRS: <span className="text-white">{retrieveResult.crs}</span></div>
                <div>Provider: <span className="text-white">{retrieveResult.retrieval_provider}</span></div>
              </div>
              {retrieveResult.warnings.length > 0 && (
                <ul className="text-[11px] text-[#D8B84C] list-disc pl-4 space-y-0.5">
                  {retrieveResult.warnings.map((w) => (
                    <li key={w}>{w}</li>
                  ))}
                </ul>
              )}
              <div className="flex flex-wrap gap-2 pt-1">
                {Object.entries(retrieveResult.artifacts).map(([label, path]) => (
                  <a
                    key={label}
                    href={acquisitionDownloadUrl(path)}
                    className="flex items-center gap-1.5 px-3 py-1.5 bg-[#1D242C] hover:bg-[#343C45] rounded text-[11px] font-medium text-white transition-colors"
                  >
                    <Download className="w-3 h-3" />
                    <span>{label.replace(/_/g, ' ')}</span>
                  </a>
                ))}
              </div>
            </div>
          )}

          {retrieveStatus === 'success' && retrieveResult && prepareSrStatus !== 'success' && (
            <div className="flex gap-2">
              <button
                onClick={onRunSr}
                disabled={prepareSrStatus === 'pending'}
                className="flex items-center justify-center gap-2 flex-1 px-5 py-2.5 text-xs font-semibold rounded transition-all cursor-pointer bg-[#D8B84C] hover:bg-[#E4C96A] text-[#151A20] shadow-md disabled:opacity-60 disabled:cursor-not-allowed"
              >
                {prepareSrStatus === 'pending' ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Cpu className="w-3.5 h-3.5" />}
                <span>{prepareSrStatus === 'pending' ? 'Preparing SR input…' : 'Run Super Resolution'}</span>
              </button>
              {prepareSrStatus === 'pending' && (
                <button
                  onClick={onCancelPrepareSr}
                  className="px-4 py-2.5 text-xs font-medium text-[#9AA3AD] hover:text-white bg-[#1D242C] hover:bg-[#343C45] rounded transition-colors cursor-pointer"
                >
                  Cancel
                </button>
              )}
            </div>
          )}

          {prepareSrStatus === 'error' && prepareSrError && (
            <div className="p-3 bg-[#0B0E12] border border-[#D94B4B]/40 rounded-lg flex items-start gap-2.5 text-xs">
              <AlertTriangle className="w-3.5 h-3.5 text-[#D94B4B] shrink-0 mt-0.5" />
              <span className="text-[#9AA3AD] leading-relaxed">{prepareSrError}</span>
            </div>
          )}

          {prepareSrStatus === 'success' && (
            <button
              onClick={onContinueToConfigure}
              className="flex items-center justify-center gap-2 w-full px-5 py-2.5 text-xs font-semibold rounded transition-all cursor-pointer bg-[#35C98A] hover:bg-[#2FB37A] text-[#0B0E12] shadow-md"
            >
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>SR input ready — continue to configuration</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
