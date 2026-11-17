/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useEffect, useRef } from 'react';
import { AlertTriangle, ArrowRight, CheckCircle2, Loader2, Terminal } from 'lucide-react';
import type { AsyncStatus } from '../../state/useFrameSession';
import type { ModelId, SRResultResponse, UploadResponse } from '../../api/types';

interface ProcessingJobProps {
  uploadResult: UploadResponse;
  model: ModelId;
  jobStatus: AsyncStatus;
  jobResult: SRResultResponse | null;
  jobError: string | null;
  onRun: () => void;
  onCancel: () => void;
  onComplete: () => void;
}

const MODEL_LABEL: Record<ModelId, string> = { lite: 'SEN2SR-Lite', mamba: 'SEN2SR-Mamba' };

/**
 * `POST /sr/run` is one synchronous request (frame/api/README.md): there is no job queue, no stage-by-stage
 * status the server reports mid-request, and no way to poll progress. This screen fires that one request, waits
 * on it for real, and only shows facts drawn from the response that actually comes back -- never a fabricated
 * multi-stage timeline.
 */
export const ProcessingJob: React.FC<ProcessingJobProps> = ({ uploadResult, model, jobStatus, jobResult, jobError, onRun, onCancel, onComplete }) => {
  const started = useRef(false);

  useEffect(() => {
    if (!started.current) {
      started.current = true;
      onRun();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 'idle' here means "cancelled" (the effect below only calls onRun once, on mount) -- never a state the analyst asked for, so it gets its own explicit copy rather than falling through to the pending branch's "Running…" text.
  const wasCancelled = jobStatus === 'idle' && started.current;

  const logs: string[] = [];
  if (jobStatus === 'pending' || jobStatus === 'success' || jobStatus === 'error' || wasCancelled) {
    logs.push(`Requested POST /sr/run — model=${model}, upload_id=${uploadResult.upload_id.slice(0, 8)}…`);
  }
  if (jobStatus === 'success' && jobResult) {
    logs.push(`Completed: ${jobResult.model_name} on ${jobResult.metadata.device}, ${jobResult.metadata.inference_seconds}s.`);
    logs.push(`Tiling: ${jobResult.metadata.tiling.tile_grid?.join(' × ') ?? '1 × 1'} tiles, ${jobResult.metadata.tiling.tile_inferences ?? 1} model calls (${jobResult.metadata.tiling.scene_passes ?? 1} TTA passes).`);
    logs.push(`Output: ${jobResult.output_shape.join(' × ')}, ${jobResult.resolution.sr_resolution_m} m pixel grid, CRS ${jobResult.crs ?? '—'}.`);
    logs.push(`Valid-pixel coverage: ${(jobResult.metadata.preprocessing_mask_coverage * 100).toFixed(1)}%.`);
  }
  if (jobStatus === 'error' && jobError) {
    logs.push(`Failed: ${jobError}`);
  }
  if (wasCancelled) {
    logs.push('Cancelled by request before a response arrived.');
  }

  return (
    <div className="w-full min-h-[calc(100vh-3.5rem)] bg-[#0B0E12] text-white flex flex-col justify-center items-center p-6">
      <div className="max-w-2xl w-full bg-[#151A20] border border-[#1D242C] rounded-xl overflow-hidden shadow-2xl">
        <div className="p-6 border-b border-[#1D242C] bg-[#151A20] flex items-center justify-between">
          <div>
            <span className="text-xs font-mono text-[#9AA3AD]">{uploadResult.filename}</span>
            <h2 className="text-lg font-bold text-white tracking-tight mt-0.5">
              {jobStatus === 'success' ? 'Reconstruction complete' : jobStatus === 'error' ? 'Reconstruction failed' : wasCancelled ? 'Request cancelled' : `Running ${MODEL_LABEL[model]}…`}
            </h2>
          </div>
          <span
            className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-mono font-medium ${
              jobStatus === 'success'
                ? 'bg-[#35C98A]/10 text-[#35C98A] border border-[#35C98A]/30'
                : jobStatus === 'error' || wasCancelled
                ? 'bg-[#D94B4B]/10 text-[#D94B4B] border border-[#D94B4B]/30'
                : 'bg-[#D8B84C]/10 text-[#D8B84C] border border-[#D8B84C]/30'
            }`}
          >
            {jobStatus === 'success' ? <CheckCircle2 className="w-3.5 h-3.5" /> : jobStatus === 'error' || wasCancelled ? <AlertTriangle className="w-3.5 h-3.5" /> : <Loader2 className="w-3.5 h-3.5 animate-spin" />}
            <span>{wasCancelled ? 'cancelled' : jobStatus === 'pending' ? 'Running (synchronous request)' : jobStatus}</span>
          </span>
        </div>

        {jobStatus === 'pending' && (
          <div className="px-6 py-4 text-xs text-[#9AA3AD] border-b border-[#1D242C] leading-relaxed">
            One blocking HTTP request: preprocessing, the model over 128 px tiles, the 6-view TTA ensemble, and geospatial export all run inside it.{' '}
            {model === 'mamba' ? 'SEN2SR-Mamba is substantially slower than Lite — this can take up to a minute or more.' : 'SEN2SR-Lite is typically fast.'}
          </div>
        )}

        <div className="p-5 bg-[#0e0e0e] border-b border-[#1D242C]">
          <div className="flex items-center gap-1.5 mb-3 text-xs text-[#9AA3AD] font-mono">
            <Terminal className="w-3.5 h-3.5 text-[#D8B84C]" />
            <span>REQUEST / RESPONSE LOG</span>
          </div>
          <div className="font-mono text-xs space-y-1.5 min-h-[80px]">
            {logs.map((line, i) => (
              <div key={i} className="text-[#9AA3AD] leading-relaxed text-[11px]">{line}</div>
            ))}
            {jobStatus === 'pending' && (
              <div className="flex items-center gap-2 text-[#9AA3AD] text-[11px]">
                <span className="inline-block w-1.5 h-3 bg-[#D8B84C] animate-pulse" />
                <span>Awaiting the response…</span>
              </div>
            )}
          </div>
        </div>

        <div className="p-4 bg-[#151A20] flex items-center justify-between">
          {jobStatus === 'pending' && (
            <button onClick={onCancel} className="px-4 py-2 text-xs font-medium text-[#9AA3AD] hover:text-white bg-[#1D242C] hover:bg-[#343C45] rounded transition-colors cursor-pointer">
              Cancel request
            </button>
          )}
          {(jobStatus === 'error' || wasCancelled) && (
            <button onClick={onRun} className="px-4 py-2 text-xs font-medium text-white bg-[#1D242C] hover:bg-[#343C45] rounded transition-colors cursor-pointer">
              {wasCancelled ? 'Run again' : 'Retry'}
            </button>
          )}
          {jobStatus === 'success' && (
            <button
              onClick={onComplete}
              className="flex items-center gap-2 px-5 py-2.5 text-xs font-bold rounded bg-[#D8B84C] hover:bg-[#E4C96A] text-[#151A20] transition-all cursor-pointer shadow-lg ml-auto"
            >
              <span>Inspect the reconstruction</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
