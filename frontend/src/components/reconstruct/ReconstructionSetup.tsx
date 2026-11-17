/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { ArrowLeft, ArrowRight, Info, Sliders } from 'lucide-react';
import type { HealthResponse, ModelId, UploadResponse } from '../../api/types';

interface ReconstructionSetupProps {
  uploadResult: UploadResponse;
  health: HealthResponse | null;
  model: ModelId;
  onChangeModel: (model: ModelId) => void;
  onStartProcessing: () => void;
  onBack: () => void;
}

const MODEL_OPTIONS: Array<{ id: ModelId; name: string; spec: string }> = [
  { id: 'lite', name: 'SEN2SR-Lite', spec: 'A lightweight CNN baseline. Runs on CPU or GPU; typically well under a second per tile.' },
  { id: 'mamba', name: 'SEN2SR-Mamba', spec: 'A larger state-space model. Requires a CUDA GPU on the server; roughly two orders of magnitude slower per tile than Lite.' },
];

/**
 * Step 2 of 2. Every control here maps to something the backend actually does with it: the model choice is the
 * `model` field of `POST /sr/run`; everything else in the pipeline (128 px tiles, the server's configured overlap,
 * the always-on 6-view TTA ensemble) is fixed and reported here as fact, not offered as a fake per-request setting
 * the backend has no way to honour.
 */
export const ReconstructionSetup: React.FC<ReconstructionSetupProps> = ({ uploadResult, health, model, onChangeModel, onStartProcessing, onBack }) => {
  const availability = new Map((health?.available_models ?? []).map((m) => [m.id, m]));

  return (
    <div className="w-full min-h-[calc(100vh-3.5rem)] bg-[#0B0E12] text-white flex flex-col">
      <div className="border-b border-[#1D242C] px-6 py-6 bg-[#151A20]">
        <div className="max-w-4xl mx-auto flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs font-mono uppercase tracking-wider text-[#D8B84C]">Step 2 of 2</span>
              <span className="text-[#343C45]">·</span>
              <span className="text-xs text-[#9AA3AD]">{uploadResult.filename}</span>
            </div>
            <h1 className="text-2xl font-bold tracking-tight text-white">Choose a model</h1>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={onBack}
              className="flex items-center gap-1.5 px-4 py-2 text-xs font-medium text-[#9AA3AD] hover:text-white bg-[#1D242C] hover:bg-[#343C45] rounded transition-colors cursor-pointer"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>Back</span>
            </button>
            <button
              onClick={onStartProcessing}
              className="flex items-center gap-2 px-6 py-2.5 text-xs font-bold rounded transition-all cursor-pointer bg-[#D8B84C] hover:bg-[#E4C96A] text-[#151A20] shadow-md"
            >
              <span>Run reconstruction</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>

      <div className="max-w-4xl mx-auto w-full flex-1 p-6 space-y-6">
        {/* Summary */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="p-4 bg-[#151A20] border border-[#1D242C] rounded-lg">
            <span className="text-[11px] font-mono text-[#9AA3AD] uppercase block mb-1">Input</span>
            <p className="text-sm font-semibold text-white">{uploadResult.width} × {uploadResult.height} px</p>
            <span className="text-[11px] text-[#9AA3AD] font-mono">{uploadResult.resolution_m ?? 10} m native grid</span>
          </div>
          <div className="p-4 bg-[#151A20] border border-[#1D242C] rounded-lg">
            <span className="text-[11px] font-mono text-[#9AA3AD] uppercase block mb-1">Bands</span>
            <p className="text-sm font-semibold text-white">{uploadResult.band_names.join(' · ')}</p>
            <span className="text-[11px] text-[#9AA3AD]">4 of the 12 Sentinel-2 L2A analytical bands</span>
          </div>
          <div className="p-4 bg-[#151A20] border border-[#D8B84C]/40 rounded-lg">
            <span className="text-[11px] font-mono text-[#D8B84C] uppercase block mb-1">Output</span>
            <p className="text-sm font-semibold text-white">{Math.round((uploadResult.resolution_m ?? 10) / 4 * 100) / 100} m pixel grid</p>
            <span className="text-[11px] text-[#9AA3AD]">4× scale factor, SR-derived — not a native observation</span>
          </div>
        </div>

        {/* Model selection */}
        <div className="p-5 bg-[#151A20] border border-[#1D242C] rounded-lg space-y-3">
          <div className="flex items-center gap-2">
            <Sliders className="w-4 h-4 text-[#D8B84C]" />
            <h3 className="text-sm font-semibold text-white">Super-resolution model</h3>
          </div>
          {MODEL_OPTIONS.map((opt) => {
            const status = availability.get(opt.id);
            const disabled = status ? !status.available : false;
            return (
              <div
                key={opt.id}
                onClick={() => !disabled && onChangeModel(opt.id)}
                className={`p-3 rounded border transition-all text-left ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'} ${
                  model === opt.id ? 'bg-[#181d26] border-[#D8B84C] ring-1 ring-[#D8B84C]/40' : 'bg-[#151A20] border-[#1D242C] hover:border-[#343C45]'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-white">{opt.name}</span>
                  <input type="radio" checked={model === opt.id} disabled={disabled} onChange={() => onChangeModel(opt.id)} className="accent-[#D8B84C]" />
                </div>
                <p className="text-[11px] text-[#9AA3AD] mt-1 leading-normal">{opt.spec}</p>
                {disabled && status?.reason && (
                  <p className="text-[11px] text-[#D94B4B] mt-1">Unavailable on this server: {status.reason}</p>
                )}
              </div>
            );
          })}
          {!health && <p className="text-[11px] text-[#9AA3AD]">Checking model availability…</p>}
        </div>

        {/* Fixed pipeline facts */}
        <div className="p-5 bg-[#151A20] border border-[#1D242C] rounded-lg space-y-2">
          <div className="flex items-center gap-2">
            <Info className="w-4 h-4 text-[#D8B84C]" />
            <h3 className="text-sm font-semibold text-white">Fixed by the server, not configurable per request</h3>
          </div>
          <ul className="text-[11px] text-[#9AA3AD] space-y-1 list-disc pl-4">
            <li>128 × 128 px tiles, with reflect padding and linear-blended overlap for scenes larger than one tile.</li>
            <li>A 6-view test-time-augmentation ensemble always runs, producing the mean reconstruction and the stability diagnostic together.</li>
            <li>The Fourier low-frequency hard constraint from the upstream SEN2SR models is always applied; it is not a FRAME setting.</li>
          </ul>
        </div>
      </div>
    </div>
  );
};
