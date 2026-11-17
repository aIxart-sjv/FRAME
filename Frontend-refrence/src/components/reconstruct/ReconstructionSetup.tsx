import React, { useState } from 'react';
import { ArrowLeft, ArrowRight, Layers, Sliders, ShieldCheck, Info } from 'lucide-react';
import { Sentinel2Scene } from '../../types';
import { SENTINEL2_BANDS } from '../../data/mockData';

interface ReconstructionSetupProps {
  selectedAOI: {
    name: string;
    coordinates: [number, number];
    bbox: [number, number, number, number];
  };
  selectedScene: Sentinel2Scene;
  onStartProcessing: (config: {
    model: 'SEN2SR-Mamba' | 'SEN2SR-Lite' | 'LDSR-S2 (Reference)' | 'Residual-CNN Baseline';
    tileSize: number;
    haloSize: number;
    enableStabilityEvaluation: boolean;
  }) => void;
  onBack: () => void;
}

export const ReconstructionSetup: React.FC<ReconstructionSetupProps> = ({
  selectedAOI,
  selectedScene,
  onStartProcessing,
  onBack,
}) => {
  const [model, setModel] = useState<'SEN2SR-Mamba' | 'SEN2SR-Lite' | 'LDSR-S2 (Reference)' | 'Residual-CNN Baseline'>('SEN2SR-Mamba');
  const [tileSize, setTileSize] = useState<number>(128);
  const [haloSize, setHaloSize] = useState<number>(16);
  const [enableStability, setEnableStability] = useState<boolean>(true);

  // Filter analytical bands (B01-B09, B8A, B11, B12; B10 excluded)
  const analyticalBands = SENTINEL2_BANDS.filter((b) => b.isAnalyticalL2A);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onStartProcessing({
      model,
      tileSize,
      haloSize,
      enableStabilityEvaluation: enableStability,
    });
  };

  return (
    <div className="w-full min-h-[calc(100vh-3.5rem)] bg-[#0B0E12] text-white flex flex-col">
      {/* Top Header */}
      <div className="border-b border-[#1D242C] px-6 py-6 bg-[#151A20]">
        <div className="max-w-5xl mx-auto flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs font-mono uppercase tracking-wider text-[#D8B84C]">Step 3 of 3</span>
              <span className="text-[#343C45]">·</span>
              <span className="text-xs text-[#9AA3AD]">Pipeline Configuration</span>
            </div>
            <h1 className="text-2xl font-bold tracking-tight text-white">Ready to reconstruct</h1>
            <p className="text-xs text-[#9AA3AD] mt-0.5">
              Review acquisition inputs, analytical bands, and deep learning super-resolution parameters.
            </p>
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
              onClick={handleSubmit}
              className="flex items-center gap-2 px-6 py-2.5 text-xs font-semibold rounded transition-all cursor-pointer bg-[#D8B84C] hover:bg-[#E4C96A] text-[#151A20] shadow-md font-bold"
            >
              <span>Reconstruct imagery</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>

      {/* Main Container */}
      <div className="max-w-5xl mx-auto w-full flex-1 p-6 space-y-6">
        {/* Summary Card Grid */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="p-4 bg-[#151A20] border border-[#1D242C] rounded-lg">
            <span className="text-[11px] font-mono text-[#9AA3AD] uppercase block mb-1">Target Area</span>
            <p className="text-sm font-semibold text-white truncate">{selectedAOI.name}</p>
            <span className="text-[11px] text-[#9AA3AD] font-mono">UTM Footprint</span>
          </div>

          <div className="p-4 bg-[#151A20] border border-[#1D242C] rounded-lg">
            <span className="text-[11px] font-mono text-[#9AA3AD] uppercase block mb-1">Acquisition Scene</span>
            <p className="text-sm font-semibold text-white">{selectedScene.tileId} L2A</p>
            <span className="text-[11px] text-[#9AA3AD] font-mono">{selectedScene.acquisitionDatetime.split(' ')[0]}</span>
          </div>

          <div className="p-4 bg-[#151A20] border border-[#1D242C] rounded-lg">
            <span className="text-[11px] font-mono text-[#9AA3AD] uppercase block mb-1">Observation Grid</span>
            <p className="text-sm font-semibold text-white">10m / 20m / 60m</p>
            <span className="text-[11px] text-[#9AA3AD]">Native multiresolution</span>
          </div>

          <div className="p-4 bg-[#151A20] border border-[#D8B84C]/40 rounded-lg">
            <span className="text-[11px] font-mono text-[#D8B84C] uppercase block mb-1">Reconstructed Target</span>
            <p className="text-sm font-semibold text-white">2.5 m Reconstructed Grid</p>
            <span className="text-[11px] text-[#9AA3AD]">4× Scale Factor (NTRO &lt; 4m)</span>
          </div>
        </div>

        {/* 12 Analytical Bands Table */}
        <div className="p-5 bg-[#151A20] border border-[#1D242C] rounded-lg">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-[#D8B84C]" />
              <h3 className="text-sm font-semibold text-white">Analytical Surface-Reflectance Bands (12 Channels)</h3>
            </div>
            <div className="flex items-center gap-1.5 text-[11px] text-[#9AA3AD] bg-[#0B0E12] px-2.5 py-1 rounded border border-[#1D242C]">
              <Info className="w-3.5 h-3.5 text-[#D8B84C]" />
              <span>B10 (Cirrus 1375nm) is excluded from L2A surface reflectance</span>
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2">
            {analyticalBands.map((band) => (
              <div
                key={band.id}
                className="p-2.5 bg-[#151A20] border border-[#1D242C] rounded text-left"
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-mono font-bold text-white">{band.id}</span>
                  <span className="text-[10px] font-mono px-1 py-0.2 bg-[#1D242C] rounded text-[#9AA3AD]">
                    {band.nativeResolution}m
                  </span>
                </div>
                <p className="text-[11px] text-[#9AA3AD] truncate">{band.name}</p>
                <span className="text-[10px] font-mono text-[#9AA3AD]">{band.centralWavelength} nm</span>
              </div>
            ))}
          </div>
        </div>

        {/* Model & Architecture Settings */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Architecture Selection */}
          <div className="p-5 bg-[#151A20] border border-[#1D242C] rounded-lg space-y-4">
            <div className="flex items-center gap-2">
              <Sliders className="w-4 h-4 text-[#D8B84C]" />
              <h3 className="text-sm font-semibold text-white">Super-Resolution Model Architecture</h3>
            </div>

            <div className="space-y-2.5">
              {[
                {
                  id: 'SEN2SR-Mamba' as const,
                  name: 'SEN2SR-Mamba (Primary Production)',
                  spec: 'State-space sequence model with low-frequency Fourier hard constraint. Highest spatial-spectral balance.',
                },
                {
                  id: 'SEN2SR-Lite' as const,
                  name: 'SEN2SR-Lite (High Throughput)',
                  spec: 'Optimized residual convolutional architecture. Fast inference for large regional scenes.',
                },
                {
                  id: 'LDSR-S2 (Reference)' as const,
                  name: 'LDSR-S2 (Latent Diffusion Reference)',
                  spec: 'Conditional latent diffusion model for probabilistic multi-sample uncertainty reference.',
                },
                {
                  id: 'Residual-CNN Baseline' as const,
                  name: 'Residual-CNN Baseline (Control)',
                  spec: 'Deterministic baseline for scientific ablation comparison without state-space attention.',
                },
              ].map((opt) => (
                <div
                  key={opt.id}
                  onClick={() => setModel(opt.id)}
                  className={`p-3 rounded border cursor-pointer transition-all text-left ${
                    model === opt.id
                      ? 'bg-[#181d26] border-[#D8B84C] ring-1 ring-[#D8B84C]/40'
                      : 'bg-[#151A20] border-[#1D242C] hover:border-[#343C45]'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-white">{opt.name}</span>
                    <input
                      type="radio"
                      checked={model === opt.id}
                      onChange={() => setModel(opt.id)}
                      className="accent-[#D8B84C]"
                    />
                  </div>
                  <p className="text-[11px] text-[#9AA3AD] mt-1 leading-normal font-normal">
                    {opt.spec}
                  </p>
                </div>
              ))}
            </div>
          </div>

          {/* Operational Tiling & Stability Options */}
          <div className="p-5 bg-[#151A20] border border-[#1D242C] rounded-lg space-y-4 flex flex-col justify-between">
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-[#D8B84C]" />
                <h3 className="text-sm font-semibold text-white">Inference &amp; Evidence Parameters</h3>
              </div>

              <div>
                <label className="text-xs text-[#9AA3AD] block mb-1 font-medium">Patch Window Tiling</label>
                <select
                  value={tileSize}
                  onChange={(e) => setTileSize(Number(e.target.value))}
                  className="w-full bg-[#0B0E12] border border-[#343C45] rounded px-3 py-2 text-xs text-white focus:outline-none focus:border-[#D8B84C]"
                >
                  <option value={128}>128 × 128 LR Patch → 512 × 512 SR Output (Standard)</option>
                  <option value={64}>64 × 64 LR Patch (Low Memory footprint)</option>
                  <option value={256}>256 × 256 LR Patch (High GPU RAM)</option>
                </select>
                <p className="text-[11px] text-[#9AA3AD] mt-1">
                  Enforces continuous windowed inference matching trained point spread function.
                </p>
              </div>

              <div>
                <label className="text-xs text-[#9AA3AD] block mb-1 font-medium">Overlap Halo / Seam Blending</label>
                <select
                  value={haloSize}
                  onChange={(e) => setHaloSize(Number(e.target.value))}
                  className="w-full bg-[#0B0E12] border border-[#343C45] rounded px-3 py-2 text-xs text-white focus:outline-none focus:border-[#D8B84C]"
                >
                  <option value={16}>16 px Weighted Linear Distance Blend (Recommended)</option>
                  <option value={8}>8 px Minimal Overlap</option>
                  <option value={32}>32 px Conservative Overlap (Eliminates border seam artefacts)</option>
                </select>
              </div>

              <div className="pt-2 border-t border-[#1D242C]">
                <label className="flex items-start gap-2.5 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={enableStability}
                    onChange={(e) => setEnableStability(e.target.checked)}
                    className="accent-[#D8B84C] mt-0.5"
                  />
                  <div>
                    <span className="text-xs font-semibold text-white block">
                      Generate 6-Pass Test-Time Transformation (TTA) Stability Raster
                    </span>
                    <span className="text-[11px] text-[#9AA3AD] leading-normal block mt-0.5">
                      Executes inference under geometric perturbations (horizontal flip, vertical flip, shift) to compute pixel-level variance evidence.
                    </span>
                  </div>
                </label>
              </div>
            </div>

            {/* Launch CTA */}
            <div className="pt-4 border-t border-[#1D242C]">
              <button
                onClick={handleSubmit}
                className="w-full py-2.5 bg-[#D8B84C] hover:bg-[#E4C96A] text-[#151A20] font-bold text-xs rounded transition-all flex items-center justify-center gap-2 cursor-pointer shadow-md"
              >
                <span>Initialize Asynchronous Reconstruction Job</span>
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
