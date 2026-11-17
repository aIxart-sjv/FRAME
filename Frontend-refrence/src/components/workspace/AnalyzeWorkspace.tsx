import React, { useState } from 'react';
import { Layers, GitCompare, ArrowRight, ShieldAlert, CheckCircle2, AlertTriangle, Calendar, Info } from 'lucide-react';
import { RegionOfInterest, LandCoverClassDistribution } from '../../types';
import { MOCK_LAND_COVER_DISTRIBUTION } from '../../data/mockData';

interface AnalyzeWorkspaceProps {
  currentROI: RegionOfInterest | null;
  onUpdateROI: (roi: RegionOfInterest | null) => void;
  tileId?: string;
}

export const AnalyzeWorkspace: React.FC<AnalyzeWorkspaceProps> = ({
  currentROI,
  onUpdateROI,
  tileId = 'T44RPR',
}) => {
  const [activeTask, setActiveTask] = useState<'land-cover' | 'change-detection'>('land-cover');
  const [compareMode, setCompareMode] = useState<'sr' | 'original' | 'diff'>('sr');

  // Change detection state
  const [dateA, setDateA] = useState<string>('2025-02-14 (Sentinel-2A)');
  const [dateB, setDateB] = useState<string>('2026-02-18 (Sentinel-2B)');
  const [changeType, setChangeType] = useState<'built-up' | 'water' | 'vegetation'>('built-up');

  return (
    <div className="w-full flex-1 flex flex-col bg-[#0B0E12] text-white overflow-y-auto">
      {/* Top Analyze Navigation Bar */}
      <div className="bg-[#151A20] border-b border-[#1D242C] px-6 py-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <p className="text-xs font-mono uppercase tracking-wider text-[#D8B84C] mb-0.5">Downstream Utility Evaluation</p>
          <h1 className="text-xl font-bold tracking-tight text-white">What can we learn from this reconstruction?</h1>
          <p className="text-xs text-[#9AA3AD]">
            Investigating whether super-resolved 2.5m data improves operational geospatial tasks over 10m Sentinel-2.
          </p>
        </div>

        {/* Task Switcher */}
        <div className="flex items-center gap-1 bg-[#0B0E12] p-1 rounded-md border border-[#1D242C]">
          <button
            onClick={() => setActiveTask('land-cover')}
            className={`flex items-center gap-2 px-3 py-1.5 text-xs font-medium rounded transition-colors ${
              activeTask === 'land-cover'
                ? 'bg-[#D8B84C] text-[#0B0E12] shadow-sm'
                : 'text-[#9AA3AD] hover:text-white'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>Land Cover Mapping</span>
          </button>
          <button
            onClick={() => setActiveTask('change-detection')}
            className={`flex items-center gap-2 px-3 py-1.5 text-xs font-medium rounded transition-colors ${
              activeTask === 'change-detection'
                ? 'bg-[#D8B84C] text-[#0B0E12] shadow-sm'
                : 'text-[#9AA3AD] hover:text-white'
            }`}
          >
            <GitCompare className="w-3.5 h-3.5" />
            <span>Multi-Temporal Change</span>
          </button>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="max-w-7xl mx-auto w-full p-6 flex-1 flex flex-col gap-6">
        {/* Task 1: Land Cover Mapping */}
        {activeTask === 'land-cover' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left: Interactive Thematic Map Viewport */}
            <div className="lg:col-span-8 flex flex-col bg-[#151A20] border border-[#1D242C] rounded-lg overflow-hidden">
              {/* Map Header with Compare Controls */}
              <div className="px-4 py-2.5 bg-[#151A20] border-b border-[#1D242C] flex items-center justify-between text-xs">
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-white">Thematic Classification Canvas</span>
                  <span className="text-[#9AA3AD] font-mono text-[11px]">(5 Standard Macro Classes)</span>
                </div>

                <div className="flex items-center gap-1 bg-[#0B0E12] p-0.5 rounded border border-[#1D242C]">
                  <button
                    onClick={() => setCompareMode('original')}
                    className={`px-2.5 py-0.5 text-[11px] font-mono rounded ${
                      compareMode === 'original' ? 'bg-[#1D242C] text-[#D8B84C]' : 'text-[#9AA3AD] hover:text-white'
                    }`}
                  >
                    Native 10m Input
                  </button>
                  <button
                    onClick={() => setCompareMode('sr')}
                    className={`px-2.5 py-0.5 text-[11px] font-mono rounded ${
                      compareMode === 'sr' ? 'bg-[#1D242C] text-[#D8B84C]' : 'text-[#9AA3AD] hover:text-white'
                    }`}
                  >
                    FRAME SR 2.5m
                  </button>
                  <button
                    onClick={() => setCompareMode('diff')}
                    className={`px-2.5 py-0.5 text-[11px] font-mono rounded ${
                      compareMode === 'diff' ? 'bg-[#1D242C] text-[#D8B84C]' : 'text-[#9AA3AD] hover:text-white'
                    }`}
                  >
                    Boundary Delta
                  </button>
                </div>
              </div>

              {/* Map SVG Stage */}
              <div className="relative flex-1 min-h-[460px] bg-[#0c1017] p-4 flex items-center justify-center">
                <svg viewBox="0 0 800 500" className="w-full h-full max-h-[500px]">
                  {/* Coarse vs Refined Thematic Raster Representation */}
                  {compareMode === 'original' ? (
                    <g id="landcover-10m-coarse">
                      {/* 10m Pixelated block shapes */}
                      <rect x="50" y="300" width="300" height="200" fill="#3182CE" />
                      <rect x="200" y="150" width="180" height="150" fill="#E05252" />
                      <rect x="380" y="120" width="220" height="180" fill="#E05252" />
                      <rect x="100" y="50" width="300" height="100" fill="#35C98A" />
                      <rect x="420" y="40" width="280" height="100" fill="#D69E2E" />
                      <rect x="620" y="240" width="150" height="180" fill="#A0AEC0" />
                      {/* Grid overlay */}
                      <g stroke="rgba(255,255,255,0.08)" strokeWidth="0.5">
                        {Array.from({ length: 40 }).map((_, i) => (
                          <line key={`x-${i}`} x1={i * 20} y1="0" x2={i * 20} y2="500" />
                        ))}
                        {Array.from({ length: 25 }).map((_, i) => (
                          <line key={`y-${i}`} x1="0" y1={i * 20} x2="800" y2={i * 20} />
                        ))}
                      </g>
                    </g>
                  ) : (
                    <g id="landcover-25m-refined">
                      {/* 2.5m Smooth, delineated sub-pixel classification */}
                      {/* Water Body */}
                      <path
                        d="M 50,500 Q 200,380 260,300 T 450,180 T 650,60 L 800,0 L 800,500 Z"
                        fill="#3182CE"
                        stroke="#2b6cb0"
                        strokeWidth="1.2"
                      />

                      {/* Built-up polygons (discrete building clusters) */}
                      <g fill="#E05252" stroke="#c53030" strokeWidth="0.8">
                        <rect x="220" y="180" width="40" height="30" rx="1" />
                        <rect x="270" y="180" width="35" height="28" rx="1" />
                        <rect x="315" y="178" width="42" height="32" rx="1" />
                        <rect x="225" y="220" width="50" height="35" rx="1" />
                        <rect x="285" y="222" width="60" height="40" rx="1" />
                        <rect x="390" y="140" width="50" height="45" rx="1" />
                        <rect x="450" y="142" width="45" height="38" rx="1" />
                        <rect x="505" y="140" width="55" height="42" rx="1" />
                        <rect x="400" y="200" width="60" height="50" rx="1" />
                        <rect x="470" y="205" width="55" height="48" rx="1" />
                      </g>

                      {/* Dense Vegetation Canopy */}
                      <polygon
                        points="100,50 380,50 340,160 80,160"
                        fill="#35C98A"
                        stroke="#276749"
                        strokeWidth="1"
                      />

                      {/* Agricultural Cropland parcels */}
                      <polygon
                        points="420,40 750,20 700,160 380,160"
                        fill="#D69E2E"
                        stroke="#b7791f"
                        strokeWidth="1"
                      />

                      {/* Bare Soil corridor */}
                      <polygon
                        points="620,240 780,210 760,420 590,440"
                        fill="#A0AEC0"
                        stroke="#718096"
                        strokeWidth="1"
                      />

                      {/* Highlighted Boundary Discrepancies if diff mode */}
                      {compareMode === 'diff' && (
                        <g>
                          <rect x="200" y="160" width="180" height="15" fill="#D8B84C" opacity="0.8" />
                          <rect x="360" y="120" width="15" height="150" fill="#D8B84C" opacity="0.8" />
                          <circle cx="280" cy="300" r="14" fill="#D8B84C" opacity="0.7" />
                        </g>
                      )}
                    </g>
                  )}

                  {/* Persistent ROI Overlay */}
                  {currentROI && (
                    <rect
                      x="180"
                      y="120"
                      width="380"
                      height="220"
                      fill="none"
                      stroke="#D8B84C"
                      strokeWidth="2"
                      strokeDasharray="4 2"
                    />
                  )}
                </svg>

                {/* HUD Readout */}
                <div className="absolute bottom-3 left-3 bg-[#0B0E12]/90 border border-[#1D242C] px-3 py-1.5 rounded text-[11px] font-mono text-[#9AA3AD]">
                  RESOLVED GRID: {compareMode === 'original' ? '10.0 m / px' : '2.500 m / px'} · METHOD: RANDOM FOREST + LOW-FREQ CONSTRAINED INPUT
                </div>
              </div>

              {/* Legend Strip */}
              <div className="p-3 bg-[#151A20] border-t border-[#1D242C] flex flex-wrap items-center justify-between text-xs">
                <div className="flex items-center gap-4">
                  {MOCK_LAND_COVER_DISTRIBUTION.map((c) => (
                    <div key={c.className} className="flex items-center gap-1.5">
                      <span className="w-3 h-3 rounded-xs" style={{ backgroundColor: c.color }} />
                      <span className="text-[#9AA3AD] text-[11px]">{c.className}</span>
                    </div>
                  ))}
                </div>
                {compareMode === 'diff' && (
                  <span className="text-[11px] font-mono text-[#D8B84C]">
                    ★ Yellow indicates boundary cells reallocated due to 2.5m sub-pixel unmixing
                  </span>
                )}
              </div>
            </div>

            {/* Right: Area Distribution & Boundary Precision Review Panel */}
            <div className="lg:col-span-4 flex flex-col gap-4">
              {/* ROI Scope Card */}
              <div className="p-4 bg-[#151A20] border border-[#1D242C] rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-mono uppercase text-[#9AA3AD]">Analysis Target Scope</span>
                  <span className="text-xs font-mono text-[#D8B84C] font-bold">
                    {currentROI ? currentROI.name : 'Full AOI Scope'}
                  </span>
                </div>
                <div className="text-xs text-[#9AA3AD] space-y-1">
                  <div className="flex justify-between">
                    <span>Evaluated Surface Area:</span>
                    <span className="font-mono text-white">{currentROI ? `${currentROI.areaHectares} ha` : '2,348.5 ha'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Inference Model:</span>
                    <span className="font-mono text-white">SEN2SR-Mamba (2.5m)</span>
                  </div>
                </div>
              </div>

              {/* Class Area Comparison Table */}
              <div className="p-4 bg-[#151A20] border border-[#1D242C] rounded-lg space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-semibold text-white uppercase tracking-wider">Surface Area Metrics</h3>
                  <span className="text-[10px] text-[#9AA3AD] font-mono">10m vs 2.5m</span>
                </div>

                <div className="space-y-2 text-xs">
                  {MOCK_LAND_COVER_DISTRIBUTION.map((item) => (
                    <div key={item.className} className="p-2.5 bg-[#151A20] rounded border border-[#1D242C]">
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-medium text-white flex items-center gap-1.5">
                          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: item.color }} />
                          {item.className}
                        </span>
                        <span
                          className={`text-[11px] font-mono ${
                            item.deltaPercent > 0 ? 'text-[#D8B84C]' : 'text-[#9AA3AD]'
                          }`}
                        >
                          {item.deltaPercent > 0 ? `+${item.deltaPercent}%` : `${item.deltaPercent}%`}
                        </span>
                      </div>

                      <div className="grid grid-cols-2 text-[11px] font-mono text-[#9AA3AD]">
                        <span>Native: {item.original10mAreaHa} ha</span>
                        <span className="text-right text-[#9AA3AD]">SR: {item.sr25mAreaHa} ha</span>
                      </div>

                      <div className="mt-1.5 flex items-center justify-between text-[10px] pt-1 border-t border-[#1D242C]">
                        <span className="text-[#9AA3AD]">Boundary Confidence:</span>
                        <span
                          className={`font-semibold ${
                            item.boundaryConfidence === 'High'
                              ? 'text-[#35C98A]'
                              : item.boundaryConfidence === 'Moderate'
                              ? 'text-[#D8B84C]'
                              : 'text-[#D94B4B]'
                          }`}
                        >
                          {item.boundaryConfidence}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Analytical Insight */}
                <div className="p-3 bg-[#0B0E12] rounded border border-[#1D242C] text-[11px] text-[#9AA3AD] leading-relaxed flex items-start gap-2">
                  <Info className="w-4 h-4 text-[#D8B84C] shrink-0 mt-0.5" />
                  <p>
                    Reconstructed 2.5m imagery resolves narrow transport corridors (+3.8% Built-up) previously aggregated into mixed roadside vegetation cells at 10m native resolution.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Task 2: Multi-Temporal Change Detection */}
        {activeTask === 'change-detection' && (
          <div className="space-y-6">
            {/* Dual Scene Selector */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-4 bg-[#151A20] border border-[#1D242C] rounded-lg">
              <div>
                <label className="text-xs font-mono uppercase text-[#9AA3AD] block mb-1">Baseline Date A</label>
                <div className="p-3 bg-[#151A20] rounded border border-[#1D242C] flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Calendar className="w-4 h-4 text-[#D8B84C]" />
                    <span className="text-xs font-semibold text-white">{dateA}</span>
                  </div>
                  <span className="text-[11px] font-mono text-[#9AA3AD]">L2A Baseline</span>
                </div>
              </div>

              <div>
                <label className="text-xs font-mono uppercase text-[#9AA3AD] block mb-1">Target Date B</label>
                <div className="p-3 bg-[#151A20] rounded border border-[#1D242C] flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Calendar className="w-4 h-4 text-[#D8B84C]" />
                    <span className="text-xs font-semibold text-white">{dateB}</span>
                  </div>
                  <span className="text-[11px] font-mono text-[#9AA3AD]">L2A Target (+369 Days)</span>
                </div>
              </div>
            </div>

            {/* Change Inspection Stages (Earlier -> Change -> Later) */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Earlier Date A */}
              <div className="bg-[#151A20] border border-[#1D242C] rounded-lg overflow-hidden flex flex-col">
                <div className="p-3 bg-[#151A20] border-b border-[#1D242C] text-xs font-mono text-[#9AA3AD] flex justify-between">
                  <span>BASELINE: 2025-02-14</span>
                  <span className="text-[#9AA3AD]">SR 2.5m</span>
                </div>
                <div className="h-52 bg-[#0c1017] flex items-center justify-center p-3 relative">
                  <svg viewBox="0 0 300 200" className="w-full h-full">
                    {/* Rural agricultural field */}
                    <polygon points="40,20 260,20 240,180 20,180" fill="#234230" stroke="#15803d" strokeWidth="1" />
                    <line x1="20" y1="180" x2="280" y2="180" stroke="#71717a" strokeWidth="4" />
                  </svg>
                  <span className="absolute bottom-2 left-2 text-[10px] font-mono bg-[#0B0E12]/80 px-2 py-0.5 rounded text-[#9AA3AD]">
                    Agricultural Open Land
                  </span>
                </div>
              </div>

              {/* Reconstructed Change (Delta) */}
              <div className="bg-[#151A20] border border-[#D8B84C]/40 rounded-lg overflow-hidden flex flex-col">
                <div className="p-3 bg-[#151A20] border-b border-[#1D242C] text-xs font-mono text-[#D8B84C] flex justify-between font-semibold">
                  <span>INTERPRETED CHANGE</span>
                  <span>NEW BUILT-UP (+4.2 ha)</span>
                </div>
                <div className="h-52 bg-[#0c1017] flex items-center justify-center p-3 relative">
                  <svg viewBox="0 0 300 200" className="w-full h-full">
                    {/* Unchanged background */}
                    <polygon points="40,20 260,20 240,180 20,180" fill="#141a16" />
                    {/* New built footprint highlighted in Signal Yellow */}
                    <rect x="80" y="70" width="70" height="50" fill="#D8B84C" stroke="#ffffff" strokeWidth="1" />
                    <rect x="160" y="75" width="60" height="45" fill="#D8B84C" stroke="#ffffff" strokeWidth="1" />
                    <line x1="80" y1="125" x2="220" y2="125" stroke="#D8B84C" strokeWidth="2" strokeDasharray="3 2" />
                  </svg>
                  <span className="absolute bottom-2 left-2 text-[10px] font-mono bg-[#0B0E12]/80 px-2 py-0.5 rounded text-[#D8B84C]">
                    Conversion: Agro → Commercial Facility
                  </span>
                </div>
              </div>

              {/* Later Date B */}
              <div className="bg-[#151A20] border border-[#1D242C] rounded-lg overflow-hidden flex flex-col">
                <div className="p-3 bg-[#151A20] border-b border-[#1D242C] text-xs font-mono text-[#9AA3AD] flex justify-between">
                  <span>TARGET: 2026-02-18</span>
                  <span className="text-[#9AA3AD]">SR 2.5m</span>
                </div>
                <div className="h-52 bg-[#0c1017] flex items-center justify-center p-3 relative">
                  <svg viewBox="0 0 300 200" className="w-full h-full">
                    <polygon points="40,20 260,20 240,180 20,180" fill="#234230" stroke="#15803d" strokeWidth="1" />
                    {/* Developed facility */}
                    <rect x="80" y="70" width="70" height="50" fill="#57483d" stroke="#854d0e" strokeWidth="1" />
                    <rect x="160" y="75" width="60" height="45" fill="#57483d" stroke="#854d0e" strokeWidth="1" />
                    <line x1="20" y1="180" x2="280" y2="180" stroke="#71717a" strokeWidth="4" />
                  </svg>
                  <span className="absolute bottom-2 left-2 text-[10px] font-mono bg-[#0B0E12]/80 px-2 py-0.5 rounded text-[#9AA3AD]">
                    Commercial Parcel Constructed
                  </span>
                </div>
              </div>
            </div>

            {/* Scientific Distinction Warning Banner */}
            {/* Requirement: Distinguish observed image difference from interpreted change! */}
            <div className="p-4 bg-[#151A20] border border-[#343C45] rounded-lg flex items-start gap-3">
              <ShieldAlert className="w-5 h-5 text-[#D8B84C] shrink-0 mt-0.5" />
              <div className="text-xs text-[#9AA3AD]">
                <h4 className="font-semibold text-white mb-1">
                  Scientific Integrity Guardrail: Difference vs. Verified Physical Change
                </h4>
                <p className="leading-relaxed">
                  Spectral difference between dates can occur due to seasonal soil moisture, phenology, and viewing sun-angles. Detected change polygons have undergone NDBI index thresholding and SCL shadow elimination to isolate physical built-up structural expansions.
                </p>
                <div className="mt-2 flex items-center gap-4 text-[11px] font-mono text-[#9AA3AD]">
                  <span>Change IoU: 0.84</span>
                  <span>·</span>
                  <span>Boundary Confidence: 91.2%</span>
                  <span>·</span>
                  <span className="text-[#35C98A]">False Growth Filter: Active</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
