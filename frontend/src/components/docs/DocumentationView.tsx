/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { ArrowLeft, BookOpen, Cpu, Download, Layers, PieChart, ShieldCheck, Terminal } from 'lucide-react';

interface DocumentationViewProps {
  onBack: () => void;
  onNavigate: (view: string) => void;
}

/**
 * This page describes the ACTUAL backend (frame/api/) — its real endpoints, its real synchronous execution model,
 * and its real RGBN-only scope. Nothing here describes a capability (a scene catalogue, an async job queue,
 * land-cover classification, COG output) that the code does not have.
 */
export const DocumentationView: React.FC<DocumentationViewProps> = ({ onBack, onNavigate }) => {
  const [activeTopic, setActiveTopic] = useState<'getting-started' | 'data' | 'reconstruction' | 'validation' | 'analysis' | 'export' | 'api'>('getting-started');

  return (
    <div className="w-full min-h-[calc(100vh-3.5rem)] bg-[#0B0E12] text-white p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        <button onClick={onBack} className="flex items-center gap-1.5 text-xs text-[#9AA3AD] hover:text-white transition-colors cursor-pointer">
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Back to Application</span>
        </button>

        <div className="pb-4 border-b border-[#1D242C]">
          <h1 className="text-2xl font-bold tracking-tight text-white">FRAME Technical Documentation</h1>
          <p className="text-xs text-[#9AA3AD] mt-0.5">How this application actually works, for SIH26142.</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
          <div className="md:col-span-3 space-y-1">
            {[
              { id: 'getting-started' as const, label: 'Getting Started', icon: <BookOpen className="w-3.5 h-3.5" /> },
              { id: 'data' as const, label: 'Data & Scaling', icon: <Layers className="w-3.5 h-3.5" /> },
              { id: 'reconstruction' as const, label: 'SR Models', icon: <Cpu className="w-3.5 h-3.5" /> },
              { id: 'validation' as const, label: 'Validation', icon: <ShieldCheck className="w-3.5 h-3.5" /> },
              { id: 'analysis' as const, label: 'Downstream Demonstration', icon: <PieChart className="w-3.5 h-3.5" /> },
              { id: 'export' as const, label: 'Output Files', icon: <Download className="w-3.5 h-3.5" /> },
              { id: 'api' as const, label: 'REST API', icon: <Terminal className="w-3.5 h-3.5" /> },
            ].map((topic) => (
              <button
                key={topic.id}
                onClick={() => setActiveTopic(topic.id)}
                className={`w-full flex items-center gap-2.5 px-3 py-2 text-xs font-medium rounded transition-colors text-left cursor-pointer ${
                  activeTopic === topic.id ? 'bg-[#151A20] text-[#D8B84C] border-l-2 border-[#D8B84C]' : 'text-[#9AA3AD] hover:text-white hover:bg-[#151A20]'
                }`}
              >
                {topic.icon}
                <span>{topic.label}</span>
              </button>
            ))}
          </div>

          <div className="md:col-span-9 p-6 bg-[#151A20] border border-[#1D242C] rounded-lg text-xs leading-relaxed space-y-6">
            {activeTopic === 'getting-started' && (
              <div className="space-y-4">
                <h2 className="text-base font-bold text-white">Getting started</h2>
                <p className="text-[#9AA3AD]">
                  FRAME ingests a Sentinel-2 L2A GeoTIFF (the RGBN bands: B04, B03, B02, B08) that you supply, super-resolves it onto a 2.5 m pixel grid, and reports evidence about that reconstruction.
                </p>
                <div className="p-4 bg-[#0B0E12] rounded border border-[#1D242C] space-y-2">
                  <h3 className="font-semibold text-white">The actual workflow:</h3>
                  <ol className="list-decimal pl-5 space-y-1.5 text-[#9AA3AD]">
                    <li><strong>Upload</strong> a GeoTIFF you already have — there is no live-catalogue search or STAC browsing here.</li>
                    <li><strong>Choose a model</strong>, SEN2SR-Lite or SEN2SR-Mamba, and run it. This is one blocking HTTP request; there is no job queue to poll.</li>
                    <li><strong>Inspect</strong> the original input, the SR output, the NDVI demonstration (once run), and the TTA stability diagnostic — all real rasters decoded in your browser.</li>
                    <li><strong>Validate</strong> what is, and is not, evidenced for this specific scene.</li>
                    <li><strong>Download</strong> the SR GeoTIFF, the stability GeoTIFF, and the JSON results directly.</li>
                  </ol>
                </div>
              </div>
            )}

            {activeTopic === 'data' && (
              <div className="space-y-4">
                <h2 className="text-base font-bold text-white">Input scope and scaling</h2>
                <p className="text-[#9AA3AD]">
                  This backend processes exactly four bands — B04, B03, B02, B08 — of the twelve usable Sentinel-2 L2A analytical bands. The other eight (including the red-edge and SWIR bands) are not read.
                </p>
                <div className="p-3 bg-[#0B0E12] border border-[#1D242C] rounded text-[#9AA3AD]">
                  <strong className="text-white block mb-1">Reflectance scaling:</strong>
                  <code>reflectance = digital_number / 10000</code>, applied only when you declare the file as raw digital numbers. The −1000 BOA offset used by processing baseline ≥ 04.00 is <strong className="text-white">not</strong> applied — a known, documented gap, not a silent approximation.
                </div>
                <p className="text-[#9AA3AD]">A scene with no valid pixel, or whose values contradict the declared scale, is refused at upload with a specific error — never silently mis-processed.</p>
              </div>
            )}

            {activeTopic === 'reconstruction' && (
              <div className="space-y-4">
                <h2 className="text-base font-bold text-white">Super-resolution models</h2>
                <p className="text-[#9AA3AD]">
                  Two published SEN2SR models are selectable, unmodified from upstream: <strong className="text-white">SEN2SR-Lite</strong>, a lightweight CNN that runs on CPU or GPU, and <strong className="text-white">SEN2SR-Mamba</strong>, a larger state-space model that
                  requires a CUDA GPU and runs roughly two orders of magnitude slower per tile. Both apply the same low-frequency Fourier hard constraint: the low-frequency content of the output is tied to the bicubic-upsampled input, and only the high-frequency detail is generated.
                </p>
                <div className="p-3 bg-[#0B0E12] border border-[#1D242C] rounded text-[#9AA3AD]">
                  Scenes larger than 128 × 128 are split into overlapping tiles and blended back together server-side; this is transparent to the request. Neither model is fine-tuned or retrained by this application.
                </div>
              </div>
            )}

            {activeTopic === 'validation' && (
              <div className="space-y-4">
                <h2 className="text-base font-bold text-white">Validation</h2>
                <p className="text-[#9AA3AD]">
                  For a live, arbitrary upload, FRAME reports self-consistency with the model's own input and the TTA stability diagnostic — never a fabricated PSNR/SSIM, because no independent high-resolution reference exists for an arbitrary scene. Separately, offline, FRAME
                  measured accuracy against real references (SEN2NEON, OpenSR-Test) and found the stability diagnostic only weakly informative and uncalibrated.
                </p>
                <button onClick={() => onNavigate('validation-docs')} className="text-[#D8B84C] hover:underline block">View the full validation summary →</button>
              </div>
            )}

            {activeTopic === 'analysis' && (
              <div className="space-y-4">
                <h2 className="text-base font-bold text-white">Downstream demonstration</h2>
                <p className="text-[#9AA3AD]">
                  The only downstream computation this backend runs is an NDVI comparison between the SR product and its own low-resolution input, on the one uploaded scene. There is no land-cover classification, no NDBI/NDWI/BSI, and no multi-temporal change detection here — see
                  the Validate tab in the app for what FRAME's own separate, offline research measured about downstream utility.
                </p>
              </div>
            )}

            {activeTopic === 'export' && (
              <div className="space-y-4">
                <h2 className="text-base font-bold text-white">Output files</h2>
                <p className="text-[#9AA3AD]">
                  Outputs are standard float32 GeoTIFFs with an embedded CRS and affine transform (not internally tiled COGs). The SR raster, the 5-band stability raster, and — once run — the three NDVI rasters can each be downloaded individually, alongside the raw JSON result of the request that produced them.
                </p>
              </div>
            )}

            {activeTopic === 'api' && (
              <div className="space-y-4">
                <h2 className="text-base font-bold text-white">REST API (synchronous, single process)</h2>
                <p className="text-[#9AA3AD]">
                  There is no job queue: <code>POST /sr/run</code> blocks until the reconstruction finishes and returns the complete result in one response. Job and analysis records live in memory for the life of the server process.
                </p>
                <div className="p-3 bg-[#0B0E12] font-mono text-[11px] border border-[#1D242C] rounded text-[#9AA3AD] space-y-1">
                  <div>GET  /health — model availability, never starts a model</div>
                  <div>POST /upload — validate a GeoTIFF; does not run SR</div>
                  <div>POST /sr/run — the reconstruction, one blocking request</div>
                  <div>POST /analysis/ndvi — the NDVI demonstration, against an already-completed job</div>
                  <div>GET  /sr/download/{'{job_id}'} · /uncertainty/download/{'{job_id}'} — the GeoTIFFs</div>
                  <div>GET  /analysis/download/{'{analysis_id}'}/{'{native-ndvi|sr-ndvi|ndvi-diff}'} — the NDVI GeoTIFFs</div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
