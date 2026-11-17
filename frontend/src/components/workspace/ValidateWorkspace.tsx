/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { Activity, Compass, Globe, HelpCircle, Layers, Ruler, ShieldCheck, Sparkles } from 'lucide-react';
import type { SRResultResponse } from '../../api/types';
import type { ValidationSection } from '../../types';

interface ValidateWorkspaceProps {
  jobResult: SRResultResponse;
}

const SECTIONS: Array<{ id: ValidationSection; title: string; subtitle: string; icon: React.ReactNode }> = [
  { id: 'spectral', title: 'Spectral Fidelity', subtitle: 'Self-consistency with the input', icon: <Layers className="w-4 h-4" /> },
  { id: 'spatial', title: 'Spatial Alignment', subtitle: 'CRS & affine grid, as written', icon: <Ruler className="w-4 h-4" /> },
  { id: 'detail', title: 'Detail & Correctness', subtitle: 'Needs an independent reference', icon: <Sparkles className="w-4 h-4" /> },
  { id: 'stability', title: 'Model Stability', subtitle: 'The real TTA diagnostic', icon: <Activity className="w-4 h-4" /> },
  { id: 'reference', title: 'Reference Evaluation', subtitle: 'Why none exists for this scene', icon: <Compass className="w-4 h-4" /> },
  { id: 'generalisation', title: 'What FRAME Has Measured', subtitle: "Separately, offline — not this scene", icon: <Globe className="w-4 h-4" /> },
];

function fmt(value: number | null | undefined, suffix = '', digits = 4): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return `${value.toFixed(digits)}${suffix}`;
}

/**
 * Every number on this screen comes from the real `SRResultResponse` of the current reconstruction. Where the
 * backend does not compute something for a live, arbitrary upload — PSNR/SSIM/SAM/ERGAS, hallucination/omission
 * rates, an independent reference, Indian evaluation — this says so plainly instead of showing a number, mirroring
 * exactly what `frame/api` itself refuses to fabricate.
 */
export const ValidateWorkspace: React.FC<ValidateWorkspaceProps> = ({ jobResult }) => {
  const [activeSection, setActiveSection] = useState<ValidationSection>('spectral');
  const sc = jobResult.self_consistency;
  const geo = jobResult.metadata.output_geospatial;
  const unc = jobResult.uncertainty;

  return (
    <div className="w-full flex-1 flex flex-col bg-[#0B0E12] text-white overflow-y-auto">
      <div className="bg-[#151A20] border-b border-[#1D242C] px-6 py-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <p className="text-xs font-mono uppercase tracking-wider text-[#D8B84C] mb-0.5">Validation</p>
          <h1 className="text-xl font-bold tracking-tight text-white">What is, and is not, evidenced for this reconstruction</h1>
        </div>
        <span className="font-mono text-[#9AA3AD] bg-[#1D242C] border border-[#343C45] px-2.5 py-1 rounded flex items-center gap-1.5 text-xs">
          <HelpCircle className="w-3.5 h-3.5 text-[#9AA3AD]" />
          <span>No independent HR reference for this scene</span>
        </span>
      </div>

      <div className="max-w-6xl mx-auto w-full p-6 flex-1 flex flex-col gap-6">
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
          {SECTIONS.map((sec) => (
            <button
              key={sec.id}
              onClick={() => setActiveSection(sec.id)}
              className={`p-3 rounded-lg border text-left transition-all cursor-pointer ${
                activeSection === sec.id ? 'bg-[#151A20] border-[#D8B84C] ring-1 ring-[#D8B84C]/50' : 'bg-[#151A20] border-[#1D242C] hover:border-[#343C45]'
              }`}
            >
              <div className="flex items-center gap-2 mb-1">
                <span className={activeSection === sec.id ? 'text-[#D8B84C]' : 'text-[#9AA3AD]'}>{sec.icon}</span>
                <span className="text-xs font-semibold text-white">{sec.title}</span>
              </div>
              <p className="text-[10px] text-[#9AA3AD] truncate">{sec.subtitle}</p>
            </button>
          ))}
        </div>

        {activeSection === 'spectral' && (
          <div className="p-5 bg-[#151A20] border border-[#1D242C] rounded-lg space-y-4">
            <h3 className="text-sm font-semibold text-white">Self-consistency with the model's own input</h3>
            <p className="text-xs text-[#9AA3AD]">{sc.note}</p>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div className="p-3.5 bg-[#0B0E12] rounded border border-[#1D242C]">
                <span className="text-[11px] font-mono text-[#9AA3AD] block mb-1">Downsample RMSE</span>
                <div className="text-xl font-bold font-mono text-white">{fmt(sc.downsample_rmse, '', 5)}</div>
              </div>
              <div className="p-3.5 bg-[#0B0E12] rounded border border-[#1D242C]">
                <span className="text-[11px] font-mono text-[#9AA3AD] block mb-1">NDVI discrepancy (mean abs.)</span>
                <div className="text-xl font-bold font-mono text-white">{fmt(sc.ndvi_discrepancy_mean_abs, '', 5)}</div>
              </div>
              <div className="p-3.5 bg-[#0B0E12] rounded border border-[#1D242C]">
                <span className="text-[11px] font-mono text-[#9AA3AD] block mb-1">B08/B04 ratio discrepancy</span>
                <div className="text-xl font-bold font-mono text-white">{fmt(sc.b08_b04_ratio_discrepancy_mean_abs, '', 5)}</div>
              </div>
            </div>
            <div className="p-4 bg-[#0B0E12] rounded border border-[#343C45] text-xs text-[#9AA3AD]">
              PSNR, SSIM, SAM and ERGAS need an independent reference and are not shown here — see "Reference Evaluation".
            </div>
          </div>
        )}

        {activeSection === 'spatial' && (
          <div className="p-5 bg-[#151A20] border border-[#1D242C] rounded-lg space-y-4">
            <h3 className="text-sm font-semibold text-white">Geospatial output, as actually written</h3>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div className="p-4 bg-[#0B0E12] rounded border border-[#1D242C]">
                <span className="text-[11px] font-mono text-[#9AA3AD] block mb-1">CRS</span>
                <div className="text-base font-bold font-mono text-white">{geo.crs ?? '—'}</div>
              </div>
              <div className="p-4 bg-[#0B0E12] rounded border border-[#1D242C]">
                <span className="text-[11px] font-mono text-[#9AA3AD] block mb-1">Output dimensions</span>
                <div className="text-base font-bold font-mono text-white">{geo.width} × {geo.height} px</div>
              </div>
              <div className="p-4 bg-[#0B0E12] rounded border border-[#1D242C]">
                <span className="text-[11px] font-mono text-[#9AA3AD] block mb-1">Pixel size</span>
                <div className="text-base font-bold font-mono text-white">{jobResult.resolution.sr_resolution_m} m</div>
              </div>
            </div>
            <div className="p-4 bg-[#0B0E12] rounded border border-[#1D242C] text-xs font-mono text-[#9AA3AD] space-y-1">
              <div className="text-white font-semibold mb-1">Affine transform (as returned):</div>
              <div className="break-all">{geo.transform.map((v) => v.toFixed(4)).join(', ')}</div>
              <div className="text-white font-semibold mt-2 mb-1">Bounds:</div>
              <div className="break-all">{geo.bounds.map((v) => v.toFixed(2)).join(', ')}</div>
            </div>
            <p className="text-[11px] text-[#9AA3AD]">
              An affine-residual or boundary-displacement figure needs an independently registered reference to compare against; none exists for this upload.
            </p>
          </div>
        )}

        {activeSection === 'detail' && (
          <div className="p-8 text-center bg-[#151A20] border border-[#1D242C] rounded-lg">
            <HelpCircle className="w-8 h-8 text-[#9AA3AD] mx-auto mb-2" />
            <h4 className="text-xs font-semibold text-white uppercase tracking-wider mb-1">Not available for this reconstruction</h4>
            <p className="text-xs text-[#9AA3AD] max-w-md mx-auto leading-relaxed">
              Decomposing added detail into synthesis, omission, and hallucination needs an independent higher-resolution reference. There is none for an arbitrary uploaded scene. FRAME's own offline research measured
              this on curated benchmark datasets (SEN2NEON, OpenSR-Test) — see "What FRAME Has Measured".
            </p>
          </div>
        )}

        {activeSection === 'stability' && (
          <div className="p-5 bg-[#151A20] border border-[#1D242C] rounded-lg space-y-4">
            <h3 className="text-sm font-semibold text-white">{unc.label}</h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div className="p-3 bg-[#0B0E12] rounded border border-[#1D242C]">
                <span className="text-[11px] text-[#9AA3AD] block">Scalar summary</span>
                <span className="text-lg font-mono font-bold text-white">{unc.scalar_summary.toExponential(3)}</span>
              </div>
              <div className="p-3 bg-[#0B0E12] rounded border border-[#1D242C]">
                <span className="text-[11px] text-[#9AA3AD] block">Mean</span>
                <span className="text-lg font-mono font-bold text-white">{unc.overall_distribution.mean.toExponential(3)}</span>
              </div>
              <div className="p-3 bg-[#0B0E12] rounded border border-[#1D242C]">
                <span className="text-[11px] text-[#9AA3AD] block">P95</span>
                <span className="text-lg font-mono font-bold text-white">{unc.overall_distribution.p95.toExponential(3)}</span>
              </div>
              <div className="p-3 bg-[#0B0E12] rounded border border-[#1D242C]">
                <span className="text-[11px] text-[#9AA3AD] block">Max</span>
                <span className="text-lg font-mono font-bold text-white">{unc.overall_distribution.max.toExponential(3)}</span>
              </div>
            </div>
            <div className="text-[11px] text-[#9AA3AD] flex flex-wrap gap-3 font-mono">
              <span>Ensemble size: {unc.n}</span>
              <span>·</span>
              <span>Seed: {unc.seed}</span>
              <span>·</span>
              <span>Transforms: {unc.transform_names.join(', ')}</span>
            </div>
            <div className="p-4 bg-[#0B0E12] rounded border border-[#1D242C] text-xs text-[#9AA3AD] space-y-2">
              <div className="flex items-center gap-2 text-white font-medium">
                <ShieldCheck className="w-4 h-4 text-[#D8B84C]" />
                <span>What this diagnostic is, verbatim from the backend</span>
              </div>
              <p className="leading-relaxed">{unc.disclaimer}</p>
            </div>
          </div>
        )}

        {activeSection === 'reference' && (
          <div className="p-8 text-center bg-[#151A20] border border-[#1D242C] rounded-lg">
            <HelpCircle className="w-8 h-8 text-[#9AA3AD] mx-auto mb-2" />
            <h4 className="text-xs font-semibold text-white uppercase tracking-wider mb-1">No Independent High-Resolution Reference Attached</h4>
            <p className="text-xs text-[#9AA3AD] max-w-md mx-auto mb-4 leading-relaxed">
              This reconstruction runs on an arbitrary uploaded scene, without a simultaneous airborne or sub-metre satellite truth. In accordance with FRAME's scientific-integrity rules, PSNR and SSIM are withheld
              rather than fabricated.
            </p>
            <span className="text-[11px] font-mono text-[#D8B84C]">Only self-consistency and model stability are evaluated for this scene.</span>
          </div>
        )}

        {activeSection === 'generalisation' && (
          <div className="p-5 bg-[#151A20] border border-[#1D242C] rounded-lg space-y-4">
            <h3 className="text-sm font-semibold text-white">What FRAME's own offline research measured — not this scene</h3>
            <p className="text-xs text-[#9AA3AD] leading-relaxed">
              Separately from any live reconstruction, FRAME's own research evaluated SEN2SR-Lite and SEN2SR-Mamba against independent references, and tested whether the stability diagnostic informs about error, on real
              data with a registration-eligibility gate:
            </p>
            <ul className="text-[11px] text-[#9AA3AD] space-y-2 list-disc pl-4 leading-relaxed">
              <li>North America (SEN2NEON, 12 of 30 sampled tiles passed the registration gate) and Spain / SPOT (OpenSR-Test, 40 of 57 tiles) — <strong className="text-white">no Indian scene has been evaluated</strong>.</li>
              <li>The stability diagnostic was found only weakly associated with reconstruction error, about as strongly as plain image texture, and its raw spread is far smaller than the actual error — it is not calibrated.</li>
              <li>On a fixed-threshold NDVI vegetation decision, super-resolution changed the outcome only slightly, with a sign that depended on the dataset; no consistent advantage over bicubic upsampling was established.</li>
              <li>Land-cover classification was not evaluated: no dataset used carries region-level land-cover labels.</li>
            </ul>
            <p className="text-[11px] text-[#9AA3AD]">Full numbers and caveats: <code className="text-white">docs/CLAIMS.md</code> and <code className="text-white">docs/RELIABILITY.md</code> in the repository.</p>
          </div>
        )}
      </div>
    </div>
  );
};
