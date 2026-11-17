/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { ArrowLeft, Activity, Compass, Globe, Layers, Ruler, Sparkles } from 'lucide-react';

interface ValidationDocsProps {
  onBack: () => void;
}

/**
 * A summary of FRAME's own separate, offline research (docs/CLAIMS.md, docs/RELIABILITY.md, docs/DOWNSTREAM.md in
 * the repository) — not a description of anything computed for a live reconstruction in this app (see the
 * Validate tab in the workspace for that). Every number and claim below matches those documents; nothing here
 * states a stronger conclusion than what was actually measured.
 */
export const ValidationDocs: React.FC<ValidationDocsProps> = ({ onBack }) => {
  return (
    <div className="w-full min-h-[calc(100vh-3.5rem)] bg-[#0B0E12] text-white p-6">
      <div className="max-w-4xl mx-auto space-y-8">
        <button onClick={onBack} className="flex items-center gap-1.5 text-xs text-[#9AA3AD] hover:text-white transition-colors cursor-pointer">
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Back to Application</span>
        </button>

        <div>
          <p className="text-xs font-mono uppercase tracking-wider text-[#D8B84C] mb-1">Separate, offline research — not this reconstruction</p>
          <h1 className="text-3xl font-bold tracking-tight text-white mb-3">What FRAME has actually measured</h1>
          <p className="text-sm text-[#9AA3AD] leading-relaxed">
            This summary describes real-data research carried out separately from any live upload in this app. Where the evidence is mixed, small, or missing, that is stated plainly rather than rounded up.
          </p>
        </div>

        <div className="space-y-6">
          <div className="p-6 bg-[#151A20] border border-[#1D242C] rounded-lg">
            <div className="flex items-center gap-2 mb-3">
              <Layers className="w-5 h-5 text-[#D8B84C]" />
              <h2 className="text-base font-semibold text-white">1. Spectral fidelity against real references</h2>
            </div>
            <p className="text-xs text-[#9AA3AD] leading-relaxed">
              Measured with PSNR, SSIM, RMSE, SAM, and ERGAS on real Sentinel-2 imagery against independent references. Both models are close to a plain bicubic upsample on these metrics; on the SEN2NEON benchmark
              both are slightly below bicubic (PSNR about 0.46 dB lower), and on OpenSR-Test the differences are within about 0.1 dB. No system is described as a winner.
            </p>
          </div>

          <div className="p-6 bg-[#151A20] border border-[#1D242C] rounded-lg">
            <div className="flex items-center gap-2 mb-3">
              <Ruler className="w-5 h-5 text-[#D8B84C]" />
              <h2 className="text-base font-semibold text-white">2. Geospatial alignment</h2>
            </div>
            <p className="text-xs text-[#9AA3AD] leading-relaxed">
              The output grid's CRS, origin and pixel size are verified preserved exactly (4× in both dimensions, no seam at tile joins for a tile-consistent model). Reference imagery itself is frequently
              mis-registered by roughly one high-resolution pixel; a registration-eligibility gate excludes tiles whose reference cannot be aligned before any accuracy number is computed from them.
            </p>
          </div>

          <div className="p-6 bg-[#151A20] border border-[#1D242C] rounded-lg">
            <div className="flex items-center gap-2 mb-3">
              <Sparkles className="w-5 h-5 text-[#D8B84C]" />
              <h2 className="text-base font-semibold text-white">3. Detail: synthesis, omission, hallucination</h2>
            </div>
            <p className="text-xs text-[#9AA3AD] leading-relaxed">
              Measured only on curated benchmark datasets with an independent reference (SEN2NEON, OpenSR-Test) — never for an arbitrary live upload, because the decomposition needs a reference this application does
              not have for your own scene. The Fourier hard constraint measurably reduces hallucinated detail compared to an unconstrained variant, at the cost of adding less new detail overall.
            </p>
          </div>

          <div className="p-6 bg-[#151A20] border border-[#1D242C] rounded-lg">
            <div className="flex items-center gap-2 mb-3">
              <Activity className="w-5 h-5 text-[#D8B84C]" />
              <h2 className="text-base font-semibold text-white">4. Model stability</h2>
            </div>
            <p className="text-xs text-[#9AA3AD] leading-relaxed">
              The six-view TTA stability diagnostic was tested against actual reconstruction error on registration-checked evidence. It is only <strong className="text-white">weakly</strong> associated with error — about
              as strongly as plain image texture alone — was not shown to identify high-error regions better than that trivial baseline, and its raw spread is 16–32× smaller than the actual error. It is not calibrated
              and is not treated as one anywhere in this application.
            </p>
          </div>

          <div className="p-6 bg-[#151A20] border border-[#1D242C] rounded-lg">
            <div className="flex items-center gap-2 mb-3">
              <Compass className="w-5 h-5 text-[#D8B84C]" />
              <h2 className="text-base font-semibold text-white">5. Independent high-resolution references used</h2>
            </div>
            <p className="text-xs text-[#9AA3AD] leading-relaxed">
              SEN2NEON (AVIRIS-NG airborne hyperspectral data, harmonised to the Sentinel-2 spectral response at 2.5 m; North America) and OpenSR-Test (SPOT and PNOA imagery; Spain). A registration gate admits only
              tiles whose reference aligns with the reconstruction grid — for SEN2NEON that was 12 of 30 sampled tiles; for OpenSR-Test, 40 of 57.
            </p>
          </div>

          <div className="p-6 bg-[#151A20] border border-[#1D242C] rounded-lg">
            <div className="flex items-center gap-2 mb-3">
              <Globe className="w-5 h-5 text-[#D8B84C]" />
              <h2 className="text-base font-semibold text-white">6. Geographic scope — no Indian evaluation</h2>
            </div>
            <p className="text-xs text-[#9AA3AD] leading-relaxed">
              All real-data evidence above is North America and Spain. <strong className="text-white">No Indian scene has an independent high-resolution reference in this project, and none has been evaluated</strong> — no
              synthetic "Indian ground truth" was created to fill that gap. Nothing in this application claims Indian validation or global generalisation.
            </p>
          </div>

          <div className="p-6 bg-[#151A20] border border-[#343C45] rounded-lg">
            <h2 className="text-base font-semibold text-white mb-2">Downstream utility (NDVI)</h2>
            <p className="text-xs text-[#9AA3AD] leading-relaxed">
              On a fixed-threshold NDVI vegetation decision, tested with the same registration-checked evidence, super-resolution changed region-level NDVI and the decision only slightly (on the order of one NDVI
              point and under one percentage point of decisions), with a sign that depended on the dataset. No consistent downstream advantage over bicubic upsampling was established, and no land-cover classification
              was evaluated — no dataset used carries region-level land-cover labels.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
