import React from 'react';
import { ArrowLeft, Layers, Ruler, Sparkles, Activity, Compass, Globe } from 'lucide-react';

interface ValidationDocsProps {
  onBack: () => void;
}

export const ValidationDocs: React.FC<ValidationDocsProps> = ({ onBack }) => {
  return (
    <div className="w-full min-h-[calc(100vh-3.5rem)] bg-[#0B0E12] text-white p-6">
      <div className="max-w-4xl mx-auto space-y-8">
        <button
          onClick={onBack}
          className="flex items-center gap-1.5 text-xs text-[#9AA3AD] hover:text-white transition-colors cursor-pointer"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Back to Application</span>
        </button>

        <div>
          <p className="text-xs font-mono uppercase tracking-wider text-[#D8B84C] mb-1">Methodology Guide</p>
          <h1 className="text-3xl font-bold tracking-tight text-white mb-3">Validation Protocol &amp; Metrics</h1>
          <p className="text-sm text-[#9AA3AD] leading-relaxed">
            Rigorous mathematical criteria for verifying super-resolved Earth observation rasters against observation consistency, reference sensors, and perturbation stability.
          </p>
        </div>

        <div className="space-y-6">
          {/* Pillar 1: Spectral Consistency */}
          <div className="p-6 bg-[#151A20] border border-[#1D242C] rounded-lg">
            <div className="flex items-center gap-2 mb-3">
              <Layers className="w-5 h-5 text-[#D8B84C]" />
              <h2 className="text-base font-semibold text-white">1. Spectral Fidelity &amp; Observation Consistency</h2>
            </div>
            <p className="text-xs text-[#9AA3AD] leading-relaxed mb-3">
              Evaluates whether the reconstructed 2.5m pixels maintain the physical surface reflectance observed by Sentinel-2. When 2.5m outputs are downsampled back to 10m using the sensor point spread function, the residual difference (reflectance bias) must be minimized.
            </p>
            <div className="font-mono text-[11px] bg-[#151A20] p-3 rounded border border-[#1D242C] text-[#9AA3AD] space-y-1">
              <div>• SAM (Spectral Angle Mapper): Measures angular difference between 12-channel spectral vectors.</div>
              <div>• ERGAS: Relative dimensionless global error in synthesis across multispectral bands.</div>
              <div>• Observation Reduction: 2.5m SR → 10m equivalent vs native Sentinel-2 observation.</div>
            </div>
          </div>

          {/* Pillar 2: Spatial Consistency */}
          <div className="p-6 bg-[#151A20] border border-[#1D242C] rounded-lg">
            <div className="flex items-center gap-2 mb-3">
              <Ruler className="w-5 h-5 text-[#D8B84C]" />
              <h2 className="text-base font-semibold text-white">2. Geospatial Alignment &amp; Geometry</h2>
            </div>
            <p className="text-xs text-[#9AA3AD] leading-relaxed mb-3">
              Super-resolution must preserve coordinate reference systems (CRS) and affine georeferencing matrices without sub-pixel coordinate drift or inter-band registration misalignment.
            </p>
            <div className="font-mono text-[11px] bg-[#151A20] p-3 rounded border border-[#1D242C] text-[#9AA3AD] space-y-1">
              <div>• Affine Residual: Geotiepoint position shift (&lt; 0.05 px target).</div>
              <div>• Boundary Displacement: Spatial distance between extracted road/water boundaries and HR references.</div>
            </div>
          </div>

          {/* Pillar 3: Detail & Correctness */}
          <div className="p-6 bg-[#151A20] border border-[#1D242C] rounded-lg">
            <div className="flex items-center gap-2 mb-3">
              <Sparkles className="w-5 h-5 text-[#D8B84C]" />
              <h2 className="text-base font-semibold text-white">3. OpenSR-Test Correctness Breakdown</h2>
            </div>
            <p className="text-xs text-[#9AA3AD] leading-relaxed mb-3">
              Following the European Space Agency OpenSR framework, generated high-frequency information is categorized into:
            </p>
            <div className="font-mono text-[11px] bg-[#151A20] p-3 rounded border border-[#1D242C] text-[#9AA3AD] space-y-1">
              <div>• Synthesis: Total quantity of newly introduced high-frequency spatial structure.</div>
              <div>• Improvement: Real spatial structure present in HR reference and reconstructed by model.</div>
              <div>• Omission: Fine detail present in HR reference but missed by super-resolution.</div>
              <div>• Hallucination: Artifacts or artificial high-frequency textures absent from the true scene.</div>
            </div>
          </div>

          {/* Pillar 4: Model Stability */}
          <div className="p-6 bg-[#151A20] border border-[#1D242C] rounded-lg">
            <div className="flex items-center gap-2 mb-3">
              <Activity className="w-5 h-5 text-[#D8B84C]" />
              <h2 className="text-base font-semibold text-white">4. Inference Stability Evidence</h2>
            </div>
            <p className="text-xs text-[#9AA3AD] leading-relaxed mb-3">
              FRAME computes pixel-level variance over 6 geometric inference passes (horizontal flip, vertical flip, patch window offset). Stability reflects invariance to arbitrary tiling and inference conditions, allowing automated flagging of unstable edges.
            </p>
          </div>

          {/* Pillar 5: Reference Comparison */}
          <div className="p-6 bg-[#151A20] border border-[#1D242C] rounded-lg">
            <div className="flex items-center gap-2 mb-3">
              <Compass className="w-5 h-5 text-[#D8B84C]" />
              <h2 className="text-base font-semibold text-white">5. Independent High-Resolution Reference</h2>
            </div>
            <p className="text-xs text-[#9AA3AD] leading-relaxed mb-3">
              Evaluated against authentic benchmarks such as SEN2NEON (AVIRIS-NG airborne hyperspectral data harmonized to Sentinel-2 spectral response functions at 2.5m) and OpenSR-Test (SPOT/PNOA).
            </p>
          </div>

          {/* Pillar 6: Domain Transfer */}
          <div className="p-6 bg-[#151A20] border border-[#1D242C] rounded-lg">
            <div className="flex items-center gap-2 mb-3">
              <Globe className="w-5 h-5 text-[#D8B84C]" />
              <h2 className="text-base font-semibold text-white">6. Geographic Domain Generalisation</h2>
            </div>
            <p className="text-xs text-[#9AA3AD] leading-relaxed mb-3">
              Testing model transferability onto held-out Indian geographic scenes (Chennai peri-urban, Punjab smallholder agriculture) without retraining to establish operational bounds.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
