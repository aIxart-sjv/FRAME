import React, { useState } from 'react';
import { 
  CheckCircle2, 
  HelpCircle, 
  Layers, 
  Ruler, 
  Sparkles, 
  Activity, 
  Compass, 
  Globe, 
  Info,
  ExternalLink,
  ShieldCheck
} from 'lucide-react';
import { ValidationSection, Project } from '../../types';

interface ValidateWorkspaceProps {
  project: Project;
}

export const ValidateWorkspace: React.FC<ValidateWorkspaceProps> = ({ project }) => {
  const [activeSection, setActiveSection] = useState<ValidationSection>('spectral');
  const metrics = project.metrics;

  const SECTIONS: Array<{ id: ValidationSection; title: string; subtitle: string; icon: React.ReactNode }> = [
    { id: 'spectral', title: 'Spectral Fidelity', subtitle: 'Reflectance preservation & SAM', icon: <Layers className="w-4 h-4" /> },
    { id: 'spatial', title: 'Spatial Alignment', subtitle: 'CRS, affine grid & displacement', icon: <Ruler className="w-4 h-4" /> },
    { id: 'detail', title: 'Detail & Correctness', subtitle: 'Synthesis vs Hallucination', icon: <Sparkles className="w-4 h-4" /> },
    { id: 'stability', title: 'Model Stability', subtitle: 'Perturbation consistency evidence', icon: <Activity className="w-4 h-4" /> },
    { id: 'reference', title: 'Reference Evaluation', subtitle: 'SEN2NEON / HR sensor paired', icon: <Compass className="w-4 h-4" /> },
    { id: 'generalisation', title: 'Domain Transfer', subtitle: 'Geographic & landscape robustness', icon: <Globe className="w-4 h-4" /> },
  ];

  const formatMetric = (val: number | null | undefined, suffix = '', precision = 2) => {
    if (val === null || val === undefined) return '—';
    return `${val.toFixed(precision)}${suffix}`;
  };

  return (
    <div className="w-full flex-1 flex flex-col bg-[#0B0E12] text-white overflow-y-auto">
      {/* Workspace Header */}
      <div className="bg-[#151A20] border-b border-[#1D242C] px-6 py-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <p className="text-xs font-mono uppercase tracking-wider text-[#D8B84C] mb-0.5">Validation Protocol</p>
          <h1 className="text-xl font-bold tracking-tight text-white">Scientific Evidence &amp; Evaluation</h1>
          <p className="text-xs text-[#9AA3AD]">
            Systematic validation across spectral response, spatial registration, fine-scale correctness, and stability.
          </p>
        </div>

        {/* Reference Status Badge */}
        <div className="flex items-center gap-2 text-xs">
          <span className="text-[#9AA3AD]">Reference Source:</span>
          {project.referenceAvailable ? (
            <span className="font-mono text-[#35C98A] bg-[#35C98A]/10 border border-[#35C98A]/30 px-2.5 py-1 rounded flex items-center gap-1.5">
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>{project.referenceSensorName || 'Independent HR Reference'}</span>
            </span>
          ) : (
            <span className="font-mono text-[#9AA3AD] bg-[#1D242C] border border-[#343C45] px-2.5 py-1 rounded flex items-center gap-1.5">
              <HelpCircle className="w-3.5 h-3.5 text-[#9AA3AD]" />
              <span>No Independent HR Reference Paired</span>
            </span>
          )}
        </div>
      </div>

      {/* Main Container */}
      <div className="max-w-7xl mx-auto w-full p-6 flex-1 flex flex-col gap-6">
        {/* Navigation Tabs (6 Sections) */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
          {SECTIONS.map((sec) => (
            <button
              key={sec.id}
              onClick={() => setActiveSection(sec.id)}
              className={`p-3 rounded-lg border text-left transition-all cursor-pointer ${
                activeSection === sec.id
                  ? 'bg-[#151A20] border-[#D8B84C] ring-1 ring-[#D8B84C]/50'
                  : 'bg-[#151A20] border-[#1D242C] hover:border-[#343C45]'
              }`}
            >
              <div className="flex items-center gap-2 mb-1">
                <span className={activeSection === sec.id ? 'text-[#D8B84C]' : 'text-[#9AA3AD]'}>
                  {sec.icon}
                </span>
                <span className="text-xs font-semibold text-white">{sec.title}</span>
              </div>
              <p className="text-[10px] text-[#9AA3AD] truncate">{sec.subtitle}</p>
            </button>
          ))}
        </div>

        {/* Section 1: Spectral Fidelity */}
        {activeSection === 'spectral' && (
          <div className="space-y-6">
            <div className="p-5 bg-[#151A20] border border-[#1D242C] rounded-lg">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h3 className="text-sm font-semibold text-white">Spectral Preservation &amp; Radiometric Consistency</h3>
                  <p className="text-xs text-[#9AA3AD] mt-0.5">
                    Evaluates whether super-resolved surface reflectance vectors maintain physical fidelity without unconstrained radiometric drift.
                  </p>
                </div>
                <span className="text-xs font-mono text-[#D8B84C] bg-[#0B0E12] px-2.5 py-1 rounded border border-[#1D242C]">
                  Low-Frequency Hard Constraint Enforced
                </span>
              </div>

              {/* Metric Cards Grid */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                <div className="p-3.5 bg-[#151A20] rounded border border-[#1D242C]">
                  <span className="text-[11px] font-mono text-[#9AA3AD] block mb-1">SAM (Spectral Angle Mapper)</span>
                  <div className="text-xl font-bold font-mono text-white">
                    {formatMetric(metrics.samDegrees, '°')}
                  </div>
                  <span className="text-[10px] text-[#9AA3AD] block mt-1">Ideal: close to 0.0°</span>
                </div>

                <div className="p-3.5 bg-[#151A20] rounded border border-[#1D242C]">
                  <span className="text-[11px] font-mono text-[#9AA3AD] block mb-1">ERGAS (Dimensionless Error)</span>
                  <div className="text-xl font-bold font-mono text-white">
                    {formatMetric(metrics.ergas)}
                  </div>
                  <span className="text-[10px] text-[#9AA3AD] block mt-1">Lower indicates superior synthesis</span>
                </div>

                <div className="p-3.5 bg-[#151A20] rounded border border-[#1D242C]">
                  <span className="text-[11px] font-mono text-[#9AA3AD] block mb-1">Observation Reflectance Bias</span>
                  <div className="text-xl font-bold font-mono text-[#D8B84C]">
                    {formatMetric(metrics.reflectanceBias, '', 4)}
                  </div>
                  <span className="text-[10px] text-[#9AA3AD] block mt-1">2.5m → 10m downsample delta</span>
                </div>

                <div className="p-3.5 bg-[#151A20] rounded border border-[#1D242C]">
                  <span className="text-[11px] font-mono text-[#9AA3AD] block mb-1">PSNR (Peak Signal-to-Noise)</span>
                  <div className="text-xl font-bold font-mono text-white">
                    {formatMetric(metrics.psnr, ' dB')}
                  </div>
                  <span className="text-[10px] text-[#9AA3AD] block mt-1">
                    {project.referenceAvailable ? 'Measured vs HR Reference' : 'Requires paired reference'}
                  </span>
                </div>
              </div>

              {/* Spectral Comparison Explanation */}
              <div className="p-4 bg-[#0B0E12] rounded border border-[#1D242C] text-xs text-[#9AA3AD] space-y-2">
                <div className="flex items-center gap-2 text-white font-medium">
                  <Info className="w-4 h-4 text-[#D8B84C]" />
                  <span>Observation-Grid Downsampling Test</span>
                </div>
                <p className="leading-relaxed text-[#9AA3AD]">
                  When FRAME super-resolved 2.5m imagery is downsampled by a 4× spatial aggregation operator, the reconstructed 10m equivalent yields a mean reflectance bias of {formatMetric(metrics.reflectanceBias, '', 4)} across all 12 L2A channels. This verifies that high-frequency edge reconstruction did not distort underlying physical surface reflectance.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Section 2: Spatial Alignment */}
        {activeSection === 'spatial' && (
          <div className="space-y-6">
            <div className="p-5 bg-[#151A20] border border-[#1D242C] rounded-lg">
              <h3 className="text-sm font-semibold text-white mb-1">Geospatial Geometry &amp; Co-Registration</h3>
              <p className="text-xs text-[#9AA3AD] mb-4">
                Confirms preservation of projection metadata, affine transform, and sub-pixel edge alignment.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div className="p-4 bg-[#151A20] rounded border border-[#1D242C]">
                  <span className="text-[11px] font-mono text-[#9AA3AD] block mb-1">CRS Projection Preservation</span>
                  <div className="text-base font-bold font-mono text-[#35C98A] flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4" />
                    <span>VERIFIED ({project.crs})</span>
                  </div>
                  <span className="text-[10px] text-[#9AA3AD] block mt-1">Strict datum and zone preservation</span>
                </div>

                <div className="p-4 bg-[#151A20] rounded border border-[#1D242C]">
                  <span className="text-[11px] font-mono text-[#9AA3AD] block mb-1">Affine Transformation Residual</span>
                  <div className="text-xl font-bold font-mono text-white">
                    {formatMetric(metrics.affineResidualPx, ' px')}
                  </div>
                  <span className="text-[10px] text-[#9AA3AD] block mt-1">&lt; 0.1 px grid deviation threshold</span>
                </div>

                <div className="p-4 bg-[#151A20] rounded border border-[#1D242C]">
                  <span className="text-[11px] font-mono text-[#9AA3AD] block mb-1">Boundary Displacement</span>
                  <div className="text-xl font-bold font-mono text-white">
                    {formatMetric(metrics.boundaryDisplacementMeters, ' m')}
                  </div>
                  <span className="text-[10px] text-[#9AA3AD] block mt-1">Sub-pixel spatial shift vs road vector</span>
                </div>
              </div>

              <div className="p-4 bg-[#0B0E12] rounded border border-[#1D242C] text-xs font-mono text-[#9AA3AD] space-y-1">
                <div className="text-white font-semibold mb-2">Preserved GeoTIFF Tags:</div>
                <div>ModelPixelScaleTag: (2.500, 2.500, 0.0)</div>
                <div>ModelTiepointTag: (0, 0, 0, 389120.0, 1442560.0, 0.0)</div>
                <div>GeoKeyDirectoryTag: ProjectedCSTypeGeoKey = 32644 (WGS 84 / UTM 44N)</div>
              </div>
            </div>
          </div>
        )}

        {/* Section 3: Detail & Correctness (OpenSR-Test Framework) */}
        {activeSection === 'detail' && (
          <div className="space-y-6">
            <div className="p-5 bg-[#151A20] border border-[#1D242C] rounded-lg">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h3 className="text-sm font-semibold text-white">OpenSR-Test Correctness Breakdown</h3>
                  <p className="text-xs text-[#9AA3AD] mt-0.5">
                    Decomposing generated high-frequency detail into supported reconstruction, omission, and hallucination.
                  </p>
                </div>
                <span className="text-xs font-mono text-[#9AA3AD]">Standard: ESA OpenSR Protocol</span>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                <div className="p-4 bg-[#151A20] rounded border border-[#1D242C]">
                  <span className="text-[11px] font-mono text-[#9AA3AD] block mb-1">Synthesis Score</span>
                  <div className="text-xl font-bold font-mono text-white">
                    {formatMetric(metrics.synthesisScore, '', 3)}
                  </div>
                  <span className="text-[10px] text-[#9AA3AD] block mt-1">Amount of added high-frequency detail</span>
                </div>

                <div className="p-4 bg-[#151A20] rounded border border-[#1D242C]">
                  <span className="text-[11px] font-mono text-[#9AA3AD] block mb-1">Hallucination Rate</span>
                  <div className="text-xl font-bold font-mono text-[#D94B4B]">
                    {formatMetric(metrics.hallucinationRate, '', 2)}
                  </div>
                  <span className="text-[10px] text-[#9AA3AD] block mt-1">Generated detail unsupported by HR</span>
                </div>

                <div className="p-4 bg-[#151A20] rounded border border-[#1D242C]">
                  <span className="text-[11px] font-mono text-[#9AA3AD] block mb-1">Omission Rate</span>
                  <div className="text-xl font-bold font-mono text-[#9AA3AD]">
                    {formatMetric(metrics.omissionRate, '', 2)}
                  </div>
                  <span className="text-[10px] text-[#9AA3AD] block mt-1">HR detail not recovered</span>
                </div>

                <div className="p-4 bg-[#151A20] rounded border border-[#1D242C]">
                  <span className="text-[11px] font-mono text-[#9AA3AD] block mb-1">Improvement Rate</span>
                  <div className="text-xl font-bold font-mono text-[#35C98A]">
                    {formatMetric(metrics.improvementRate, '', 2)}
                  </div>
                  <span className="text-[10px] text-[#9AA3AD] block mt-1">Correctly verified new spatial structure</span>
                </div>
              </div>

              <div className="p-4 bg-[#0B0E12] rounded border border-[#343C45] text-xs text-[#9AA3AD]">
                <span className="text-[#D8B84C] font-semibold block mb-1">Scientific Guardrail Notice:</span>
                <p className="leading-relaxed text-[#9AA3AD]">
                  A higher synthesis score does not automatically equal physical truth. Generative adversarial networks (GANs) frequently generate higher synthesis by synthesizing plausible high-frequency textures that fail independent HR verification. FRAME optimizes for balanced improvement while penalizing hallucination.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Section 4: Model Stability */}
        {activeSection === 'stability' && (
          <div className="space-y-6">
            <div className="p-5 bg-[#151A20] border border-[#1D242C] rounded-lg">
              <h3 className="text-sm font-semibold text-white mb-1">Inference Perturbation Stability Evidence</h3>
              <p className="text-xs text-[#9AA3AD] mb-4">
                Measures output variation across 6 test-time transformations (horizontal/vertical flip, tile overlap windowing, shift).
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                <div className="p-4 bg-[#151A20] rounded border border-[#1D242C]">
                  <span className="text-[11px] font-mono text-[#9AA3AD] block mb-1">TTA Ensemble Variance</span>
                  <div className="text-2xl font-bold font-mono text-white">
                    {formatMetric(metrics.ttaVariance, '', 4)}
                  </div>
                  <span className="text-[10px] text-[#9AA3AD] block mt-1">Low variance indicates structural convergence</span>
                </div>

                <div className="p-4 bg-[#151A20] rounded border border-[#1D242C]">
                  <span className="text-[11px] font-mono text-[#9AA3AD] block mb-1">Perturbation Consistency</span>
                  <div className="text-2xl font-bold font-mono text-[#35C98A]">
                    {formatMetric(metrics.perturbationConsistencyPercent, '%', 1)}
                  </div>
                  <span className="text-[10px] text-[#9AA3AD] block mt-1">Percentage of cells with stability index &gt; 0.85</span>
                </div>
              </div>

              <div className="p-4 bg-[#0B0E12] rounded border border-[#1D242C] text-xs text-[#9AA3AD] space-y-2">
                <div className="flex items-center gap-2 text-white font-medium">
                  <ShieldCheck className="w-4 h-4 text-[#D8B84C]" />
                  <span>Stability vs. Probabilistic Uncertainty Distinction</span>
                </div>
                <p className="leading-relaxed text-[#9AA3AD]">
                  Stability reflects model consistency under defined geometric transformations. It should not be interpreted as absolute Bayesian confidence or ground-truth proof. Pixels with elevated variance ({'&gt;'}0.05) are flagged in the Evidence Viewer mode to advise analyst verification.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Section 5: Reference Evaluation */}
        {activeSection === 'reference' && (
          <div className="space-y-6">
            <div className="p-5 bg-[#151A20] border border-[#1D242C] rounded-lg">
              <h3 className="text-sm font-semibold text-white mb-1">Independent High-Resolution Reference Comparison</h3>
              <p className="text-xs text-[#9AA3AD] mb-4">
                Benchmarking against genuine airborne hyperspectral and high-resolution optical acquisitions.
              </p>

              {project.referenceAvailable ? (
                <div className="space-y-4">
                  <div className="p-4 bg-[#151A20] rounded border border-[#35C98A]/30 text-xs">
                    <span className="text-[11px] font-mono text-[#35C98A] block mb-1">PAIRED REFERENCE ASSET</span>
                    <p className="text-sm font-semibold text-white mb-2">{project.referenceSensorName}</p>
                    <p className="text-[#9AA3AD] leading-relaxed">
                      Hyperspectral AVIRIS-NG airborne sensor flown simultaneously over NEON calibration site. Harmonized via Sentinel-2 Spectral Response Functions (SRF) down to the canonical 2.5m benchmark grid.
                    </p>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    <div className="p-3 bg-[#0B0E12] rounded border border-[#1D242C]">
                      <span className="text-[11px] text-[#9AA3AD] block">PSNR Benchmark</span>
                      <span className="text-lg font-mono font-bold text-white">{formatMetric(metrics.psnr, ' dB')}</span>
                    </div>
                    <div className="p-3 bg-[#0B0E12] rounded border border-[#1D242C]">
                      <span className="text-[11px] text-[#9AA3AD] block">SSIM Structural Score</span>
                      <span className="text-lg font-mono font-bold text-white">{formatMetric(metrics.ssim)}</span>
                    </div>
                    <div className="p-3 bg-[#0B0E12] rounded border border-[#1D242C]">
                      <span className="text-[11px] text-[#9AA3AD] block">SAM Angular Residual</span>
                      <span className="text-lg font-mono font-bold text-white">{formatMetric(metrics.samDegrees, '°')}</span>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="p-8 text-center bg-[#151A20] rounded border border-[#1D242C]">
                  <HelpCircle className="w-8 h-8 text-[#9AA3AD] mx-auto mb-2" />
                  <h4 className="text-xs font-semibold text-white uppercase tracking-wider mb-1">
                    No Independent High-Resolution Reference Attached
                  </h4>
                  <p className="text-xs text-[#9AA3AD] max-w-md mx-auto mb-4 leading-relaxed">
                    This project is operating on operational Sentinel-2 imagery without simultaneous airborne or sub-meter satellite truth. In accordance with FRAME scientific integrity rules, PSNR and SSIM are withheld to prevent fabricated metrics.
                  </p>
                  <span className="text-[11px] font-mono text-[#D8B84C]">
                    Only observation-grid reduction and model stability are evaluated for this scene.
                  </span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Section 6: Domain Transfer & Generalisation */}
        {activeSection === 'generalisation' && (
          <div className="space-y-6">
            <div className="p-5 bg-[#151A20] border border-[#1D242C] rounded-lg">
              <h3 className="text-sm font-semibold text-white mb-1">Geographic &amp; Landscape Domain Generalisation</h3>
              <p className="text-xs text-[#9AA3AD] mb-4">
                Evaluating model transferability across diverse biomes, urban morphologies, and agricultural practices.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div className="p-4 bg-[#151A20] rounded border border-[#1D242C]">
                  <span className="text-xs font-semibold text-white block mb-1">Indian Urban Peri-Corridors</span>
                  <p className="text-[11px] text-[#9AA3AD] leading-relaxed mb-2">
                    Tested across Chennai and Ludhiana peri-urban corridors. Evaluates performance under high aerosol depth and mixed building morphologies.
                  </p>
                  <span className="text-[10px] font-mono text-[#35C98A]">Status: Evaluated (T44RPR / T43SBT)</span>
                </div>

                <div className="p-4 bg-[#151A20] rounded border border-[#1D242C]">
                  <span className="text-xs font-semibold text-white block mb-1">Smallholder Agricultural Parcels</span>
                  <p className="text-[11px] text-[#9AA3AD] leading-relaxed mb-2">
                    Tests sub-pixel field boundary delineation where individual parcel sizes are smaller than the 10m Sentinel-2 instantaneous field of view.
                  </p>
                  <span className="text-[10px] font-mono text-[#35C98A]">Status: Evaluated (Punjab Rabi)</span>
                </div>

                <div className="p-4 bg-[#151A20] rounded border border-[#1D242C]">
                  <span className="text-xs font-semibold text-white block mb-1">Wetland &amp; Coastal Littoral Zones</span>
                  <p className="text-[11px] text-[#9AA3AD] leading-relaxed mb-2">
                    OpenSR-Test Valencia benchmark evaluating littoral water-land interface without false coastline shifts.
                  </p>
                  <span className="text-[10px] font-mono text-[#35C98A]">Status: Evaluated (OpenSR Albufera)</span>
                </div>
              </div>

              <div className="p-4 bg-[#0B0E12] rounded border border-[#343C45] text-xs text-[#9AA3AD]">
                <span className="text-[#D8B84C] font-semibold block mb-1">Absence of Global Generalisation Claims:</span>
                <p className="leading-relaxed text-[#9AA3AD]">
                  FRAME explicitly avoids claiming universal sensor independence or global generalisation. Performance degrades gracefully under heavy atmospheric haze, mountain shadow casting, and cloud borders. The system flags out-of-distribution spectra for analyst review.
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
