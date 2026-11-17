import React, { useState } from 'react';
import { ArrowLeft, BookOpen, Layers, Cpu, ShieldCheck, PieChart, Download, Terminal } from 'lucide-react';

interface DocumentationViewProps {
  onBack: () => void;
  onNavigate: (view: string) => void;
}

export const DocumentationView: React.FC<DocumentationViewProps> = ({ onBack, onNavigate }) => {
  const [activeTopic, setActiveTopic] = useState<'getting-started' | 'data' | 'reconstruction' | 'validation' | 'analysis' | 'export' | 'api'>('getting-started');

  return (
    <div className="w-full min-h-[calc(100vh-3.5rem)] bg-[#0B0E12] text-white p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        <button
          onClick={onBack}
          className="flex items-center gap-1.5 text-xs text-[#9AA3AD] hover:text-white transition-colors cursor-pointer"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Back to Application</span>
        </button>

        <div className="pb-4 border-b border-[#1D242C]">
          <h1 className="text-2xl font-bold tracking-tight text-white">FRAME Technical Documentation</h1>
          <p className="text-xs text-[#9AA3AD] mt-0.5">
            Architecture, workflows, and scientific principles for SIH26142 Sentinel-2 Super-Resolution Mapping.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
          {/* Left Navigation Topics */}
          <div className="md:col-span-3 space-y-1">
            {[
              { id: 'getting-started' as const, label: 'Getting Started', icon: <BookOpen className="w-3.5 h-3.5" /> },
              { id: 'data' as const, label: 'Data & Harmonisation', icon: <Layers className="w-3.5 h-3.5" /> },
              { id: 'reconstruction' as const, label: 'SR Pipeline (Mamba)', icon: <Cpu className="w-3.5 h-3.5" /> },
              { id: 'validation' as const, label: 'Validation Protocols', icon: <ShieldCheck className="w-3.5 h-3.5" /> },
              { id: 'analysis' as const, label: 'Downstream Utility', icon: <PieChart className="w-3.5 h-3.5" /> },
              { id: 'export' as const, label: 'GIS Export & COGs', icon: <Download className="w-3.5 h-3.5" /> },
              { id: 'api' as const, label: 'REST API & Batching', icon: <Terminal className="w-3.5 h-3.5" /> },
            ].map((topic) => (
              <button
                key={topic.id}
                onClick={() => setActiveTopic(topic.id)}
                className={`w-full flex items-center gap-2.5 px-3 py-2 text-xs font-medium rounded transition-colors text-left cursor-pointer ${
                  activeTopic === topic.id
                    ? 'bg-[#151A20] text-[#D8B84C] border-l-2 border-[#D8B84C]'
                    : 'text-[#9AA3AD] hover:text-white hover:bg-[#151A20]'
                }`}
              >
                {topic.icon}
                <span>{topic.label}</span>
              </button>
            ))}
          </div>

          {/* Right Topic Details */}
          <div className="md:col-span-9 p-6 bg-[#151A20] border border-[#1D242C] rounded-lg text-xs leading-relaxed space-y-6">
            {activeTopic === 'getting-started' && (
              <div className="space-y-4">
                <h2 className="text-base font-bold text-white">Getting Started with FRAME</h2>
                <p className="text-[#9AA3AD]">
                  FRAME is an end-to-end scientific processing pipeline that ingests medium-resolution Sentinel-2 Level-2A imagery, super-resolves the 12 analytical surface reflectance bands onto a 2.5m grid, and evaluates the reconstruction for geospatial analysis.
                </p>
                <div className="p-4 bg-[#0B0E12] rounded border border-[#1D242C] space-y-2">
                  <h3 className="font-semibold text-white">Standard Analyst Workflow:</h3>
                  <ol className="list-decimal pl-5 space-y-1.5 text-[#9AA3AD]">
                    <li><strong>Define AOI</strong>: Search by place, input WGS84 coordinates, or draw a bounding polygon.</li>
                    <li><strong>Filter Imagery</strong>: Discover Sentinel-2 acquisitions via STAC from Copernicus Data Space or Bhoonidhi.</li>
                    <li><strong>Execute Asynchronous Job</strong>: Tiled inference with weighted patch boundary blending and Fourier constraints.</li>
                    <li><strong>Inspect in 4 Modes</strong>: Compare Original (10m) vs SR (2.5m), review Analytical indices, or inspect Evidence stability.</li>
                    <li><strong>Validate &amp; Export</strong>: Benchmark against HR references and export GIS-ready Cloud-Optimized GeoTIFFs.</li>
                  </ol>
                </div>
              </div>
            )}

            {activeTopic === 'data' && (
              <div className="space-y-4">
                <h2 className="text-base font-bold text-white">Sentinel-2 Multispectral Preprocessing</h2>
                <p className="text-[#9AA3AD]">
                  Sentinel-2 MSI records bands at three different native ground sample distances: 10m (B2, B3, B4, B8), 20m (B5, B6, B7, B8A, B11, B12), and 60m (B1, B9, B10).
                </p>
                <div className="p-3 bg-[#0B0E12] border border-[#1D242C] rounded text-[#9AA3AD]">
                  <strong className="text-white block mb-1">Radiometric Baseline Conversion:</strong>
                  <code>Surface_Reflectance = (DN - BOA_ADD_OFFSET) / QUANTIFICATION_VALUE</code><br />
                  For baseline 04.00+ products: <code>(DN - 1000) / 10000</code>.
                </div>
                <p className="text-[#9AA3AD]">
                  Band 10 (1375 nm cirrus) is systematically excluded from the analytical surface reflectance stack because its extreme atmospheric absorption renders it unsuitable for ground-level reflectance calculations.
                </p>
              </div>
            )}

            {activeTopic === 'reconstruction' && (
              <div className="space-y-4">
                <h2 className="text-base font-bold text-white">Super-Resolution Model Architecture</h2>
                <p className="text-[#9AA3AD]">
                  FRAME builds upon the European Space Agency OpenSR / SEN2SR framework. Rather than unconstrained GANs which risk high-frequency hallucinations, FRAME adopts a state-space Mamba architecture with an integrated low-frequency Fourier hard constraint.
                </p>
                <div className="p-3 bg-[#0B0E12] border border-[#1D242C] rounded text-[#9AA3AD]">
                  <strong className="text-white block mb-1">Low-Frequency Hard Constraint:</strong>
                  The network separates high-frequency spatial synthesis from low-frequency structural reflectance. The low-frequency component of the input is strictly preserved, ensuring that the 2.5m product aggregates back to the physical 10m Sentinel-2 observation.
                </div>
              </div>
            )}

            {activeTopic === 'validation' && (
              <div className="space-y-4">
                <h2 className="text-base font-bold text-white">Scientific Accuracy &amp; Validation Protocol</h2>
                <p className="text-[#9AA3AD]">
                  Reconstruction metrics are decomposed into 6 distinct dimensions: Spectral (SAM, ERGAS), Spatial (affine transform, registration), Detail (synthesis vs hallucination), Stability (perturbation variance), Reference comparison (SEN2NEON benchmark), and Generalisation (held-out Indian test scenes).
                </p>
                <button
                  onClick={() => onNavigate('validation-docs')}
                  className="text-[#D8B84C] hover:underline block"
                >
                  View full Validation Documentation →
                </button>
              </div>
            )}

            {activeTopic === 'analysis' && (
              <div className="space-y-4">
                <h2 className="text-base font-bold text-white">Downstream Thematic Utility</h2>
                <p className="text-[#9AA3AD]">
                  The central thesis of FRAME is that visual sharpness does not guarantee analytical improvement. Downstream utility is experimentally evaluated on 5-class land-cover classification and multi-temporal change detection, comparing 10m native models against 2.5m super-resolved inputs.
                </p>
              </div>
            )}

            {activeTopic === 'export' && (
              <div className="space-y-4">
                <h2 className="text-base font-bold text-white">GIS-Ready Export &amp; Cloud Optimized GeoTIFFs (COGs)</h2>
                <p className="text-[#9AA3AD]">
                  Exported products strictly preserve Coordinate Reference Systems (e.g. UTM zones WGS84) and Affine georeferencing matrices. Cloud Optimized GeoTIFFs feature internal tile structuring and overviews for dynamic rendering in QGIS, ArcGIS, or TiTiler web map servers.
                </p>
              </div>
            )}

            {activeTopic === 'api' && (
              <div className="space-y-4">
                <h2 className="text-base font-bold text-white">REST API Specification (FastAPI / Asynchronous Queue)</h2>
                <p className="text-[#9AA3AD]">
                  Large satellite scenes are processed asynchronously to avoid blocking HTTP timeouts.
                </p>
                <div className="p-3 bg-[#0B0E12] font-mono text-[11px] border border-[#1D242C] rounded text-[#9AA3AD] space-y-1">
                  <div>POST /jobs - Submit AOI &amp; Scene for tiled reconstruction</div>
                  <div>GET /jobs/{'{job_id}'} - Query progress and stage logs</div>
                  <div>GET /products/{'{id}'}/cog - Stream 2.5m Cloud-Optimized GeoTIFF</div>
                  <div>GET /products/{'{id}'}/stability - Retrieve TTA perturbation variance raster</div>
                  <div>GET /health - Cluster GPU and worker availability</div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
