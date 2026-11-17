/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { ArrowLeft, Layers, ShieldAlert } from 'lucide-react';

interface AboutSentinel2Props {
  onBack: () => void;
}

/** General Sentinel-2 mission facts (not a FRAME capability claim) — ESA's own published MSI band specification. */
const SENTINEL2_BANDS = [
  { id: 'B01', name: 'Coastal aerosol', centralWavelength: 443, nativeResolution: 60, processedByFrame: false },
  { id: 'B02', name: 'Blue', centralWavelength: 490, nativeResolution: 10, processedByFrame: true },
  { id: 'B03', name: 'Green', centralWavelength: 560, nativeResolution: 10, processedByFrame: true },
  { id: 'B04', name: 'Red', centralWavelength: 665, nativeResolution: 10, processedByFrame: true },
  { id: 'B05', name: 'Red Edge 1', centralWavelength: 705, nativeResolution: 20, processedByFrame: false },
  { id: 'B06', name: 'Red Edge 2', centralWavelength: 740, nativeResolution: 20, processedByFrame: false },
  { id: 'B07', name: 'Red Edge 3', centralWavelength: 783, nativeResolution: 20, processedByFrame: false },
  { id: 'B08', name: 'NIR', centralWavelength: 842, nativeResolution: 10, processedByFrame: true },
  { id: 'B8A', name: 'NIR Narrow', centralWavelength: 865, nativeResolution: 20, processedByFrame: false },
  { id: 'B09', name: 'Water vapour', centralWavelength: 945, nativeResolution: 60, processedByFrame: false },
  { id: 'B10', name: 'Cirrus', centralWavelength: 1375, nativeResolution: 60, processedByFrame: false, excludedFromL2A: true },
  { id: 'B11', name: 'SWIR 1', centralWavelength: 1610, nativeResolution: 20, processedByFrame: false },
  { id: 'B12', name: 'SWIR 2', centralWavelength: 2190, nativeResolution: 20, processedByFrame: false },
];

export const AboutSentinel2: React.FC<AboutSentinel2Props> = ({ onBack }) => {
  return (
    <div className="w-full min-h-[calc(100vh-3.5rem)] bg-[#0B0E12] text-white p-6">
      <div className="max-w-4xl mx-auto space-y-8">
        <button onClick={onBack} className="flex items-center gap-1.5 text-xs text-[#9AA3AD] hover:text-white transition-colors cursor-pointer">
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Back to Application</span>
        </button>

        <div>
          <p className="text-xs font-mono uppercase tracking-wider text-[#D8B84C] mb-1">Background</p>
          <h1 className="text-3xl font-bold tracking-tight text-white mb-3">About Sentinel-2 &amp; MSI Imagery</h1>
          <p className="text-sm text-[#9AA3AD] leading-relaxed">
            The European Space Agency's Copernicus Sentinel-2 mission carries the Multi-Spectral Instrument (MSI), providing systematic global optical Earth observations across 13 spectral channels.
          </p>
        </div>

        <div className="p-4 bg-[#151A20] border border-[#D8B84C]/40 rounded-lg flex items-start gap-3">
          <ShieldAlert className="w-5 h-5 text-[#D8B84C] shrink-0 mt-0.5" />
          <p className="text-xs text-[#9AA3AD] leading-relaxed">
            <strong className="text-white">This application processes 4 of these 12 usable bands</strong> — B04, B03, B02, B08 (RGBN). The table below is general Sentinel-2 background, not a description of what this
            reconstruction pipeline reads; the "Processed by FRAME" column says which is which.
          </p>
        </div>

        <section className="p-6 bg-[#151A20] border border-[#1D242C] rounded-lg space-y-4">
          <div className="flex items-center gap-2 text-white font-semibold">
            <Layers className="w-5 h-5 text-[#D8B84C]" />
            <h2>The 12 Usable Level-2A Analytical Bands</h2>
          </div>
          <p className="text-xs text-[#9AA3AD] leading-relaxed">
            Band 10 (1375 nm) is excluded from Level-2A surface-reflectance products: it sits in a strong atmospheric water-vapour absorption region designed for high-altitude cirrus screening at Top-of-Atmosphere, not ground reflectance.
          </p>
          <div className="overflow-x-auto pt-2">
            <table className="w-full text-left text-xs border border-[#1D242C]">
              <thead className="bg-[#151A20] font-mono text-[#9AA3AD] border-b border-[#1D242C]">
                <tr>
                  <th className="p-2.5">Band</th>
                  <th className="p-2.5">Name</th>
                  <th className="p-2.5">Wavelength</th>
                  <th className="p-2.5">Native Resolution</th>
                  <th className="p-2.5">Processed by FRAME</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1D242C] font-mono">
                {SENTINEL2_BANDS.map((b) => (
                  <tr key={b.id} className={b.excludedFromL2A ? 'bg-[#D94B4B]/10 text-[#D94B4B]' : ''}>
                    <td className="p-2.5 font-bold text-white">{b.id}</td>
                    <td className="p-2.5 font-sans text-[#9AA3AD]">{b.name}</td>
                    <td className="p-2.5">{b.centralWavelength} nm</td>
                    <td className="p-2.5">{b.nativeResolution} m</td>
                    <td className="p-2.5">
                      {b.excludedFromL2A ? <span className="text-[#D94B4B]">Excluded from L2A (cirrus only)</span> : b.processedByFrame ? <span className="text-[#35C98A]">Yes</span> : <span className="text-[#9AA3AD]">No</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="space-y-4">
          <h2 className="text-xl font-bold text-white tracking-tight">Why super-resolution is explored for Sentinel-2</h2>
          <p className="text-xs text-[#9AA3AD] leading-relaxed">
            Sentinel-2 provides high temporal revisit (5 days at the equator) but its 10 m native spatial resolution struggles to resolve smallholder agricultural parcels, narrow road networks, and peri-urban boundary
            expansions. Commercial high-resolution satellites offer finer resolution but at commercial pricing and without Sentinel-2's spectral depth or open revisit cadence.
          </p>
          <p className="text-xs text-[#9AA3AD] leading-relaxed">
            FRAME explores super-resolving the RGBN subset of Sentinel-2 onto a 2.5 m analysis grid using a Fourier-domain low-frequency hard constraint intended to preserve physical spectral reflectance in the reconstruction.
          </p>
        </section>

        <section className="p-5 bg-[#151A20] border border-[#343C45] rounded-lg flex items-start gap-3">
          <ShieldAlert className="w-5 h-5 text-[#D8B84C] shrink-0 mt-0.5" />
          <div className="text-xs text-[#9AA3AD] space-y-1">
            <h4 className="font-semibold text-white">Scientific Integrity Distinction</h4>
            <p className="leading-relaxed">
              FRAME's 2.5 m product represents a mathematically reconstructed grid, not a physically observed 2.5 m sensor measurement. Visual sharpness must never be conflated with empirical accuracy; see the Validate
              tab for what is, and is not, evidenced for any given reconstruction.
            </p>
          </div>
        </section>
      </div>
    </div>
  );
};
