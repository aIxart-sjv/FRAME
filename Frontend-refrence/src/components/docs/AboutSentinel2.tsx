import React from 'react';
import { SENTINEL2_BANDS } from '../../data/mockData';
import { Layers, Info, ShieldAlert, ArrowLeft } from 'lucide-react';

interface AboutSentinel2Props {
  onBack: () => void;
}

export const AboutSentinel2: React.FC<AboutSentinel2Props> = ({ onBack }) => {
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
          <p className="text-xs font-mono uppercase tracking-wider text-[#D8B84C] mb-1">Scientific Source of Truth</p>
          <h1 className="text-3xl font-bold tracking-tight text-white mb-3">About Sentinel-2 &amp; MSI Imagery</h1>
          <p className="text-sm text-[#9AA3AD] leading-relaxed">
            The European Space Agency’s Copernicus Sentinel-2 mission carries the Multi-Spectral Instrument (MSI), providing systematic global optical Earth observations.
          </p>
        </div>

        {/* 13 Bands vs 12 L2A Bands Detailed Explanation */}
        <section className="p-6 bg-[#151A20] border border-[#1D242C] rounded-lg space-y-4">
          <div className="flex items-center gap-2 text-white font-semibold">
            <Layers className="w-5 h-5 text-[#D8B84C]" />
            <h2>The 12 Usable Level-2A Analytical Bands</h2>
          </div>
          <p className="text-xs text-[#9AA3AD] leading-relaxed">
            While the Sentinel-2 MSI instrument records across 13 spectral channels, <strong className="text-white">Band 10 (1375 nm) is excluded from Level-2A surface-reflectance products</strong>. Band 10 sits in a strong atmospheric water vapour absorption region designed exclusively for high-altitude cirrus cloud screening at Top-of-Atmosphere (L1C). For ground surface reflectance, FRAME standardizes on the 12 analytical bands: B01–B09, B8A, B11, and B12.
          </p>

          <div className="overflow-x-auto pt-2">
            <table className="w-full text-left text-xs border border-[#1D242C]">
              <thead className="bg-[#151A20] font-mono text-[#9AA3AD] border-b border-[#1D242C]">
                <tr>
                  <th className="p-2.5">Band ID</th>
                  <th className="p-2.5">Band Name</th>
                  <th className="p-2.5">Central Wavelength</th>
                  <th className="p-2.5">Native Resolution</th>
                  <th className="p-2.5">L2A Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1D242C] font-mono">
                {SENTINEL2_BANDS.map((b) => (
                  <tr key={b.id} className={b.isAnalyticalL2A ? '' : 'bg-[#D94B4B]/10 text-[#D94B4B]'}>
                    <td className="p-2.5 font-bold text-white">{b.id}</td>
                    <td className="p-2.5 font-sans text-[#9AA3AD]">{b.name}</td>
                    <td className="p-2.5">{b.centralWavelength} nm</td>
                    <td className="p-2.5">{b.nativeResolution} m</td>
                    <td className="p-2.5">
                      {b.isAnalyticalL2A ? (
                        <span className="text-[#35C98A]">Included (Analytical)</span>
                      ) : (
                        <span className="text-[#D94B4B]">Excluded (Cirrus Only)</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* Why Super-Resolution is Explored */}
        <section className="space-y-4">
          <h2 className="text-xl font-bold text-white tracking-tight">
            Why Super-Resolution is Explored for Sentinel-2
          </h2>
          <p className="text-xs text-[#9AA3AD] leading-relaxed">
            Sentinel-2 provides high temporal revisit (5 days at the equator) and rich 12-band multispectral coverage, but its 10m native spatial resolution struggles to resolve smallholder agricultural parcels (common in Indian agriculture), narrow road networks, and peri-urban boundary expansions. High-resolution commercial satellites (e.g. WorldView, PlanetScope) offer sub-3m resolution but are commercially priced and lack the red-edge and SWIR spectral depth of Sentinel-2.
          </p>
          <p className="text-xs text-[#9AA3AD] leading-relaxed">
            FRAME explores super-resolving Sentinel-2 onto a 2.5m analysis grid while constraining the high-frequency reconstruction with Fourier-domain low-frequency hard constraints to preserve physical spectral reflectance.
          </p>
        </section>

        {/* Scientific Limitation & Framing */}
        <section className="p-5 bg-[#151A20] border border-[#343C45] rounded-lg flex items-start gap-3">
          <ShieldAlert className="w-5 h-5 text-[#D8B84C] shrink-0 mt-0.5" />
          <div className="text-xs text-[#9AA3AD] space-y-1">
            <h4 className="font-semibold text-white">Scientific Integrity Distinction</h4>
            <p className="text-[#9AA3AD] leading-relaxed">
              FRAME’s 2.5m product represents a mathematically reconstructed grid, not a physically observed 2.5m sensor measurement. Visual sharpness must never be conflated with empirical accuracy; all reconstructed outputs must be validated using spectral consistency, spatial alignment, and model stability evidence.
            </p>
          </div>
        </section>
      </div>
    </div>
  );
};
