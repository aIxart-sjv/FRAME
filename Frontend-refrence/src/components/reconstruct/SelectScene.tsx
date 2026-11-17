import React, { useState } from 'react';
import { Calendar, Cloud, CheckCircle2, ArrowRight, ArrowLeft } from 'lucide-react';
import { AVAILABLE_SCENES } from '../../data/mockData';
import { Sentinel2Scene } from '../../types';

interface SelectSceneProps {
  selectedAOI: {
    name: string;
    coordinates: [number, number];
    bbox: [number, number, number, number];
  };
  onSelectScene: (scene: Sentinel2Scene) => void;
  onBack: () => void;
}

export const SelectScene: React.FC<SelectSceneProps> = ({ selectedAOI, onSelectScene, onBack }) => {
  const [maxCloudCover, setMaxCloudCover] = useState<number>(10);
  const [selectedScene, setSelectedScene] = useState<Sentinel2Scene>(AVAILABLE_SCENES[0]);
  const [filterProvider, setFilterProvider] = useState<'all' | 'Copernicus' | 'Bhoonidhi'>('all');

  const filteredScenes = AVAILABLE_SCENES.filter((scene) => {
    if (scene.cloudCoverPercentage > maxCloudCover) return false;
    if (filterProvider === 'Copernicus' && !scene.provider.includes('Copernicus')) return false;
    if (filterProvider === 'Bhoonidhi' && !scene.provider.includes('Bhoonidhi')) return false;
    return true;
  });

  return (
    <div className="w-full min-h-[calc(100vh-3.5rem)] bg-[#0B0E12] text-white flex flex-col">
      {/* Top Stepper Header */}
      <div className="border-b border-[#1D242C] px-6 py-6 bg-[#151A20]">
        <div className="max-w-6xl mx-auto flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs font-mono uppercase tracking-wider text-[#D8B84C]">Step 2 of 3</span>
              <span className="text-[#343C45]">·</span>
              <span className="text-xs text-[#9AA3AD]">{selectedAOI.name}</span>
            </div>
            <h1 className="text-2xl font-bold tracking-tight text-white">Choose your imagery</h1>
            <p className="text-xs text-[#9AA3AD] mt-0.5">
              Select a Sentinel-2 Level-2A acquisition covering your area.
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
              disabled={!selectedScene}
              onClick={() => selectedScene && onSelectScene(selectedScene)}
              className="flex items-center gap-2 px-5 py-2 text-xs font-semibold rounded transition-all cursor-pointer bg-[#D8B84C] hover:bg-[#E4C96A] text-[#151A20] shadow-md"
            >
              <span>Confirm Scene Selection</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>

      {/* Main Layout: Left Scene Catalog, Right Map Swath Visualization */}
      <div className="max-w-6xl mx-auto w-full flex-1 p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Filter Controls and Scene List */}
        <div className="lg:col-span-6 flex flex-col gap-4">
          {/* Filters Bar */}
          <div className="p-4 bg-[#151A20] border border-[#1D242C] rounded-lg space-y-3">
            <div className="flex items-center justify-between text-xs">
              <span className="text-[#9AA3AD] font-medium">Catalogue Filtering</span>
              <span className="text-[11px] font-mono text-[#D8B84C]">
                {filteredScenes.length} acquisition{filteredScenes.length === 1 ? '' : 's'} available
              </span>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="flex justify-between text-[11px] mb-1">
                  <span className="text-[#9AA3AD]">Max Cloud Threshold</span>
                  <span className="font-mono text-white">{maxCloudCover}%</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="30"
                  value={maxCloudCover}
                  onChange={(e) => setMaxCloudCover(Number(e.target.value))}
                  className="w-full accent-[#D8B84C] bg-[#1D242C] h-1.5 rounded cursor-pointer"
                />
              </div>

              <div>
                <label className="text-[11px] text-[#9AA3AD] block mb-1">Dissemination Source</label>
                <select
                  value={filterProvider}
                  onChange={(e) => setFilterProvider(e.target.value as any)}
                  className="w-full bg-[#0B0E12] border border-[#343C45] rounded px-2.5 py-1 text-xs text-white focus:outline-none focus:border-[#D8B84C]"
                >
                  <option value="all">All (Copernicus &amp; Bhoonidhi)</option>
                  <option value="Bhoonidhi">Bhoonidhi NRSC/ISRO</option>
                  <option value="Copernicus">Copernicus Data Space</option>
                </select>
              </div>
            </div>
          </div>

          {/* Scene Cards List */}
          <div className="space-y-3 overflow-y-auto max-h-[520px] pr-1">
            {filteredScenes.map((scene) => {
              const isSelected = selectedScene.id === scene.id;
              return (
                <div
                  key={scene.id}
                  onClick={() => setSelectedScene(scene)}
                  className={`p-4 rounded-lg border text-left cursor-pointer transition-all ${
                    isSelected
                      ? 'bg-[#181d26] border-[#D8B84C] ring-1 ring-[#D8B84C]/50'
                      : 'bg-[#151A20] border-[#1D242C] hover:border-[#343C45]'
                  }`}
                >
                  <div className="flex items-start justify-between gap-3 mb-2">
                    <div>
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-xs font-mono font-bold text-white tracking-wide">
                          {scene.tileId}
                        </span>
                        <span className="text-[11px] px-1.5 py-0.2 bg-[#1D242C] rounded text-[#9AA3AD] font-mono">
                          L2A
                        </span>
                        <span className="text-[11px] text-[#9AA3AD]">{scene.provider}</span>
                      </div>
                      <p className="text-[11px] font-mono text-[#9AA3AD] break-all leading-tight">
                        {scene.id}
                      </p>
                    </div>
                    {isSelected && (
                      <CheckCircle2 className="w-5 h-5 text-[#D8B84C] shrink-0 mt-0.5" />
                    )}
                  </div>

                  <div className="grid grid-cols-3 gap-2 pt-2 border-t border-[#1D242C] text-[11px] font-mono text-[#9AA3AD]">
                    <div className="flex items-center gap-1.5">
                      <Calendar className="w-3.5 h-3.5 text-[#D8B84C]" />
                      <span>{scene.acquisitionDatetime.split(' ')[0]}</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <Cloud className="w-3.5 h-3.5 text-[#9AA3AD]" />
                      <span>{scene.cloudCoverPercentage}% Cloud</span>
                    </div>
                    <div className="text-right text-[#9AA3AD]">
                      <span>{scene.crs.split(' ')[0]}</span>
                    </div>
                  </div>

                  {scene.surfaceConditions && (
                    <div className="mt-2 text-[11px] text-[#9AA3AD] italic">
                      Condition: {scene.surfaceConditions}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Right Column: Scene Footprint Swath Inspector */}
        <div className="lg:col-span-6 flex flex-col bg-[#151A20] border border-[#1D242C] rounded-lg overflow-hidden">
          <div className="px-4 py-2.5 bg-[#151A20] border-b border-[#1D242C] flex items-center justify-between text-xs">
            <span className="text-[#9AA3AD] font-medium">Selected Tile Coverage &amp; Swath Overlap</span>
            <span className="font-mono text-[11px] text-[#D8B84C]">{selectedScene.tileId}</span>
          </div>

          <div className="relative flex-1 min-h-[380px] bg-[#0b0e14] p-6 flex flex-col justify-between">
            {/* Swath Schematic Diagram */}
            <div className="flex-1 flex items-center justify-center">
              <svg viewBox="0 0 400 300" className="w-full max-w-sm">
                {/* Orbit track vector line */}
                <line x1="280" y1="20" x2="120" y2="280" stroke="#D8B84C" strokeWidth="1.5" strokeDasharray="4 4" />
                <text x="210" y="45" fill="#D8B84C" fontSize="9" fontFamily="monospace">
                  ORBIT DESCENDING TRACK (R{selectedScene.id.split('_')[4] || '019'})
                </text>

                {/* S2 Full Swath Envelope (290 km wide) */}
                <polygon
                  points="180,20 340,20 220,280 60,280"
                  fill="rgba(0, 47, 167, 0.08)"
                  stroke="#D8B84C"
                  strokeWidth="1"
                />

                {/* Specific 100x100 km MGRS Tile Box */}
                <polygon
                  points="160,80 260,80 200,200 100,200"
                  fill="rgba(255, 204, 0, 0.12)"
                  stroke="#D8B84C"
                  strokeWidth="1.5"
                />
                <text x="145" y="145" fill="#D8B84C" fontSize="11" fontFamily="monospace" fontWeight="bold">
                  {selectedScene.tileId} (100×100 km)
                </text>

                {/* User Defined AOI (sub-box inside tile) */}
                <rect
                  x="145"
                  y="120"
                  width="45"
                  height="35"
                  fill="rgba(255, 255, 255, 0.2)"
                  stroke="#ffffff"
                  strokeWidth="1.2"
                />
                <circle cx="167" cy="137" r="2.5" fill="#ffffff" />
                <text x="135" y="170" fill="#ffffff" fontSize="9" fontFamily="monospace">
                  TARGET AOI (100% COVERAGE)
                </text>
              </svg>
            </div>

            {/* Structured Metadata Box */}
            <div className="bg-[#0B0E12] p-3 rounded border border-[#1D242C] space-y-1.5 text-xs">
              <div className="flex justify-between">
                <span className="text-[#9AA3AD]">STAC Item ID:</span>
                <span className="font-mono text-white text-[11px]">{selectedScene.stacItemId}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#9AA3AD]">Coordinate Reference System:</span>
                <span className="font-mono text-white text-[11px]">{selectedScene.crs}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#9AA3AD]">Sensing Datetime:</span>
                <span className="font-mono text-white text-[11px]">{selectedScene.acquisitionDatetime}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#9AA3AD]">Processing Baseline:</span>
                <span className="font-mono text-[#D8B84C] text-[11px]">{selectedScene.processingBaseline} (Radiometric offset calibrated)</span>
              </div>
            </div>
          </div>

          <div className="p-3 bg-[#151A20] border-t border-[#1D242C] flex items-center justify-between text-xs">
            <span className="text-[#9AA3AD]">Ready to configure super-resolution pipeline.</span>
            <button
              onClick={() => onSelectScene(selectedScene)}
              className="px-4 py-1.5 bg-[#D8B84C] text-[#0B0E12] font-semibold text-xs rounded hover:bg-[#E4C96A] transition-colors cursor-pointer"
            >
              Select This Scene
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
