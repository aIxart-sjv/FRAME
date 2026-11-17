import React, { useState } from 'react';
import { Search, MapPin, Square, Upload, ArrowRight, CheckCircle2 } from 'lucide-react';

interface SelectAreaProps {
  onContinue: (aoi: {
    name: string;
    coordinates: [number, number];
    bbox: [number, number, number, number];
    method: string;
  }) => void;
  onCancel: () => void;
}

export const SelectArea: React.FC<SelectAreaProps> = ({ onContinue, onCancel }) => {
  const [method, setMethod] = useState<'search' | 'coordinates' | 'draw' | 'upload'>('search');
  const [searchQuery, setSearchQuery] = useState('');
  const [lat, setLat] = useState('13.0125');
  const [lon, setLon] = useState('80.2144');
  const [selectedAOI, setSelectedAOI] = useState<{
    name: string;
    coordinates: [number, number];
    bbox: [number, number, number, number];
    method: string;
  } | null>({
    name: 'Chennai Peri-Urban Corridor, Tamil Nadu',
    coordinates: [80.2144, 13.0125],
    bbox: [80.12, 12.85, 80.32, 13.15],
    method: 'Geographic Preset',
  });

  const [drawActive, setDrawActive] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<string | null>(null);

  // Preset geographic regions (strictly not defaulting to Mumbai)
  const PRESETS = [
    {
      name: 'Chennai Peri-Urban Corridor, Tamil Nadu',
      region: 'India (UTM Zone 44N)',
      coordinates: [80.2144, 13.0125] as [number, number],
      bbox: [80.12, 12.85, 80.32, 13.15] as [number, number, number, number],
      description: 'Rapid urban fringe expansion, mixed built-up & wetlands',
    },
    {
      name: 'Punjab Smallholder Agricultural Parcel, Ludhiana',
      region: 'India (UTM Zone 43N)',
      coordinates: [75.8573, 30.901] as [number, number],
      bbox: [75.65, 30.82, 75.95, 31.08] as [number, number, number, number],
      description: 'Dense field boundaries, winter Rabi wheat parcels',
    },
    {
      name: 'Valencia Wetland & Albufera Basin',
      region: 'Spain (UTM Zone 30N)',
      coordinates: [-0.3421, 39.4215] as [number, number],
      bbox: [-0.45, 39.38, -0.22, 39.58] as [number, number, number, number],
      description: 'OpenSR-Test site, complex water-vegetation interfaces',
    },
    {
      name: 'SEN2NEON Validation Site NEON-SCBI',
      region: 'USA (UTM Zone 18N)',
      coordinates: [-78.145, 38.892] as [number, number],
      bbox: [-78.22, 38.85, -78.08, 38.96] as [number, number, number, number],
      description: 'AVIRIS-NG 2.5m hyperspectral ground reference paired',
    },
  ];

  const handleSelectPreset = (preset: typeof PRESETS[0]) => {
    setSelectedAOI({
      name: preset.name,
      coordinates: preset.coordinates,
      bbox: preset.bbox,
      method: 'Catalogue Area',
    });
  };

  const handleCoordinatesSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const latitude = parseFloat(lat);
    const longitude = parseFloat(lon);
    if (!isNaN(latitude) && !isNaN(longitude)) {
      setSelectedAOI({
        name: `Custom AOI (${latitude.toFixed(4)}°N, ${longitude.toFixed(4)}°E)`,
        coordinates: [longitude, latitude],
        bbox: [longitude - 0.1, latitude - 0.1, longitude + 0.1, latitude + 0.1],
        method: 'Coordinate Input',
      });
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setUploadedFile(file.name);
      setSelectedAOI({
        name: `AOI from ${file.name}`,
        coordinates: [80.21, 13.01],
        bbox: [80.12, 12.85, 80.32, 13.15],
        method: 'GeoJSON Polygon Upload',
      });
    }
  };

  return (
    <div className="w-full min-h-[calc(100vh-3.5rem)] bg-[#0B0E12] text-white flex flex-col">
      {/* Top Header */}
      <div className="border-b border-[#1D242C] px-6 py-6 bg-[#151A20]">
        <div className="max-w-6xl mx-auto flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <p className="text-xs font-mono uppercase tracking-wider text-[#D8B84C] mb-1">Step 1 of 3</p>
            <h1 className="text-2xl font-bold tracking-tight text-white">Where should we look?</h1>
            <p className="text-xs text-[#9AA3AD] mt-0.5">
              Select an area to begin your Sentinel-2 reconstruction.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={onCancel}
              className="px-4 py-2 text-xs font-medium text-[#9AA3AD] hover:text-white bg-[#1D242C] hover:bg-[#343C45] rounded transition-colors cursor-pointer"
            >
              Cancel
            </button>
            <button
              disabled={!selectedAOI}
              onClick={() => selectedAOI && onContinue(selectedAOI)}
              className={`flex items-center gap-2 px-5 py-2 text-xs font-semibold rounded transition-all cursor-pointer ${
                selectedAOI
                  ? 'bg-[#D8B84C] hover:bg-[#E4C96A] text-[#151A20] shadow-md'
                  : 'bg-[#1D242C] text-[#9AA3AD] cursor-not-allowed'
              }`}
            >
              <span>Continue to Scene Selection</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>

      {/* Main Content Layout */}
      <div className="max-w-6xl mx-auto w-full flex-1 p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Selection Methods & Preset Cards */}
        <div className="lg:col-span-5 flex flex-col gap-5">
          {/* Method Tabs */}
          <div className="grid grid-cols-4 gap-1 p-1 bg-[#151A20] border border-[#1D242C] rounded-lg">
            <button
              onClick={() => setMethod('search')}
              className={`flex flex-col items-center justify-center py-2 px-1 text-[11px] font-medium rounded transition-colors ${
                method === 'search' ? 'bg-[#1D242C] text-[#D8B84C]' : 'text-[#9AA3AD] hover:text-white'
              }`}
            >
              <Search className="w-3.5 h-3.5 mb-1" />
              <span>Search</span>
            </button>
            <button
              onClick={() => setMethod('coordinates')}
              className={`flex flex-col items-center justify-center py-2 px-1 text-[11px] font-medium rounded transition-colors ${
                method === 'coordinates' ? 'bg-[#1D242C] text-[#D8B84C]' : 'text-[#9AA3AD] hover:text-white'
              }`}
            >
              <MapPin className="w-3.5 h-3.5 mb-1" />
              <span>Coords</span>
            </button>
            <button
              onClick={() => {
                setMethod('draw');
                setDrawActive(true);
              }}
              className={`flex flex-col items-center justify-center py-2 px-1 text-[11px] font-medium rounded transition-colors ${
                method === 'draw' ? 'bg-[#1D242C] text-[#D8B84C]' : 'text-[#9AA3AD] hover:text-white'
              }`}
            >
              <Square className="w-3.5 h-3.5 mb-1" />
              <span>Draw AOI</span>
            </button>
            <button
              onClick={() => setMethod('upload')}
              className={`flex flex-col items-center justify-center py-2 px-1 text-[11px] font-medium rounded transition-colors ${
                method === 'upload' ? 'bg-[#1D242C] text-[#D8B84C]' : 'text-[#9AA3AD] hover:text-white'
              }`}
            >
              <Upload className="w-3.5 h-3.5 mb-1" />
              <span>Upload</span>
            </button>
          </div>

          {/* Form based on selected method */}
          <div className="p-4 bg-[#151A20] border border-[#1D242C] rounded-lg">
            {method === 'search' && (
              <div>
                <label className="text-xs font-medium text-[#9AA3AD] block mb-2">Location Name or Landmark</label>
                <div className="relative">
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="e.g. Chennai Corridor, Ludhiana, Valencia..."
                    className="w-full bg-[#0B0E12] border border-[#343C45] rounded px-3 py-2 text-xs text-white placeholder-[#9AA3AD] focus:outline-none focus:border-[#D8B84C]"
                  />
                  <Search className="w-4 h-4 text-[#9AA3AD] absolute right-3 top-2.5" />
                </div>
                <p className="text-[11px] text-[#9AA3AD] mt-2">
                  Searches global administrative boundaries and Sentinel-2 reference footprints.
                </p>
              </div>
            )}

            {method === 'coordinates' && (
              <form onSubmit={handleCoordinatesSubmit} className="space-y-3">
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-xs font-mono text-[#9AA3AD] block mb-1">Latitude (°N)</label>
                    <input
                      type="number"
                      step="any"
                      value={lat}
                      onChange={(e) => setLat(e.target.value)}
                      className="w-full bg-[#0B0E12] border border-[#343C45] rounded px-3 py-2 text-xs font-mono text-white focus:outline-none focus:border-[#D8B84C]"
                    />
                  </div>
                  <div>
                    <label className="text-xs font-mono text-[#9AA3AD] block mb-1">Longitude (°E)</label>
                    <input
                      type="number"
                      step="any"
                      value={lon}
                      onChange={(e) => setLon(e.target.value)}
                      className="w-full bg-[#0B0E12] border border-[#343C45] rounded px-3 py-2 text-xs font-mono text-white focus:outline-none focus:border-[#D8B84C]"
                    />
                  </div>
                </div>
                <button
                  type="submit"
                  className="w-full py-2 bg-[#1D242C] hover:bg-[#343C45] text-xs font-medium text-white border border-[#343C45] rounded transition-colors cursor-pointer"
                >
                  Locate Point &amp; Generate AOI
                </button>
              </form>
            )}

            {method === 'draw' && (
              <div className="text-xs text-[#9AA3AD] space-y-2">
                <p className="leading-relaxed">
                  Interactive AOI bounding box drawing is activated on the map stage. Click and drag across the viewport to define your area of interest.
                </p>
                <div className="p-2.5 bg-[#0B0E12] rounded border border-[#1D242C] font-mono text-[11px] text-[#D8B84C]">
                  Status: Rectangle tool ready. Maximum recommended tile size: 100 km².
                </div>
              </div>
            )}

            {method === 'upload' && (
              <div className="text-center py-4">
                <input
                  type="file"
                  id="geojson-upload"
                  accept=".geojson,.json,.kml,.zip"
                  onChange={handleFileUpload}
                  className="hidden"
                />
                <label
                  htmlFor="geojson-upload"
                  className="inline-flex flex-col items-center justify-center p-6 border-2 border-dashed border-[#343C45] hover:border-[#D8B84C] rounded-lg cursor-pointer transition-colors w-full"
                >
                  <Upload className="w-6 h-6 text-[#9AA3AD] mb-2" />
                  <span className="text-xs font-medium text-white mb-1">Upload GeoJSON, Shapefile (zip), or KML</span>
                  <span className="text-[11px] text-[#9AA3AD]">Supported CRS: EPSG:4326 or standard UTM zones</span>
                </label>
                {uploadedFile && (
                  <p className="text-xs text-[#D8B84C] font-mono mt-2">Loaded file: {uploadedFile}</p>
                )}
              </div>
            )}
          </div>

          {/* Preset Evaluated Locations */}
          <div>
            <p className="text-xs font-mono uppercase tracking-wider text-[#9AA3AD] mb-3">
              Evaluated Reference Locations
            </p>
            <div className="space-y-2">
              {PRESETS.map((preset) => {
                const isSelected = selectedAOI?.name === preset.name;
                return (
                  <div
                    key={preset.name}
                    onClick={() => handleSelectPreset(preset)}
                    className={`p-3 rounded-lg border text-left cursor-pointer transition-all ${
                      isSelected
                        ? 'bg-[#151A20] border-[#D8B84C] ring-1 ring-[#D8B84C]/40'
                        : 'bg-[#151A20] border-[#1D242C] hover:border-[#343C45]'
                    }`}
                  >
                    <div className="flex items-start justify-between">
                      <div>
                        <h4 className="text-xs font-semibold text-white">{preset.name}</h4>
                        <span className="text-[11px] text-[#9AA3AD] block mt-0.5">{preset.region}</span>
                      </div>
                      {isSelected && <CheckCircle2 className="w-4 h-4 text-[#D8B84C] shrink-0" />}
                    </div>
                    <p className="text-[11px] text-[#9AA3AD] mt-1.5 leading-relaxed font-normal">
                      {preset.description}
                    </p>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right Column: Geographic Map Display */}
        <div className="lg:col-span-7 flex flex-col bg-[#151A20] border border-[#1D242C] rounded-lg overflow-hidden">
          {/* Map Top Bar */}
          <div className="px-4 py-2.5 bg-[#151A20] border-b border-[#1D242C] flex items-center justify-between text-xs">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-[#35C98A]" />
              <span className="text-[#9AA3AD] font-medium">Cartographic Viewport</span>
            </div>
            {selectedAOI && (
              <span className="font-mono text-[11px] text-[#9AA3AD]">
                Footprint: [{selectedAOI.bbox.map((v) => v.toFixed(2)).join(', ')}]
              </span>
            )}
          </div>

          {/* Interactive Map Surface */}
          <div className="relative flex-1 min-h-[380px] bg-[#0c1017] overflow-hidden flex items-center justify-center p-4">
            {/* Grid overlay simulating cartographic lines */}
            <div
              className="absolute inset-0 opacity-15 pointer-events-none"
              style={{
                backgroundImage:
                  'linear-gradient(to right, #ffffff 1px, transparent 1px), linear-gradient(to bottom, #ffffff 1px, transparent 1px)',
                backgroundSize: '40px 40px',
              }}
            />

            {/* Stylized Geographic Map Graphic */}
            <svg
              viewBox="0 0 600 400"
              className="w-full h-full max-h-[460px] pointer-events-auto select-none"
            >
              {/* Coastlines / Water Body Elements */}
              <path
                d="M 50,400 Q 180,260 220,180 T 380,80 T 520,20 L 600,0 L 600,400 Z"
                fill="#0c1a2e"
                stroke="#1e3a8a"
                strokeWidth="1.5"
              />
              <path
                d="M 120,400 C 180,310 240,240 310,190 C 380,140 450,110 580,90"
                fill="none"
                stroke="#2563eb"
                strokeWidth="1"
                strokeDasharray="3 3"
              />

              {/* Major arterial road / railway network representation */}
              <line x1="80" y1="360" x2="380" y2="120" stroke="#374151" strokeWidth="1.5" />
              <line x1="200" y1="380" x2="480" y2="190" stroke="#374151" strokeWidth="1" />
              <line x1="320" y1="20" x2="310" y2="380" stroke="#374151" strokeWidth="1" strokeDasharray="4 2" />

              {/* Agricultural parcel mosaic textures */}
              <rect x="120" y="80" width="70" height="50" fill="#14281d" stroke="#1f442d" strokeWidth="0.8" />
              <rect x="195" y="75" width="55" height="55" fill="#183023" stroke="#1f442d" strokeWidth="0.8" />
              <rect x="130" y="135" width="60" height="45" fill="#12251a" stroke="#1f442d" strokeWidth="0.8" />

              {/* Urban density cluster representation */}
              <circle cx="280" cy="220" r="48" fill="#2d2822" stroke="#4b3e2e" strokeWidth="1" opacity="0.6" />
              <circle cx="340" cy="180" r="32" fill="#2a241f" stroke="#4b3e2e" strokeWidth="1" opacity="0.5" />

              {/* Selected AOI Bounding Box */}
              {selectedAOI && (
                <g>
                  <rect
                    x="210"
                    y="140"
                    width="180"
                    height="130"
                    fill="rgba(0, 47, 167, 0.15)"
                    stroke="#D8B84C"
                    strokeWidth="2"
                    strokeDasharray={drawActive ? '5 5' : 'none'}
                  />
                  {/* Corner handles */}
                  <rect x="206" y="136" width="8" height="8" fill="#D8B84C" />
                  <rect x="386" y="136" width="8" height="8" fill="#D8B84C" />
                  <rect x="206" y="266" width="8" height="8" fill="#D8B84C" />
                  <rect x="386" y="266" width="8" height="8" fill="#D8B84C" />

                  {/* Centroid reticle */}
                  <circle cx="300" cy="205" r="4" fill="#D8B84C" />
                  <line x1="290" y1="205" x2="310" y2="205" stroke="#D8B84C" strokeWidth="1" />
                  <line x1="300" y1="195" x2="300" y2="215" stroke="#D8B84C" strokeWidth="1" />

                  {/* Badge */}
                  <rect x="215" y="145" width="125" height="18" fill="#0B0E12" rx="2" />
                  <text x="220" y="158" fill="#D8B84C" fontSize="10" fontFamily="monospace">
                    SELECTED AOI (~120 km²)
                  </text>
                </g>
              )}
            </svg>

            {/* Coordinate Readout HUD */}
            <div className="absolute bottom-3 left-3 bg-[#0B0E12]/90 border border-[#1D242C] px-2.5 py-1 rounded font-mono text-[10px] text-[#9AA3AD]">
              PROJECTION: WGS84 / UTM ZONE 44N · SCALE: 1:50,000
            </div>
          </div>

          {/* Bottom Confirmation Strip */}
          {selectedAOI && (
            <div className="p-3 bg-[#151A20] border-t border-[#1D242C] flex items-center justify-between text-xs">
              <div>
                <span className="text-[#9AA3AD]">Active AOI: </span>
                <span className="font-semibold text-white">{selectedAOI.name}</span>
                <span className="text-[#9AA3AD] text-[11px] ml-2 font-mono">({selectedAOI.method})</span>
              </div>
              <button
                onClick={() => onContinue(selectedAOI)}
                className="px-3 py-1 bg-[#D8B84C] text-[#0B0E12] font-semibold text-xs rounded hover:bg-[#E4C96A] transition-colors cursor-pointer"
              >
                Proceed
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
