import React, { useState, useRef, useEffect } from 'react';
import { 
  Split, 
  Layers, 
  Square, 
  Crosshair, 
  Ruler, 
  Info, 
  Eye, 
  ShieldAlert, 
  Maximize2,
  CheckCircle2,
  X
} from 'lucide-react';
import { ViewerMode, AnalyticalLayer, RegionOfInterest, PixelSpectralSample } from '../../types';
import { SAMPLE_PIXEL_SPECTRA, SENTINEL2_BANDS } from '../../data/mockData';

interface InspectViewerProps {
  currentROI: RegionOfInterest | null;
  onUpdateROI: (roi: RegionOfInterest | null) => void;
  onNavigateToAnalyze: () => void;
  onNavigateToValidate: () => void;
  projectCrs?: string;
  tileId?: string;
}

export const InspectViewer: React.FC<InspectViewerProps> = ({
  currentROI,
  onUpdateROI,
  onNavigateToAnalyze,
  onNavigateToValidate,
  projectCrs = 'EPSG:32644 (UTM zone 44N)',
  tileId = 'T44RPR',
}) => {
  const [viewerMode, setViewerMode] = useState<ViewerMode>('super-resolved');
  const [analyticalSubLayer, setAnalyticalSubLayer] = useState<AnalyticalLayer>('ndvi');
  const [isSplitCompare, setIsSplitCompare] = useState<boolean>(true);
  const [sliderPosition, setSliderPosition] = useState<number>(50); // percentage 0 - 100
  const [activeTool, setActiveTool] = useState<'pan' | 'roi' | 'pixel-probe' | 'measure'>('pan');
  const [selectedPixelSample, setSelectedPixelSample] = useState<PixelSpectralSample | null>(SAMPLE_PIXEL_SPECTRA.vegetation);
  const [showMetadataDrawer, setShowMetadataDrawer] = useState<boolean>(false);
  const [measurePoints, setMeasurePoints] = useState<Array<{ x: number; y: number }>>([]);
  const [zoomLevel, setZoomLevel] = useState<number>(100);

  const containerRef = useRef<HTMLDivElement | null>(null);
  const isDraggingSlider = useRef<boolean>(false);

  // Handle slider drag interaction
  const handleMouseDown = () => {
    isDraggingSlider.current = true;
  };

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDraggingSlider.current || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
      const percent = (x / rect.width) * 100;
      setSliderPosition(percent);
    };

    const handleMouseUp = () => {
      isDraggingSlider.current = false;
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, []);

  // Handle canvas click depending on active tool
  const handleCanvasClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (activeTool === 'pixel-probe') {
      // Pick simulated spectra based on location
      if (x < rect.width * 0.4 && y > rect.height * 0.5) {
        setSelectedPixelSample(SAMPLE_PIXEL_SPECTRA.waterBody);
      } else if (x > rect.width * 0.6) {
        setSelectedPixelSample(SAMPLE_PIXEL_SPECTRA.builtUpEdge);
      } else {
        setSelectedPixelSample(SAMPLE_PIXEL_SPECTRA.vegetation);
      }
    } else if (activeTool === 'roi') {
      // Create or update ROI
      const newROI: RegionOfInterest = {
        id: 'roi-01',
        name: 'ROI 01 (Inspection Focus)',
        bounds: {
          north: 13.028,
          south: 13.011,
          east: 80.231,
          west: 80.208,
        },
        areaHectares: 48.6,
        createdAt: new Date().toISOString(),
        color: '#D8B84C',
      };
      onUpdateROI(newROI);
    } else if (activeTool === 'measure') {
      if (measurePoints.length >= 2) {
        setMeasurePoints([{ x, y }]);
      } else {
        setMeasurePoints([...measurePoints, { x, y }]);
      }
    }
  };

  return (
    <div className="w-full flex-1 flex flex-col bg-[#0B0E12] text-white overflow-hidden relative">
      {/* Top Workspace Command Bar */}
      <div className="h-12 bg-[#151A20] border-b border-[#1D242C] px-6 flex items-center justify-between z-20">
        {/* Mode Selector (4 Modes: Original, Super-Resolved, Analytical, Evidence) */}
        <div className="flex items-center gap-1 bg-[#0B0E12] p-0.5 rounded border border-[#1D242C]">
          <button
            onClick={() => setViewerMode('original')}
            className={`px-3 py-1 text-xs font-medium rounded transition-colors ${
              viewerMode === 'original'
                ? 'bg-[#1D242C] text-[#D8B84C] font-semibold'
                : 'text-[#9AA3AD] hover:text-white'
            }`}
          >
            1. Original (10m)
          </button>
          <button
            onClick={() => setViewerMode('super-resolved')}
            className={`px-3 py-1 text-xs font-medium rounded transition-colors ${
              viewerMode === 'super-resolved'
                ? 'bg-[#1D242C] text-[#D8B84C] font-semibold'
                : 'text-[#9AA3AD] hover:text-white'
            }`}
          >
            2. Super-resolved (2.5m)
          </button>
          <button
            onClick={() => setViewerMode('analytical')}
            className={`px-3 py-1 text-xs font-medium rounded transition-colors ${
              viewerMode === 'analytical'
                ? 'bg-[#1D242C] text-[#D8B84C] font-semibold'
                : 'text-[#9AA3AD] hover:text-white'
            }`}
          >
            3. Analytical
          </button>
          <button
            onClick={() => setViewerMode('evidence')}
            className={`px-3 py-1 text-xs font-medium rounded transition-colors ${
              viewerMode === 'evidence'
                ? 'bg-[#1D242C] text-[#D8B84C] font-semibold'
                : 'text-[#9AA3AD] hover:text-white'
            }`}
          >
            4. Evidence
          </button>
        </div>

        {/* Analytical Sub-layer Selector when in Analytical mode */}
        {viewerMode === 'analytical' && (
          <div className="hidden md:flex items-center gap-1.5 text-xs">
            <span className="text-[#9AA3AD] text-[11px] font-mono">INDEX:</span>
            {(['ndvi', 'ndbi', 'ndwi', 'bsi'] as AnalyticalLayer[]).map((layer) => (
              <button
                key={layer}
                onClick={() => setAnalyticalSubLayer(layer)}
                className={`px-2.5 py-0.5 uppercase text-[11px] font-mono rounded ${
                  analyticalSubLayer === layer
                    ? 'bg-[#D8B84C] text-[#0B0E12] font-bold'
                    : 'text-[#9AA3AD] hover:text-white bg-[#151A20]'
                }`}
              >
                {layer}
              </button>
            ))}
          </div>
        )}

        {/* Toolbar Affordances */}
        <div className="flex items-center gap-2">
          {/* Split Compare Toggle */}
          <button
            onClick={() => setIsSplitCompare(!isSplitCompare)}
            className={`flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded border cursor-pointer transition-colors ${
              isSplitCompare
                ? 'bg-[#151A20] border-[#D8B84C] text-[#D8B84C]'
                : 'bg-[#151A20] border-[#1D242C] text-[#9AA3AD] hover:text-white'
            }`}
            title="Toggle Split Comparison Slider (10m vs 2.5m)"
          >
            <Split className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Split Compare</span>
          </button>

          {/* Interactive Tools */}
          <div className="flex items-center bg-[#0B0E12] p-0.5 rounded border border-[#1D242C]">
            <button
              onClick={() => setActiveTool('pan')}
              className={`p-1.5 rounded transition-colors cursor-pointer ${
                activeTool === 'pan' ? 'bg-[#1D242C] text-white' : 'text-[#9AA3AD] hover:text-white'
              }`}
              title="Pan Viewport"
            >
              <Eye className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => setActiveTool('roi')}
              className={`p-1.5 rounded transition-colors cursor-pointer ${
                activeTool === 'roi' ? 'bg-[#1D242C] text-[#D8B84C]' : 'text-[#9AA3AD] hover:text-white'
              }`}
              title="Draw Region of Interest (ROI)"
            >
              <Square className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => setActiveTool('pixel-probe')}
              className={`p-1.5 rounded transition-colors cursor-pointer ${
                activeTool === 'pixel-probe' ? 'bg-[#1D242C] text-[#D8B84C]' : 'text-[#9AA3AD] hover:text-white'
              }`}
              title="Pixel Spectral Probe (12-Band Analysis)"
            >
              <Crosshair className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => setActiveTool('measure')}
              className={`p-1.5 rounded transition-colors cursor-pointer ${
                activeTool === 'measure' ? 'bg-[#1D242C] text-[#D8B84C]' : 'text-[#9AA3AD] hover:text-white'
              }`}
              title="Sub-pixel Distance Measurement"
            >
              <Ruler className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Metadata Disclosure Button */}
          <button
            onClick={() => setShowMetadataDrawer(!showMetadataDrawer)}
            className="flex items-center gap-1 px-2 py-1 text-xs text-[#9AA3AD] hover:text-white bg-[#151A20] hover:bg-[#1D242C] border border-[#1D242C] rounded transition-colors cursor-pointer"
          >
            <Info className="w-3.5 h-3.5 text-[#D8B84C]" />
            <span className="hidden md:inline">Metadata</span>
          </button>
        </div>
      </div>

      {/* Main Imagery Viewer Stage */}
      <div
        ref={containerRef}
        onClick={handleCanvasClick}
        className={`relative flex-1 bg-[#0b0e14] overflow-hidden select-none ${
          activeTool === 'pixel-probe'
            ? 'cursor-crosshair'
            : activeTool === 'roi'
            ? 'cursor-cell'
            : activeTool === 'measure'
            ? 'cursor-crosshair'
            : 'cursor-grab'
        }`}
      >
        {/* Render Layer 1: Left / Background Layer */}
        {/* In Split mode: Left side displays Original 10m Sentinel-2 observation */}
        <div className="absolute inset-0 flex items-center justify-center">
          <svg viewBox="0 0 1000 650" className="w-full h-full object-cover">
            <defs>
              {/* Coarse 10m pixel mosaic pattern */}
              <pattern id="pixel-grid-10m" width="20" height="20" patternUnits="userSpaceOnUse">
                <rect width="20" height="20" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="0.5" />
              </pattern>
              {/* Fine 2.5m sub-pixel pattern */}
              <pattern id="pixel-grid-25m" width="5" height="5" patternUnits="userSpaceOnUse">
                <rect width="5" height="5" fill="none" stroke="rgba(255,204,0,0.08)" strokeWidth="0.25" />
              </pattern>
            </defs>

            {/* Base Geographic Imagery Canvas */}
            {/* Native 10m Observation Representation (Blended, aggregated, coarse boundaries) */}
            <g id="layer-original-10m">
              {/* Water reservoir with pixelated stepped shorelines */}
              <path
                d="M 50,650 Q 220,440 280,320 T 480,180 T 680,60 L 1000,0 L 1000,650 Z"
                fill={viewerMode === 'analytical' ? '#1e40af' : '#142a42'}
              />
              {/* Coarse pixelated urban blocks (10m aggregation blur) */}
              <rect x="240" y="240" width="160" height="120" fill={viewerMode === 'analytical' ? '#b91c1c' : '#453a32'} />
              <rect x="420" y="200" width="140" height="140" fill={viewerMode === 'analytical' ? '#991b1b' : '#3f352e'} />
              <rect x="300" y="380" width="220" height="130" fill={viewerMode === 'analytical' ? '#b91c1c' : '#483d36'} />

              {/* Coarse agricultural parcels */}
              <polygon points="120,80 320,80 290,220 90,220" fill={viewerMode === 'analytical' ? '#15803d' : '#1e3828'} />
              <polygon points="340,70 540,60 520,180 320,190" fill={viewerMode === 'analytical' ? '#166534' : '#234230'} />
              <polygon points="560,50 820,30 780,170 540,180" fill={viewerMode === 'analytical' ? '#15803d' : '#1b3425'} />

              {/* Roads / Transit corridors (blurred 10m representation) */}
              <line x1="80" y1="600" x2="680" y2="120" stroke="#52525b" strokeWidth="12" strokeLinecap="square" />
              <line x1="300" y1="620" x2="780" y2="240" stroke="#52525b" strokeWidth="10" strokeLinecap="square" />

              {/* 10m Pixel grid overlay */}
              <rect width="1000" height="650" fill="url(#pixel-grid-10m)" />
            </g>

            {/* In Evidence Mode: Render TTA Stability Perturbation Variance */}
            {viewerMode === 'evidence' && (
              <g id="layer-evidence-stability">
                {/* High stability regions (Deep Blue / Translucent) */}
                <rect width="1000" height="650" fill="rgba(0, 47, 167, 0.25)" />
                {/* Boundary perturbation variance heat regions (Yellow/Amber - requiring scrutiny) */}
                <path
                  d="M 70,605 L 670,125"
                  stroke="#D8B84C"
                  strokeWidth="18"
                  opacity="0.55"
                  filter="blur(4px)"
                />
                <rect
                  x="235"
                  y="235"
                  width="170"
                  height="130"
                  fill="none"
                  stroke="#D8B84C"
                  strokeWidth="8"
                  opacity="0.6"
                  filter="blur(3px)"
                />
                <rect
                  x="415"
                  y="195"
                  width="150"
                  height="150"
                  fill="none"
                  stroke="#D8B84C"
                  strokeWidth="8"
                  opacity="0.6"
                  filter="blur(3px)"
                />
              </g>
            )}

            {/* Split Comparison Mask for Right Side (Super-Resolved 2.5m) */}
            {isSplitCompare && (
              <clipPath id="split-clip">
                <rect x={(sliderPosition / 100) * 1000} y="0" width={1000} height="650" />
              </clipPath>
            )}

            {/* 2.5m Super-Resolved Reconstructed Layer (Sharp edges, sub-pixel structures) */}
            <g clipPath={isSplitCompare ? 'url(#split-clip)' : undefined} id="layer-sr-25m">
              {/* Background fill */}
              <rect width="1000" height="650" fill="#0c1017" />

              {/* Sharpened water body with crisp sub-pixel littoral zone */}
              <path
                d="M 50,650 Q 220,440 280,320 T 480,180 T 680,60 L 1000,0 L 1000,650 Z"
                fill={viewerMode === 'analytical' ? '#1d4ed8' : '#0e243b'}
                stroke="#38bdf8"
                strokeWidth="1.5"
              />

              {/* Crisp individual building footprints and roof geometries at 2.5m */}
              <g fill={viewerMode === 'analytical' ? '#dc2626' : '#57483d'} stroke="#854d0e" strokeWidth="0.75">
                {/* Cluster A */}
                <rect x="250" y="250" width="30" height="24" rx="1" />
                <rect x="290" y="250" width="28" height="22" rx="1" />
                <rect x="330" y="248" width="32" height="26" rx="1" />
                <rect x="255" y="285" width="40" height="25" rx="1" />
                <rect x="305" y="285" width="45" height="28" rx="1" />
                <rect x="252" y="320" width="50" height="24" rx="1" />
                <rect x="312" y="322" width="48" height="26" rx="1" />
                {/* Cluster B */}
                <rect x="430" y="210" width="36" height="32" rx="1" />
                <rect x="480" y="212" width="40" height="30" rx="1" />
                <rect x="435" y="255" width="44" height="34" rx="1" />
                <rect x="490" y="258" width="38" height="30" rx="1" />
                <rect x="440" y="300" width="42" height="28" rx="1" />
                <rect x="492" y="298" width="38" height="26" rx="1" />
              </g>

              {/* Delineated Agricultural Smallholder parcels with sharp hedgerow boundaries */}
              <polygon
                points="120,80 320,80 290,220 90,220"
                fill={viewerMode === 'analytical' ? '#16a34a' : '#224830'}
                stroke="#15803d"
                strokeWidth="1.2"
              />
              <polygon
                points="340,70 540,60 520,180 320,190"
                fill={viewerMode === 'analytical' ? '#15803d' : '#285438'}
                stroke="#15803d"
                strokeWidth="1.2"
              />
              <polygon
                points="560,50 820,30 780,170 540,180"
                fill={viewerMode === 'analytical' ? '#16a34a' : '#1f402c'}
                stroke="#15803d"
                strokeWidth="1.2"
              />

              {/* High-frequency internal parcel furrows and crop rows */}
              <line x1="140" y1="100" x2="300" y2="100" stroke="rgba(255,255,255,0.12)" strokeWidth="1" strokeDasharray="3 3" />
              <line x1="135" y1="120" x2="295" y2="120" stroke="rgba(255,255,255,0.12)" strokeWidth="1" strokeDasharray="3 3" />
              <line x1="130" y1="140" x2="290" y2="140" stroke="rgba(255,255,255,0.12)" strokeWidth="1" strokeDasharray="3 3" />
              <line x1="125" y1="160" x2="285" y2="160" stroke="rgba(255,255,255,0.12)" strokeWidth="1" strokeDasharray="3 3" />

              {/* Crisp dual-lane arterial road with center divider */}
              <line x1="80" y1="600" x2="680" y2="120" stroke="#71717a" strokeWidth="5" />
              <line x1="80" y1="600" x2="680" y2="120" stroke="#ffffff" strokeWidth="0.6" strokeDasharray="6 4" />
              <line x1="300" y1="620" x2="780" y2="240" stroke="#71717a" strokeWidth="4.5" />

              {/* 2.5m Sub-pixel Grid Texture */}
              <rect width="1000" height="650" fill="url(#pixel-grid-25m)" />
            </g>

            {/* Persistent Region of Interest (ROI) Box */}
            {currentROI && (
              <g id="persistent-roi-overlay">
                <rect
                  x="230"
                  y="210"
                  width="340"
                  height="170"
                  fill="rgba(0, 47, 167, 0.08)"
                  stroke="#D8B84C"
                  strokeWidth="2"
                  strokeDasharray="4 2"
                />
                {/* ROI Label Banner */}
                <rect x="230" y="192" width="145" height="18" fill="#0B0E12" rx="2" />
                <text x="236" y="205" fill="#D8B84C" fontSize="10" fontFamily="monospace" fontWeight="bold">
                  {currentROI.name}
                </text>
                {/* Area in Hectares */}
                <rect x="495" y="192" width="75" height="18" fill="#0B0E12" rx="2" />
                <text x="500" y="205" fill="#9AA3AD" fontSize="10" fontFamily="monospace">
                  {currentROI.areaHectares} ha
                </text>
              </g>
            )}

            {/* Measurement Points */}
            {measurePoints.map((pt, i) => (
              <g key={i}>
                <circle cx={pt.x} cy={pt.y} r="4" fill="#D8B84C" />
                <circle cx={pt.x} cy={pt.y} r="8" fill="none" stroke="#D8B84C" strokeWidth="1" />
              </g>
            ))}
            {measurePoints.length === 2 && (
              <line
                x1={measurePoints[0].x}
                y1={measurePoints[0].y}
                x2={measurePoints[1].x}
                y2={measurePoints[1].y}
                stroke="#D8B84C"
                strokeWidth="1.5"
                strokeDasharray="4 2"
              />
            )}
          </svg>
        </div>

        {/* Draggable Split Comparison Divider Line */}
        {isSplitCompare && (
          <div
            style={{ left: `${sliderPosition}%` }}
            className="absolute top-0 bottom-0 w-0.5 bg-[#D8B84C] z-10 pointer-events-none"
          >
            {/* Draggable Center Thumb handle */}
            <div
              onMouseDown={handleMouseDown}
              className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-8 h-8 rounded-full bg-[#151A20] border-2 border-[#D8B84C] shadow-xl flex items-center justify-center cursor-ew-resize pointer-events-auto"
            >
              <Split className="w-3.5 h-3.5 text-[#D8B84C]" />
            </div>

            {/* Side Labels */}
            <div className="absolute top-3 right-3 bg-[#0B0E12]/90 border border-[#1D242C] px-2 py-0.5 rounded text-[11px] font-mono text-white pointer-events-none whitespace-nowrap">
              FRAME SR 2.5m (Reconstructed Grid)
            </div>
            <div className="absolute top-3 left-[-165px] bg-[#0B0E12]/90 border border-[#1D242C] px-2 py-0.5 rounded text-[11px] font-mono text-[#9AA3AD] pointer-events-none whitespace-nowrap">
              Sentinel-2 10m (Original Observation)
            </div>
          </div>
        )}

        {/* Floating Bottom Left Telemetry HUD */}
        <div className="absolute bottom-3 left-3 bg-[#0B0E12]/95 border border-[#1D242C] px-3 py-1.5 rounded text-xs font-mono text-[#9AA3AD] space-y-0.5 pointer-events-none z-10 shadow-lg">
          <div className="flex items-center gap-3">
            <span className="text-white font-semibold">{tileId}</span>
            <span>·</span>
            <span>CRS: {projectCrs.split(' ')[0]}</span>
            <span>·</span>
            <span className="text-[#D8B84C]">PIXEL: 2.5 m (4×)</span>
          </div>
        </div>

        {/* Floating Bottom Right Legend HUD */}
        <div className="absolute bottom-3 right-3 bg-[#0B0E12]/95 border border-[#1D242C] px-3 py-2 rounded text-xs z-10 shadow-lg">
          {viewerMode === 'evidence' ? (
            <div>
              <span className="text-[11px] font-mono text-[#D8B84C] uppercase block mb-1">
                TTA Stability Deviation
              </span>
              <div className="flex items-center gap-2 text-[10px] font-mono text-[#9AA3AD]">
                <div className="w-12 h-2 bg-gradient-to-r from-[#D8B84C] to-[#D8B84C] rounded" />
                <span>0.0 (Stable) → 0.12 (Review)</span>
              </div>
            </div>
          ) : viewerMode === 'analytical' ? (
            <div>
              <span className="text-[11px] font-mono text-[#D8B84C] uppercase block mb-1">
                {analyticalSubLayer.toUpperCase()} Index Palette
              </span>
              <div className="flex items-center gap-2 text-[10px] font-mono text-[#9AA3AD]">
                <div className="w-12 h-2 bg-gradient-to-r from-[#b91c1c] via-[#D8B84C] to-[#16a34a] rounded" />
                <span>-1.0 → +1.0</span>
              </div>
            </div>
          ) : (
            <div className="flex items-center gap-3 text-[11px] font-mono text-[#9AA3AD]">
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-sm bg-[#57483d]" /> Built-up
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-sm bg-[#224830]" /> Canopy
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-sm bg-[#0e243b]" /> Water
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Bottom Panel: Persistent ROI & Pixel Spectral Probe */}
      <div className="border-t border-[#1D242C] bg-[#151A20] p-4 z-20">
        <div className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-12 gap-4 items-center">
          {/* ROI Status & Action */}
          <div className="md:col-span-5 flex items-center justify-between p-3 bg-[#0B0E12] rounded border border-[#1D242C]">
            <div>
              <div className="flex items-center gap-2">
                <Square className="w-3.5 h-3.5 text-[#D8B84C]" />
                <span className="text-xs font-semibold text-white">
                  {currentROI ? currentROI.name : 'No Active ROI Selected'}
                </span>
              </div>
              <span className="text-[11px] font-mono text-[#9AA3AD] block mt-0.5">
                {currentROI
                  ? `Delineated Area: ${currentROI.areaHectares} ha (WGS84 Extent)`
                  : 'Click "Draw ROI" tool above to focus inspection'}
              </span>
            </div>

            {currentROI && (
              <div className="flex items-center gap-2">
                <button
                  onClick={onNavigateToAnalyze}
                  className="px-2.5 py-1 text-xs font-medium text-[#0B0E12] bg-[#D8B84C] hover:bg-[#E4C96A] rounded transition-colors cursor-pointer"
                >
                  Analyze ROI
                </button>
                <button
                  onClick={onNavigateToValidate}
                  className="px-2.5 py-1 text-xs font-medium text-white bg-[#1D242C] hover:bg-[#343C45] rounded transition-colors cursor-pointer"
                >
                  Validate
                </button>
              </div>
            )}
          </div>

          {/* Pixel Spectral Probe (12-Band Reflectance Chart) */}
          <div className="md:col-span-7 p-3 bg-[#0B0E12] rounded border border-[#1D242C]">
            <div className="flex items-center justify-between mb-1.5 text-xs">
              <div className="flex items-center gap-2">
                <Crosshair className="w-3.5 h-3.5 text-[#D8B84C]" />
                <span className="font-semibold text-white">
                  {selectedPixelSample ? selectedPixelSample.landCoverClass : 'Click anywhere on image with Probe tool'}
                </span>
              </div>
              {selectedPixelSample && (
                <div className="flex items-center gap-3 text-[11px] font-mono">
                  <span>NDVI (10m): <strong className="text-white">{selectedPixelSample.ndviOriginal.toFixed(2)}</strong></span>
                  <span>NDVI (2.5m): <strong className="text-[#D8B84C]">{selectedPixelSample.ndviSR.toFixed(2)}</strong></span>
                  <span className="text-[#35C98A]">Stability: {(selectedPixelSample.stabilityScore * 100).toFixed(0)}%</span>
                </div>
              )}
            </div>

            {/* 12-Band Reflectance Bar Chart (B01-B12 excluding B10) */}
            {selectedPixelSample && (
              <div className="flex items-end gap-1 h-12 pt-1 border-t border-[#1D242C]">
                {SENTINEL2_BANDS.filter((b) => b.isAnalyticalL2A).map((band) => {
                  const val10m = selectedPixelSample.original10mBands[band.id] || 0;
                  const val25m = selectedPixelSample.sr25mBands[band.id] || 0;
                  return (
                    <div key={band.id} className="flex-1 flex flex-col items-center gap-0.5 group relative" title={`${band.id} (${band.name}): 10m=${val10m.toFixed(3)}, 2.5m=${val25m.toFixed(3)}`}>
                      <div className="w-full flex items-end justify-center gap-0.5 h-8">
                        <div
                          style={{ height: `${Math.min(100, val10m * 180)}%` }}
                          className="w-1/2 bg-[#555555] rounded-t-xs"
                        />
                        <div
                          style={{ height: `${Math.min(100, val25m * 180)}%` }}
                          className="w-1/2 bg-[#D8B84C] rounded-t-xs"
                        />
                      </div>
                      <span className="text-[9px] font-mono text-[#9AA3AD]">{band.id}</span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Geospatial Metadata Progressive Disclosure Drawer */}
      {showMetadataDrawer && (
        <div className="absolute right-0 top-12 bottom-0 w-80 bg-[#151A20] border-l border-[#1D242C] p-5 z-30 shadow-2xl flex flex-col justify-between overflow-y-auto">
          <div>
            <div className="flex items-center justify-between pb-3 border-b border-[#1D242C] mb-4">
              <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-white">Geospatial Metadata</h3>
              <button
                onClick={() => setShowMetadataDrawer(false)}
                className="text-[#9AA3AD] hover:text-white p-1 rounded hover:bg-[#1D242C] cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-3 text-xs">
              <div>
                <span className="text-[11px] font-mono text-[#9AA3AD] block">Target Resolution Grid</span>
                <span className="font-mono text-[#D8B84C] font-semibold">2.500 m / pixel</span>
                <span className="text-[10px] text-[#9AA3AD] block">Original Sentinel-2: 10m / 20m / 60m</span>
              </div>

              <div>
                <span className="text-[11px] font-mono text-[#9AA3AD] block">CRS (Projected System)</span>
                <span className="font-mono text-white">{projectCrs}</span>
              </div>

              <div>
                <span className="text-[11px] font-mono text-[#9AA3AD] block">Affine Transform Matrix</span>
                <pre className="font-mono text-[10px] text-[#9AA3AD] bg-[#0B0E12] p-2 rounded border border-[#1D242C] mt-1">
                  | 2.50  0.00  389120.0 |{'\n'}
                  | 0.00 -2.50 1442560.0 |{'\n'}
                  | 0.00  0.00       1.0 |
                </pre>
              </div>

              <div>
                <span className="text-[11px] font-mono text-[#9AA3AD] block">Sensor &amp; Instrument</span>
                <span className="text-white">Sentinel-2B MSI Level-2A</span>
              </div>

              <div>
                <span className="text-[11px] font-mono text-[#9AA3AD] block">Radiometric Calibration</span>
                <span className="text-white">BOA surface reflectance</span>
                <span className="text-[10px] text-[#9AA3AD] block">Processing baseline 05.12, offset -1000 DN</span>
              </div>

              <div>
                <span className="text-[11px] font-mono text-[#9AA3AD] block">Dissemination Provenance</span>
                <span className="text-white">Bhoonidhi ISRO / Copernicus STAC</span>
              </div>
            </div>
          </div>

          <div className="pt-4 border-t border-[#1D242C]">
            <p className="text-[10px] text-[#9AA3AD] leading-relaxed">
              Note: 2.5m pixel boundaries represent super-resolved mathematical reconstructions, not directly measured Sentinel-2 ground detections.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
