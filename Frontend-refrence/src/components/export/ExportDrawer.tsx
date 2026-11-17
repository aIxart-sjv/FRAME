import React, { useState } from 'react';
import { X, Download, CheckSquare, Square, FileText, Layers, ShieldCheck, Database, CheckCircle2 } from 'lucide-react';
import { Project } from '../../types';

interface ExportDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  project: Project;
}

export const ExportDrawer: React.FC<ExportDrawerProps> = ({ isOpen, onClose, project }) => {
  const [exportSR, setExportSR] = useState<boolean>(true);
  const [exportEvidence, setExportEvidence] = useState<boolean>(true);
  const [exportAnalysis, setExportAnalysis] = useState<boolean>(true);
  const [exportSTAC, setExportSTAC] = useState<boolean>(true);
  const [exportReport, setExportReport] = useState<boolean>(true);

  const [downloadState, setDownloadState] = useState<'idle' | 'preparing' | 'downloaded'>('idle');

  if (!isOpen) return null;

  const handleTriggerExport = () => {
    setDownloadState('preparing');
    setTimeout(() => {
      setDownloadState('downloaded');
      setTimeout(() => {
        setDownloadState('idle');
        onClose();
      }, 2000);
    }, 1500);
  };

  return (
    <div className="fixed inset-0 z-50 overflow-hidden flex justify-end">
      {/* Backdrop */}
      <div
        onClick={onClose}
        className="fixed inset-0 bg-black/60 backdrop-blur-xs transition-opacity"
      />

      {/* Slide-in Drawer Container */}
      <div className="relative w-full max-w-md bg-[#151A20] border-l border-[#1D242C] text-white flex flex-col justify-between z-10 shadow-2xl">
        {/* Drawer Header */}
        <div className="p-6 border-b border-[#1D242C] flex items-center justify-between">
          <div>
            <h2 className="text-base font-bold tracking-tight text-white">Export GIS-Ready Package</h2>
            <p className="text-xs text-[#9AA3AD] mt-0.5 font-mono">{project.name}</p>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded text-[#9AA3AD] hover:text-white hover:bg-[#1D242C] transition-colors cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Export Options Body */}
        <div className="p-6 flex-1 overflow-y-auto space-y-4">
          <p className="text-xs text-[#9AA3AD] leading-relaxed">
            Select GIS-ready datasets and metadata records to bundle into the operational export package.
          </p>

          <div className="space-y-3">
            {/* Option 1: Super-resolved GeoTIFF COG */}
            <div
              onClick={() => setExportSR(!exportSR)}
              className="p-3.5 bg-[#151A20] border border-[#1D242C] hover:border-[#343C45] rounded-lg cursor-pointer flex items-start gap-3 transition-colors"
            >
              <div className="mt-0.5 text-[#D8B84C]">
                {exportSR ? <CheckSquare className="w-4 h-4" /> : <Square className="w-4 h-4 text-[#9AA3AD]" />}
              </div>
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-white">FRAME 2.5m Super-Resolved Raster</span>
                  <span className="text-[10px] font-mono px-1.5 py-0.2 bg-[#1D242C] rounded text-[#D8B84C]">COG GeoTIFF</span>
                </div>
                <p className="text-[11px] text-[#9AA3AD] mt-1 leading-normal">
                  12-Band L2A surface reflectance stack (B01-B09, B8A, B11, B12). Includes CRS ({project.crs}) and Affine transform metadata.
                </p>
              </div>
            </div>

            {/* Option 2: Stability Evidence Layer */}
            <div
              onClick={() => setExportEvidence(!exportEvidence)}
              className="p-3.5 bg-[#151A20] border border-[#1D242C] hover:border-[#343C45] rounded-lg cursor-pointer flex items-start gap-3 transition-colors"
            >
              <div className="mt-0.5 text-[#D8B84C]">
                {exportEvidence ? <CheckSquare className="w-4 h-4" /> : <Square className="w-4 h-4 text-[#9AA3AD]" />}
              </div>
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-white">TTA Model Stability Raster</span>
                  <span className="text-[10px] font-mono px-1.5 py-0.2 bg-[#1D242C] rounded text-[#D8B84C] font-bold">1-Band Float32</span>
                </div>
                <p className="text-[11px] text-[#9AA3AD] mt-1 leading-normal">
                  Pixel-level variance layer generated from 6-pass geometric perturbations. Identifies high-risk boundary cells for quality control.
                </p>
              </div>
            </div>

            {/* Option 3: Downstream Thematic Classification */}
            <div
              onClick={() => setExportAnalysis(!exportAnalysis)}
              className="p-3.5 bg-[#151A20] border border-[#1D242C] hover:border-[#343C45] rounded-lg cursor-pointer flex items-start gap-3 transition-colors"
            >
              <div className="mt-0.5 text-[#D8B84C]">
                {exportAnalysis ? <CheckSquare className="w-4 h-4" /> : <Square className="w-4 h-4 text-[#9AA3AD]" />}
              </div>
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-white">Downstream Land Cover Raster</span>
                  <span className="text-[10px] font-mono px-1.5 py-0.2 bg-[#1D242C] rounded text-[#35C98A]">GeoTIFF Byte</span>
                </div>
                <p className="text-[11px] text-[#9AA3AD] mt-1 leading-normal">
                  5-class thematic land-cover raster (Built-up, Vegetation, Cropland, Water, Bare soil) calibrated from 2.5m features.
                </p>
              </div>
            </div>

            {/* Option 4: STAC Item JSON Metadata */}
            <div
              onClick={() => setExportSTAC(!exportSTAC)}
              className="p-3.5 bg-[#151A20] border border-[#1D242C] hover:border-[#343C45] rounded-lg cursor-pointer flex items-start gap-3 transition-colors"
            >
              <div className="mt-0.5 text-[#D8B84C]">
                {exportSTAC ? <CheckSquare className="w-4 h-4" /> : <Square className="w-4 h-4 text-[#9AA3AD]" />}
              </div>
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-white">STAC Item Specification v1.0.0</span>
                  <span className="text-[10px] font-mono px-1.5 py-0.2 bg-[#1D242C] rounded text-[#9AA3AD]">JSON Record</span>
                </div>
                <p className="text-[11px] text-[#9AA3AD] mt-1 leading-normal">
                  Complete spatio-temporal asset catalogue metadata linking source Sentinel-2 asset, model weights hash, and bounding coordinates.
                </p>
              </div>
            </div>

            {/* Option 5: Scientific Validation Audit Report */}
            <div
              onClick={() => setExportReport(!exportReport)}
              className="p-3.5 bg-[#151A20] border border-[#1D242C] hover:border-[#343C45] rounded-lg cursor-pointer flex items-start gap-3 transition-colors"
            >
              <div className="mt-0.5 text-[#D8B84C]">
                {exportReport ? <CheckSquare className="w-4 h-4" /> : <Square className="w-4 h-4 text-[#9AA3AD]" />}
              </div>
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-white">Validation Audit Report</span>
                  <span className="text-[10px] font-mono px-1.5 py-0.2 bg-[#1D242C] rounded text-[#9AA3AD]">PDF / JSON</span>
                </div>
                <p className="text-[11px] text-[#9AA3AD] mt-1 leading-normal">
                  Comprehensive audit trail with SAM, ERGAS, observation reflectance delta, and OpenSR synthesis evaluation logs.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Drawer Bottom Actions */}
        <div className="p-6 border-t border-[#1D242C] bg-[#151A20] space-y-3">
          <div className="flex items-center justify-between text-xs font-mono text-[#9AA3AD]">
            <span>TOTAL ESTIMATED BUNDLE:</span>
            <span className="text-white">~148 MB (.tar.gz)</span>
          </div>

          <button
            disabled={downloadState !== 'idle'}
            onClick={handleTriggerExport}
            className="w-full py-2.5 bg-[#D8B84C] hover:bg-[#E4C96A] text-[#151A20] font-bold text-xs rounded transition-all flex items-center justify-center gap-2 cursor-pointer shadow-md"
          >
            {downloadState === 'idle' && (
              <>
                <Download className="w-4 h-4" />
                <span>Export Package</span>
              </>
            )}
            {downloadState === 'preparing' && (
              <>
                <span className="inline-block w-3.5 h-3.5 border-2 border-[#151A20] border-t-transparent rounded-full animate-spin" />
                <span>Bundling COGs &amp; Lineage Records...</span>
              </>
            )}
            {downloadState === 'downloaded' && (
              <>
                <CheckCircle2 className="w-4 h-4 text-[#151A20]" />
                <span>Archive Dispatched</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
