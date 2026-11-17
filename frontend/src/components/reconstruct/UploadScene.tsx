/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useCallback, useState } from 'react';
import { AlertTriangle, ArrowRight, CheckCircle2, FileWarning, Loader2, Upload } from 'lucide-react';
import type { AsyncStatus } from '../../state/useFrameSession';
import type { InputScale, UploadResponse } from '../../api/types';

interface UploadSceneProps {
  file: File | null;
  inputScale: InputScale;
  onChangeInputScale: (scale: InputScale) => void;
  onSelectFile: (file: File | null) => void;
  uploadStatus: AsyncStatus;
  uploadResult: UploadResponse | null;
  uploadError: string | null;
  onContinue: () => void;
  onCancel: () => void;
}

/**
 * The manual fallback ingestion path -- POST /upload of a user-supplied Sentinel-2 L2A GeoTIFF. The primary
 * path is now the real map-first flow (SelectArea -> SelectScene, frame.acquisition, docs/ACQUISITION.md):
 * draw an AOI, search the real Copernicus catalogue, pick a scene. This page remains for when no real scene
 * covers what's needed, or a scene's own real bands are already on disk as a GeoTIFF.
 */
export const UploadScene: React.FC<UploadSceneProps> = ({
  file,
  inputScale,
  onChangeInputScale,
  onSelectFile,
  uploadStatus,
  uploadResult,
  uploadError,
  onContinue,
  onCancel,
}) => {
  const [isDragActive, setIsDragActive] = useState(false);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      const picked = files?.[0];
      if (picked) onSelectFile(picked);
    },
    [onSelectFile],
  );

  return (
    <div className="w-full min-h-[calc(100vh-3.5rem)] bg-[#0B0E12] text-white flex flex-col">
      <div className="border-b border-[#1D242C] px-6 py-6 bg-[#151A20]">
        <div className="max-w-4xl mx-auto flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <p className="text-xs font-mono uppercase tracking-wider text-[#D8B84C] mb-1">Manual upload (fallback)</p>
            <h1 className="text-2xl font-bold tracking-tight text-white">Provide a Sentinel-2 scene</h1>
            <p className="text-xs text-[#9AA3AD] mt-0.5 max-w-xl">
              Supply a Sentinel-2 L2A GeoTIFF with bands B04, B03, B02, B08 directly. Prefer the real map/AOI
              search on the landing page's "Start Reconstruction" for real Copernicus scene discovery.
            </p>
          </div>
          <button
            onClick={onCancel}
            className="px-4 py-2 text-xs font-medium text-[#9AA3AD] hover:text-white bg-[#1D242C] hover:bg-[#343C45] rounded transition-colors cursor-pointer self-start"
          >
            Cancel
          </button>
        </div>
      </div>

      <div className="max-w-4xl mx-auto w-full flex-1 p-6 flex flex-col gap-5">
        {/* Input scale selector */}
        <div className="p-4 bg-[#151A20] border border-[#1D242C] rounded-lg">
          <span className="text-xs font-medium text-[#9AA3AD] block mb-2">Pixel values in this file are…</span>
          <div className="grid grid-cols-2 gap-1 p-1 bg-[#0B0E12] border border-[#1D242C] rounded-lg max-w-md">
            <button
              onClick={() => onChangeInputScale('raw_digital_number')}
              className={`py-2 text-xs font-medium rounded transition-colors cursor-pointer ${
                inputScale === 'raw_digital_number' ? 'bg-[#1D242C] text-[#D8B84C]' : 'text-[#9AA3AD] hover:text-white'
              }`}
            >
              Raw digital number
            </button>
            <button
              onClick={() => onChangeInputScale('reflectance')}
              className={`py-2 text-xs font-medium rounded transition-colors cursor-pointer ${
                inputScale === 'reflectance' ? 'bg-[#1D242C] text-[#D8B84C]' : 'text-[#9AA3AD] hover:text-white'
              }`}
            >
              Reflectance (0–1)
            </button>
          </div>
          <p className="text-[11px] text-[#9AA3AD] mt-2 leading-relaxed">
            Most Sentinel-2 L2A distributions are raw digital numbers. The wrong choice is refused below rather than silently mis-scaled — a reflectance file declared as raw DN would otherwise produce a black image with no error.
          </p>
        </div>

        {/* Drop zone */}
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragActive(true);
          }}
          onDragLeave={() => setIsDragActive(false)}
          onDrop={(e) => {
            e.preventDefault();
            setIsDragActive(false);
            handleFiles(e.dataTransfer.files);
          }}
          className={`p-4 bg-[#151A20] border rounded-lg transition-colors ${isDragActive ? 'border-[#D8B84C]' : 'border-[#1D242C]'}`}
        >
          <input type="file" id="scene-upload" accept=".tif,.tiff" className="hidden" onChange={(e) => handleFiles(e.target.files)} />
          <label
            htmlFor="scene-upload"
            className="flex flex-col items-center justify-center p-8 border-2 border-dashed border-[#343C45] hover:border-[#D8B84C] rounded-lg cursor-pointer transition-colors w-full"
          >
            <Upload className="w-6 h-6 text-[#9AA3AD] mb-2" />
            <span className="text-xs font-medium text-white mb-1">
              {file ? `Selected: ${file.name}` : 'Drop a GeoTIFF here, or click to browse'}
            </span>
            <span className="text-[11px] text-[#9AA3AD]">.tif / .tiff — B04, B03, B02, B08, any height and width, an embedded CRS</span>
          </label>
        </div>

        {/* Upload status */}
        {uploadStatus === 'pending' && (
          <div className="p-4 bg-[#151A20] border border-[#1D242C] rounded-lg flex items-center gap-2.5 text-xs text-[#9AA3AD]">
            <Loader2 className="w-4 h-4 animate-spin text-[#D8B84C]" />
            <span>Validating the file against the backend…</span>
          </div>
        )}

        {uploadStatus === 'error' && uploadError && (
          <div className="p-4 bg-[#151A20] border border-[#D94B4B]/40 rounded-lg flex items-start gap-2.5 text-xs">
            <FileWarning className="w-4 h-4 text-[#D94B4B] shrink-0 mt-0.5" />
            <div>
              <span className="font-semibold text-white block mb-0.5">Upload rejected</span>
              <span className="text-[#9AA3AD] leading-relaxed">{uploadError}</span>
            </div>
          </div>
        )}

        {uploadStatus === 'success' && uploadResult && (
          <div className="p-4 bg-[#151A20] border border-[#35C98A]/30 rounded-lg space-y-3">
            <div className="flex items-center gap-2 text-xs text-[#35C98A] font-semibold">
              <CheckCircle2 className="w-4 h-4" />
              <span>Validated by the backend</span>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              <div>
                <span className="text-[11px] font-mono text-[#9AA3AD] block">Dimensions</span>
                <span className="font-mono text-white">{uploadResult.width} × {uploadResult.height} px</span>
              </div>
              <div>
                <span className="text-[11px] font-mono text-[#9AA3AD] block">Resolution</span>
                <span className="font-mono text-white">{uploadResult.resolution_m ?? '—'} m</span>
              </div>
              <div>
                <span className="text-[11px] font-mono text-[#9AA3AD] block">Bands</span>
                <span className="font-mono text-white">{uploadResult.band_names.join(', ')}</span>
              </div>
              <div>
                <span className="text-[11px] font-mono text-[#9AA3AD] block">CRS</span>
                <span className="font-mono text-white">{uploadResult.crs ?? '—'}</span>
              </div>
            </div>
            {uploadResult.validation_messages.length > 0 && (
              <ul className="text-[11px] text-[#9AA3AD] list-disc pl-4 space-y-0.5">
                {uploadResult.validation_messages.map((m) => (
                  <li key={m}>{m}</li>
                ))}
              </ul>
            )}
          </div>
        )}

        <div className="flex-1" />

        <div className="p-4 bg-[#151A20] border border-[#1D242C] rounded-lg flex items-start gap-2.5 text-[11px] text-[#9AA3AD]">
          <AlertTriangle className="w-3.5 h-3.5 text-[#D8B84C] shrink-0 mt-0.5" />
          <span>Only the four bands above are processed. Uploaded pixels are held by the backend for this session only (an in-memory workspace) and are not persisted after it restarts.</span>
        </div>

        <button
          disabled={uploadStatus !== 'success'}
          onClick={onContinue}
          className={`flex items-center justify-center gap-2 px-5 py-2.5 text-xs font-semibold rounded transition-all cursor-pointer self-end ${
            uploadStatus === 'success' ? 'bg-[#D8B84C] hover:bg-[#E4C96A] text-[#151A20] shadow-md' : 'bg-[#1D242C] text-[#9AA3AD] cursor-not-allowed'
          }`}
        >
          <span>Continue to configuration</span>
          <ArrowRight className="w-3.5 h-3.5" />
        </button>
      </div>
    </div>
  );
};
