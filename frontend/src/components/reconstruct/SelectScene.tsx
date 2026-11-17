/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useMemo, useState } from 'react';
import { AlertTriangle, ArrowLeft, ArrowRight, Calendar, CheckCircle2, Cloud, Loader2, SatelliteDish } from 'lucide-react';
import { AcquisitionMap } from './AcquisitionMap';
import type { AoiPolygonGeojson, Bbox } from '../../lib/aoi';
import { formatBbox } from '../../lib/aoi';
import type { AsyncStatus } from '../../state/useFrameSession';
import type { AcquisitionSearchResponse, SceneCandidate } from '../../api/types';

interface SelectSceneProps {
  aoiBbox: Bbox;
  aoiGeojson: AoiPolygonGeojson | null;
  searchStatus: AsyncStatus;
  searchResult: AcquisitionSearchResponse | null;
  searchError: string | null;
  selectedSceneId: string | null;
  onSelectScene: (sceneId: string | null) => void;
  onBack: () => void;
  onConfirm: (scene: SceneCandidate) => void;
}

/**
 * Step 2 of the real map-first acquisition flow: the real scenes `POST /acquisition/search` returned for the
 * AOI/date/cloud query from Step 1, with real footprints on the map. Nothing here is fabricated -- a scene id,
 * date, cloud percentage, tile, or footprint not present in `searchResult` simply is not shown.
 */
export const SelectScene: React.FC<SelectSceneProps> = ({
  aoiBbox,
  aoiGeojson,
  searchStatus,
  searchResult,
  searchError,
  selectedSceneId,
  onSelectScene,
  onBack,
  onConfirm,
}) => {
  const [cloudRefine, setCloudRefine] = useState<number>(100);

  const scenes = searchResult?.scenes ?? [];
  const filteredScenes = useMemo(
    () => scenes.filter((scene) => scene.cloud_cover === null || scene.cloud_cover <= cloudRefine),
    [scenes, cloudRefine],
  );

  const selectedScene = filteredScenes.find((s) => s.id === selectedSceneId) ?? null;

  return (
    <div className="w-full min-h-[calc(100vh-3.5rem)] bg-[#0B0E12] text-white flex flex-col">
      <div className="border-b border-[#1D242C] px-6 py-6 bg-[#151A20]">
        <div className="max-w-6xl mx-auto flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs font-mono uppercase tracking-wider text-[#D8B84C]">Step 2 of 2</span>
              <span className="text-[#343C45]">·</span>
              <span className="text-xs text-[#9AA3AD] font-mono">{formatBbox(aoiBbox, 2)}</span>
            </div>
            <h1 className="text-2xl font-bold tracking-tight text-white">Choose your imagery</h1>
            <p className="text-xs text-[#9AA3AD] mt-0.5">
              Real Sentinel-2 L2A scenes from Copernicus Data Space covering this area and date range.
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
              onClick={() => selectedScene && onConfirm(selectedScene)}
              className={`flex items-center gap-2 px-5 py-2 text-xs font-semibold rounded transition-all cursor-pointer ${
                selectedScene ? 'bg-[#D8B84C] hover:bg-[#E4C96A] text-[#151A20] shadow-md' : 'bg-[#1D242C] text-[#9AA3AD] cursor-not-allowed'
              }`}
            >
              <span>Confirm scene selection</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto w-full flex-1 p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left column: filters + real scene list */}
        <div className="lg:col-span-6 flex flex-col gap-4">
          {searchStatus === 'pending' && (
            <div className="p-6 bg-[#151A20] border border-[#1D242C] rounded-lg flex items-center gap-2.5 text-xs text-[#9AA3AD]">
              <Loader2 className="w-4 h-4 animate-spin text-[#D8B84C]" />
              <span>Searching the real Copernicus Data Space STAC catalogue…</span>
            </div>
          )}

          {searchStatus === 'error' && searchError && (
            <div className="p-4 bg-[#151A20] border border-[#D94B4B]/40 rounded-lg flex items-start gap-2.5 text-xs">
              <AlertTriangle className="w-4 h-4 text-[#D94B4B] shrink-0 mt-0.5" />
              <div>
                <span className="font-semibold text-white block mb-0.5">Search failed</span>
                <span className="text-[#9AA3AD] leading-relaxed">{searchError}</span>
              </div>
            </div>
          )}

          {searchStatus === 'success' && searchResult && (
            <>
              <div className="p-4 bg-[#151A20] border border-[#1D242C] rounded-lg space-y-3">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-[#9AA3AD] font-medium">Real search results</span>
                  <span className="text-[11px] font-mono text-[#D8B84C]">
                    {filteredScenes.length} of {scenes.length} scene{scenes.length === 1 ? '' : 's'}
                  </span>
                </div>
                <div>
                  <div className="flex justify-between text-[11px] mb-1">
                    <span className="text-[#9AA3AD]">Refine by cloud cover</span>
                    <span className="font-mono text-white">{cloudRefine}%</span>
                  </div>
                  <input
                    type="range"
                    min={1}
                    max={100}
                    value={cloudRefine}
                    onChange={(e) => setCloudRefine(Number(e.target.value))}
                    className="w-full accent-[#D8B84C] bg-[#1D242C] h-1.5 rounded cursor-pointer"
                  />
                </div>
                <p className="text-[11px] text-[#9AA3AD] leading-relaxed font-mono">
                  Provider: {searchResult.provider} · {searchResult.stac_endpoint}
                </p>
              </div>

              {scenes.length === 0 && (
                <div className="p-6 bg-[#151A20] border border-[#1D242C] rounded-lg text-center">
                  <SatelliteDish className="w-6 h-6 text-[#9AA3AD] mx-auto mb-2" />
                  <p className="text-xs text-white font-medium mb-1">No real scenes matched</p>
                  <p className="text-[11px] text-[#9AA3AD] leading-relaxed">
                    {searchResult.note ?? 'Try a wider date range, a higher cloud-cover threshold, or a different area.'}
                  </p>
                </div>
              )}

              <div className="space-y-3 overflow-y-auto max-h-[480px] pr-1">
                {filteredScenes.map((scene) => {
                  const isSelected = selectedSceneId === scene.id;
                  return (
                    <div
                      key={scene.id}
                      onClick={() => onSelectScene(scene.id)}
                      className={`p-4 rounded-lg border text-left cursor-pointer transition-all ${
                        isSelected ? 'bg-[#181d26] border-[#D8B84C] ring-1 ring-[#D8B84C]/50' : 'bg-[#151A20] border-[#1D242C] hover:border-[#343C45]'
                      }`}
                    >
                      <div className="flex items-start justify-between gap-3 mb-2">
                        <div className="min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="text-xs font-mono font-bold text-white tracking-wide">{scene.mgrs_tile ?? '—'}</span>
                            <span className="text-[11px] px-1.5 py-0.2 bg-[#1D242C] rounded text-[#9AA3AD] font-mono">L2A</span>
                          </div>
                          <p className="text-[11px] font-mono text-[#9AA3AD] break-all leading-tight">{scene.id}</p>
                        </div>
                        {isSelected && <CheckCircle2 className="w-5 h-5 text-[#D8B84C] shrink-0 mt-0.5" />}
                      </div>
                      <div className="grid grid-cols-3 gap-2 pt-2 border-t border-[#1D242C] text-[11px] font-mono text-[#9AA3AD]">
                        <div className="flex items-center gap-1.5">
                          <Calendar className="w-3.5 h-3.5 text-[#D8B84C]" />
                          <span>{scene.datetime.split('T')[0]}</span>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <Cloud className="w-3.5 h-3.5 text-[#9AA3AD]" />
                          <span>{scene.cloud_cover !== null ? `${scene.cloud_cover.toFixed(1)}% cloud` : 'cloud n/a'}</span>
                        </div>
                        <div className="text-right text-[#9AA3AD] truncate">{scene.platform ?? '—'}</div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </>
          )}
        </div>

        {/* Right column: real map with AOI + real scene footprints */}
        <div className="lg:col-span-6 flex flex-col bg-[#151A20] border border-[#1D242C] rounded-lg overflow-hidden">
          <div className="px-4 py-2.5 bg-[#151A20] border-b border-[#1D242C] flex items-center justify-between text-xs">
            <span className="text-[#9AA3AD] font-medium">Real scene footprints</span>
            {selectedScene && <span className="font-mono text-[11px] text-[#D8B84C]">{selectedScene.mgrs_tile}</span>}
          </div>
          <div className="relative flex-1 min-h-[420px]">
            <AcquisitionMap
              aoiGeojson={aoiGeojson}
              scenes={filteredScenes}
              selectedSceneId={selectedSceneId}
              onSceneClick={onSelectScene}
              drawMode={null}
            />
          </div>

          {selectedScene && (
            <div className="bg-[#0B0E12] p-3 border-t border-[#1D242C] space-y-1.5 text-xs">
              <div className="flex justify-between">
                <span className="text-[#9AA3AD]">STAC item id:</span>
                <span className="font-mono text-white text-[11px] truncate max-w-[60%]">{selectedScene.id}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#9AA3AD]">Provider:</span>
                <span className="font-mono text-white text-[11px]">{selectedScene.provider}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#9AA3AD]">Acquisition datetime:</span>
                <span className="font-mono text-white text-[11px]">{selectedScene.datetime}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#9AA3AD]">Processing baseline:</span>
                <span className="font-mono text-[#D8B84C] text-[11px]">{selectedScene.processing_baseline ?? '—'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#9AA3AD]">Real asset count:</span>
                <span className="font-mono text-white text-[11px]">{selectedScene.asset_count}</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
