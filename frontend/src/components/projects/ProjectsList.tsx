/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { Calendar, FolderOpen, Info, Plus } from 'lucide-react';
import type { SessionProject } from '../../api/types';

interface ProjectsListProps {
  projects: SessionProject[];
  onOpenProject: (project: SessionProject) => void;
  onNewReconstruction: () => void;
}

/**
 * The backend keeps no database of past jobs — `frame/api/README.md`'s "Prototype behavior" is explicit that its
 * in-memory job registry is discarded when the process restarts. So "Projects" here is exactly that: this
 * browser tab's own list of reconstructions completed this session, nothing pre-populated or looked up.
 */
export const ProjectsList: React.FC<ProjectsListProps> = ({ projects, onOpenProject, onNewReconstruction }) => {
  return (
    <div className="w-full min-h-[calc(100vh-3.5rem)] bg-[#0B0E12] text-[#E8EBEE] p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-6 border-b border-[#343C45] gap-4">
          <div>
            <p className="text-xs font-mono uppercase tracking-wider text-[#9AA3AD] mb-1">This session</p>
            <h1 className="text-2xl font-bold tracking-tight text-[#E8EBEE]">Reconstructions</h1>
            <p className="text-xs text-[#9AA3AD] mt-0.5 max-w-lg">
              Kept in this browser tab only — the backend has no database of past jobs, so nothing here survives a page reload or a backend restart.
            </p>
          </div>
          <button
            onClick={onNewReconstruction}
            className="flex items-center gap-2 px-4 py-2 bg-[#D8B84C] hover:bg-[#E4C96A] active:bg-[#A88B32] text-[#0B0E12] font-semibold text-xs rounded transition-all cursor-pointer shadow-md self-start sm:self-auto"
          >
            <Plus className="w-4 h-4 text-[#0B0E12]" />
            <span>New reconstruction</span>
          </button>
        </div>

        {projects.length === 0 ? (
          <div className="p-10 text-center bg-[#151A20] border border-[#343C45] rounded-lg">
            <Info className="w-6 h-6 text-[#9AA3AD] mx-auto mb-2" />
            <p className="text-sm text-white font-semibold mb-1">No reconstructions yet this session</p>
            <p className="text-xs text-[#9AA3AD] max-w-sm mx-auto">Start one to see it listed here.</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {projects.map((proj) => (
              <div key={proj.id} className="p-5 bg-[#151A20] border border-[#343C45] hover:border-[#A88B32] rounded-lg transition-all flex flex-col justify-between">
                <div>
                  <div className="flex items-start justify-between gap-3 mb-2">
                    <h2 className="text-sm font-semibold text-[#E8EBEE] tracking-tight truncate">{proj.label}</h2>
                    <span className="inline-flex items-center gap-1 text-[11px] font-mono px-2 py-0.5 bg-[#35C98A]/10 text-[#35C98A] border border-[#35C98A]/30 rounded shrink-0">
                      <span>Ready</span>
                    </span>
                  </div>
                  <div className="my-3 py-2.5 px-3 bg-[#1D242C] rounded border border-[#343C45] font-mono text-[11px] text-[#9AA3AD] space-y-1">
                    <div className="flex justify-between"><span>Model:</span><span className="text-[#E8EBEE]">{proj.result.model_name}</span></div>
                    <div className="flex justify-between"><span>Output:</span><span className="text-[#E8EBEE]">{proj.result.output_shape.join(' × ')}</span></div>
                    <div className="flex justify-between"><span>CRS:</span><span className="text-[#E8EBEE]">{proj.result.crs ?? '—'}</span></div>
                    <div className="flex justify-between"><span>NDVI demonstration:</span><span className={proj.ndviAnalysis ? 'text-[#35C98A]' : 'text-[#9AA3AD]'}>{proj.ndviAnalysis ? 'Run' : 'Not run'}</span></div>
                  </div>
                </div>
                <div className="pt-3 border-t border-[#343C45] flex items-center justify-between">
                  <span className="text-[11px] text-[#9AA3AD] flex items-center gap-1">
                    <Calendar className="w-3 h-3" />
                    <span>{new Date(proj.createdAt).toLocaleString()}</span>
                  </span>
                  <button
                    onClick={() => onOpenProject(proj)}
                    className="flex items-center gap-1.5 px-3.5 py-1.5 bg-[#1D242C] hover:bg-[#343C45] border border-[#343C45] text-xs font-medium rounded transition-colors cursor-pointer text-[#E8EBEE]"
                  >
                    <FolderOpen className="w-3.5 h-3.5 text-[#D8B84C]" />
                    <span>Open</span>
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
