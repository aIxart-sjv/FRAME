import React from 'react';
import { Project } from '../../types';
import { DEFAULT_PROJECTS } from '../../data/mockData';
import { FolderOpen, Plus, Calendar, MapPin, CheckCircle2 } from 'lucide-react';

interface ProjectsListProps {
  onOpenProject: (project: Project) => void;
  onNewReconstruction: () => void;
}

export const ProjectsList: React.FC<ProjectsListProps> = ({
  onOpenProject,
  onNewReconstruction,
}) => {
  return (
    <div className="w-full min-h-[calc(100vh-3.5rem)] bg-[#0B0E12] text-[#E8EBEE] p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-6 border-b border-[#343C45] gap-4">
          <div>
            <p className="text-xs font-mono uppercase tracking-wider text-[#9AA3AD] mb-1">Previous Investigations</p>
            <h1 className="text-2xl font-bold tracking-tight text-[#E8EBEE]">Projects &amp; Reconstructions</h1>
            <p className="text-xs text-[#9AA3AD] mt-0.5">
              Access previous super-resolution mapping projects and continue inspection or validation.
            </p>
          </div>
          <button
            onClick={onNewReconstruction}
            className="flex items-center gap-2 px-4 py-2 bg-[#D8B84C] hover:bg-[#E4C96A] active:bg-[#A88B32] text-[#0B0E12] font-semibold text-xs rounded transition-all cursor-pointer shadow-md self-start sm:self-auto"
          >
            <Plus className="w-4 h-4 text-[#0B0E12]" />
            <span>New Reconstruction</span>
          </button>
        </div>

        {/* Project Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {DEFAULT_PROJECTS.map((proj) => (
            <div
              key={proj.id}
              className="p-5 bg-[#151A20] border border-[#343C45] hover:border-[#A88B32] rounded-lg transition-all flex flex-col justify-between"
            >
              <div>
                <div className="flex items-start justify-between gap-3 mb-2">
                  <div>
                    <h2 className="text-sm font-semibold text-[#E8EBEE] tracking-tight">{proj.name}</h2>
                    <span className="text-xs text-[#9AA3AD] flex items-center gap-1.5 mt-0.5">
                      <MapPin className="w-3.5 h-3.5 text-[#D8B84C]" />
                      <span>{proj.location}</span>
                    </span>
                  </div>
                  <span className="inline-flex items-center gap-1 text-[11px] font-mono px-2 py-0.5 bg-[#35C98A]/10 text-[#35C98A] border border-[#35C98A]/30 rounded">
                    <CheckCircle2 className="w-3 h-3" />
                    <span>{proj.status}</span>
                  </span>
                </div>

                <div className="my-3 py-2.5 px-3 bg-[#1D242C] rounded border border-[#343C45] font-mono text-[11px] text-[#9AA3AD] space-y-1">
                  <div className="flex justify-between">
                    <span>Scene Tile:</span>
                    <span className="text-[#E8EBEE]">{proj.tileId} ({proj.crs})</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Acquisition:</span>
                    <span className="text-[#E8EBEE]">{proj.acquisitionDate}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Reference:</span>
                    <span className={proj.referenceAvailable ? 'text-[#35C98A]' : 'text-[#9AA3AD]'}>
                      {proj.referenceAvailable ? 'Paired HR Reference' : 'Unpaired Observation'}
                    </span>
                  </div>
                </div>
              </div>

              <div className="pt-3 border-t border-[#343C45] flex items-center justify-between">
                <span className="text-[11px] text-[#9AA3AD] flex items-center gap-1">
                  <Calendar className="w-3 h-3" />
                  <span>Modified {proj.lastModified}</span>
                </span>
                <button
                  onClick={() => onOpenProject(proj)}
                  className="flex items-center gap-1.5 px-3.5 py-1.5 bg-[#1D242C] hover:bg-[#343C45] hover:text-[#E8EBEE] border border-[#343C45] text-xs font-medium rounded transition-colors cursor-pointer text-[#E8EBEE]"
                >
                  <FolderOpen className="w-3.5 h-3.5 text-[#D8B84C]" />
                  <span>Open Project</span>
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
