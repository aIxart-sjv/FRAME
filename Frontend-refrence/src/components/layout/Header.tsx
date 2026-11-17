import React from 'react';
import { Download, Compass } from 'lucide-react';

interface HeaderProps {
  currentView: string;
  onNavigate: (view: string) => void;
  activeProjectName?: string;
  activeSceneId?: string;
  onOpenExport?: () => void;
  workspaceTab?: 'inspect' | 'analyze' | 'validate';
  onWorkspaceTabChange?: (tab: 'inspect' | 'analyze' | 'validate') => void;
}

export const Header: React.FC<HeaderProps> = ({
  currentView,
  onNavigate,
  activeProjectName,
  activeSceneId,
  onOpenExport,
  workspaceTab = 'inspect',
  onWorkspaceTabChange,
}) => {
  const isInsideWorkspace = currentView.startsWith('project-workspace');

  return (
    <header className="sticky top-0 z-40 w-full h-14 bg-[#0B0E12] border-b border-[#343C45] px-4 md:px-8 flex items-center justify-between">
      {/* Zone 1: Brand title */}
      <div className="flex items-center gap-4">
        <button
          onClick={() => onNavigate('landing')}
          className="flex items-center gap-2 group text-left cursor-pointer focus:outline-none"
        >
          <span className="text-lg font-bold tracking-tight text-[#E8EBEE] group-hover:text-[#D8B84C] transition-colors">
            FRAME
          </span>
          <span className="hidden sm:inline-block text-xs font-mono uppercase text-[#9AA3AD] tracking-wider border-l border-[#343C45] pl-3">
            SIH26142
          </span>
        </button>

        {isInsideWorkspace && activeProjectName && (
          <div className="hidden lg:flex items-center gap-2 pl-4 border-l border-[#343C45] text-xs">
            <span className="text-[#E8EBEE] font-medium truncate max-w-xs">{activeProjectName}</span>
            {activeSceneId && (
              <span className="text-[#9AA3AD] font-mono truncate max-w-xs text-[11px]">
                ({activeSceneId.split('_')[0]}_{activeSceneId.split('_')[5] || 'L2A'})
              </span>
            )}
          </div>
        )}
      </div>

      {/* Zone 2: Navigation Links or Workspace Tabs */}
      {isInsideWorkspace && onWorkspaceTabChange ? (
        <nav className="flex items-center gap-1 bg-[#151A20] p-1 rounded-md border border-[#343C45]" aria-label="Workspace tabs">
          <button
            onClick={() => onWorkspaceTabChange('inspect')}
            className={`px-3 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap cursor-pointer ${
              workspaceTab === 'inspect'
                ? 'bg-[#D8B84C] text-[#0B0E12] font-semibold shadow-sm'
                : 'text-[#9AA3AD] hover:text-[#E8EBEE]'
            }`}
          >
            Inspect
          </button>
          <button
            onClick={() => onWorkspaceTabChange('analyze')}
            className={`px-3 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap cursor-pointer ${
              workspaceTab === 'analyze'
                ? 'bg-[#D8B84C] text-[#0B0E12] font-semibold shadow-sm'
                : 'text-[#9AA3AD] hover:text-[#E8EBEE]'
            }`}
          >
            Analyze
          </button>
          <button
            onClick={() => onWorkspaceTabChange('validate')}
            className={`px-3 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap cursor-pointer ${
              workspaceTab === 'validate'
                ? 'bg-[#D8B84C] text-[#0B0E12] font-semibold shadow-sm'
                : 'text-[#9AA3AD] hover:text-[#E8EBEE]'
            }`}
          >
            Validate
          </button>
        </nav>
      ) : (
        <nav className="hidden md:flex items-center gap-6 text-xs font-medium text-[#9AA3AD]">
          <button
            onClick={() => onNavigate('projects')}
            className={`transition-colors hover:text-[#E8EBEE] cursor-pointer ${
              currentView === 'projects' ? 'text-[#D8B84C] font-semibold' : ''
            }`}
          >
            Projects
          </button>
          <button
            onClick={() => onNavigate('documentation')}
            className={`transition-colors hover:text-[#E8EBEE] cursor-pointer ${
              currentView === 'documentation' ? 'text-[#D8B84C] font-semibold' : ''
            }`}
          >
            Documentation
          </button>
          <button
            onClick={() => onNavigate('about-sentinel2')}
            className={`transition-colors hover:text-[#E8EBEE] cursor-pointer ${
              currentView === 'about-sentinel2' ? 'text-[#D8B84C] font-semibold' : ''
            }`}
          >
            About Sentinel-2
          </button>
          <button
            onClick={() => onNavigate('validation-docs')}
            className={`transition-colors hover:text-[#E8EBEE] cursor-pointer ${
              currentView === 'validation-docs' ? 'text-[#D8B84C] font-semibold' : ''
            }`}
          >
            Validation
          </button>
        </nav>
      )}

      {/* Zone 3: Primary Actions */}
      <div className="flex items-center gap-3">
        {isInsideWorkspace ? (
          <button
            onClick={onOpenExport}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-[#0B0E12] bg-[#D8B84C] hover:bg-[#E4C96A] active:bg-[#A88B32] rounded transition-colors whitespace-nowrap cursor-pointer shadow-sm"
          >
            <Download className="w-3.5 h-3.5 text-[#0B0E12]" />
            <span>Export</span>
          </button>
        ) : (
          <button
            onClick={() => onNavigate('reconstruct-area')}
            className="flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-semibold text-[#0B0E12] bg-[#D8B84C] hover:bg-[#E4C96A] active:bg-[#A88B32] rounded transition-colors whitespace-nowrap cursor-pointer shadow-sm"
          >
            <Compass className="w-3.5 h-3.5 text-[#0B0E12]" />
            <span>Start Reconstruction</span>
          </button>
        )}
      </div>
    </header>
  );
};
