import React from 'react';
import { ArrowRight } from 'lucide-react';
import { HeroVisual } from '../hero/HeroVisual';

interface LandingPageProps {
  onStartReconstruction: () => void;
  onExploreProjects: () => void;
  onNavigate?: (view: string) => void;
}

export const LandingPage: React.FC<LandingPageProps> = ({
  onStartReconstruction,
  onExploreProjects,
}) => {
  return (
    <div className="w-full h-[calc(100vh-3.5rem)] bg-[#0B0E12] text-[#E8EBEE] overflow-hidden relative">
      {/* Full-Viewport Hero Section */}
      <section className="relative w-full h-full flex flex-col justify-start items-start px-8 sm:px-12 md:px-16 lg:px-24 pt-12 sm:pt-16 md:pt-20 lg:pt-24 pointer-events-none">
        {/* Cinematic 3D Earth + Satellite Visualization (Pure visual layer behind UI) */}
        <HeroVisual />

        {/* Hero Content Overlay (Top-Left positioned and left-aligned) */}
        <div className="relative z-10 max-w-2xl text-left pointer-events-auto">
          {/* Kicker with satellite blue accent */}
          <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded bg-[#151A20] border border-[#343C45] mb-4">
            <span className="w-1.5 h-1.5 rounded-full bg-[#456A8A] shadow-[0_0_8px_#456A8A]" />
            <p className="text-xs font-mono tracking-widest text-[#D8B84C] uppercase font-semibold">
              SENTINEL-2 SUPER-RESOLUTION
            </p>
          </div>

          {/* Headline */}
          <h1 className="text-4xl sm:text-5xl md:text-6xl font-bold tracking-tight text-[#E8EBEE] mb-6 leading-tight">
            See beyond the pixel.
          </h1>

          {/* Body */}
          <p className="text-base sm:text-lg text-[#9AA3AD] max-w-xl mb-8 leading-relaxed font-normal">
            FRAME reconstructs finer-scale information from Sentinel-2 imagery and gives you the evidence to evaluate what that reconstruction means for geospatial analysis.
          </p>

          {/* Primary & Secondary CTAs */}
          <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-start gap-4">
            <button
              onClick={onStartReconstruction}
              className="px-6 py-3 text-sm font-semibold text-[#0B0E12] bg-[#D8B84C] hover:bg-[#E4C96A] active:bg-[#A88B32] rounded transition-all flex items-center justify-center gap-2 cursor-pointer shadow-md"
            >
              <span>Start a reconstruction</span>
              <ArrowRight className="w-4 h-4 text-[#0B0E12]" />
            </button>
            <button
              onClick={onExploreProjects}
              className="px-6 py-3 text-sm font-medium text-[#E8EBEE] bg-[#151A20] hover:bg-[#1D242C] border border-[#343C45] rounded transition-all cursor-pointer text-center"
            >
              Explore a project
            </button>
          </div>
        </div>
      </section>
    </div>
  );
};
