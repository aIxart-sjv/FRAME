/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { Header } from './components/layout/Header';
import { LandingPage } from './components/landing/LandingPage';
import { SelectArea } from './components/reconstruct/SelectArea';
import { SelectScene } from './components/reconstruct/SelectScene';
import { ReconstructionSetup } from './components/reconstruct/ReconstructionSetup';
import { ProcessingJob } from './components/reconstruct/ProcessingJob';
import { InspectViewer } from './components/workspace/InspectViewer';
import { AnalyzeWorkspace } from './components/workspace/AnalyzeWorkspace';
import { ValidateWorkspace } from './components/workspace/ValidateWorkspace';
import { ExportDrawer } from './components/export/ExportDrawer';
import { ProjectsList } from './components/projects/ProjectsList';
import { DocumentationView } from './components/docs/DocumentationView';
import { AboutSentinel2 } from './components/docs/AboutSentinel2';
import { ValidationDocs } from './components/docs/ValidationDocs';
import { RegionOfInterest, Project, Sentinel2Scene } from './types';
import { DEFAULT_PROJECTS, AVAILABLE_SCENES } from './data/mockData';

export default function App() {
  // Navigation & View state
  const [currentView, setCurrentView] = useState<string>('landing');
  const [workspaceTab, setWorkspaceTab] = useState<'inspect' | 'analyze' | 'validate'>('inspect');
  const [isExportDrawerOpen, setIsExportDrawerOpen] = useState<boolean>(false);

  // Active Project & Scene context
  const [activeProject, setActiveProject] = useState<Project>(DEFAULT_PROJECTS[0]);
  const [selectedAOI, setSelectedAOI] = useState<{
    name: string;
    coordinates: [number, number];
    bbox: [number, number, number, number];
  }>({
    name: 'Chennai Peri-Urban Corridor, Tamil Nadu',
    coordinates: [80.2144, 13.0125],
    bbox: [80.12, 12.85, 80.32, 13.15],
  });
  const [selectedScene, setSelectedScene] = useState<Sentinel2Scene>(AVAILABLE_SCENES[0]);
  const [jobConfig, setJobConfig] = useState<any>({
    model: 'SEN2SR-Mamba',
    tileSize: 128,
    haloSize: 16,
    enableStabilityEvaluation: true,
    aoiName: 'Chennai Peri-Urban Corridor, Tamil Nadu',
    sceneId: AVAILABLE_SCENES[0].id,
    tileId: AVAILABLE_SCENES[0].tileId,
  });

  // Persistent Region of Interest (ROI) across Inspect, Analyze, and Validate
  const [currentROI, setCurrentROI] = useState<RegionOfInterest | null>({
    id: 'roi-01',
    name: 'ROI 01 (Inspection Focus)',
    bounds: {
      north: 13.028,
      south: 13.011,
      east: 80.231,
      west: 80.208,
    },
    areaHectares: 48.6,
    createdAt: '2026-02-18T10:15:00Z',
    color: '#FFCC00',
  });

  // Navigation handlers
  const handleNavigate = (view: string) => {
    setCurrentView(view);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleStartReconstruction = () => {
    setCurrentView('reconstruct-area');
  };

  const handleExploreProjects = () => {
    setCurrentView('projects');
  };

  const handleAOIContinue = (aoi: any) => {
    setSelectedAOI(aoi);
    setCurrentView('reconstruct-scene');
  };

  const handleSceneSelect = (scene: Sentinel2Scene) => {
    setSelectedScene(scene);
    setCurrentView('reconstruct-setup');
  };

  const handleSetupComplete = (config: any) => {
    setJobConfig({
      ...config,
      aoiName: selectedAOI.name,
      sceneId: selectedScene.id,
      tileId: selectedScene.tileId,
    });
    setCurrentView('reconstruct-processing');
  };

  const handleJobComplete = (newProjectData: any) => {
    const proj: Project = {
      id: `proj-${Date.now()}`,
      name: newProjectData.name,
      location: newProjectData.name.includes('Chennai') ? 'Tamil Nadu, India' : 'Target AOI Footprint',
      sceneId: newProjectData.sceneId,
      tileId: newProjectData.tileId,
      acquisitionDate: '2026-02-18',
      crs: 'EPSG:32644',
      status: 'Ready',
      lastModified: 'Just now',
      referenceAvailable: false,
      metrics: {
        psnr: null,
        ssim: null,
        rmse: null,
        samDegrees: null,
        ergas: null,
        reflectanceBias: -0.0024,
        synthesisScore: 0.019,
        hallucinationRate: null,
        omissionRate: null,
        improvementRate: null,
        crsPreserved: true,
        affineResidualPx: 0.03,
        boundaryDisplacementMeters: 0.52,
        ttaVariance: 0.0031,
        perturbationConsistencyPercent: 96.5,
      },
    };
    setActiveProject(proj);
    setWorkspaceTab('inspect');
    setCurrentView('project-workspace');
  };

  const handleOpenProject = (proj: Project) => {
    setActiveProject(proj);
    setWorkspaceTab('inspect');
    setCurrentView('project-workspace');
  };

  return (
    <div className="min-h-screen bg-[#0B0E12] text-[#E8EBEE] flex flex-col font-sans antialiased selection:bg-[#D8B84C] selection:text-[#0B0E12]">
      {/* Top Header */}
      <Header
        currentView={currentView}
        onNavigate={handleNavigate}
        activeProjectName={activeProject.name}
        activeSceneId={activeProject.sceneId}
        workspaceTab={workspaceTab}
        onWorkspaceTabChange={(tab) => setWorkspaceTab(tab)}
        onOpenExport={() => setIsExportDrawerOpen(true)}
      />

      {/* Main View Router */}
      <main className="flex-1 flex flex-col">
        {currentView === 'landing' && (
          <LandingPage
            onStartReconstruction={handleStartReconstruction}
            onExploreProjects={handleExploreProjects}
            onNavigate={handleNavigate}
          />
        )}

        {currentView === 'reconstruct-area' && (
          <SelectArea
            onContinue={handleAOIContinue}
            onCancel={() => handleNavigate('landing')}
          />
        )}

        {currentView === 'reconstruct-scene' && (
          <SelectScene
            selectedAOI={selectedAOI}
            onSelectScene={handleSceneSelect}
            onBack={() => handleNavigate('reconstruct-area')}
          />
        )}

        {currentView === 'reconstruct-setup' && (
          <ReconstructionSetup
            selectedAOI={selectedAOI}
            selectedScene={selectedScene}
            onStartProcessing={handleSetupComplete}
            onBack={() => handleNavigate('reconstruct-scene')}
          />
        )}

        {currentView === 'reconstruct-processing' && (
          <ProcessingJob
            jobConfig={jobConfig}
            onComplete={handleJobComplete}
            onCancel={() => handleNavigate('reconstruct-setup')}
          />
        )}

        {currentView === 'project-workspace' && (
          <div className="flex-1 flex flex-col">
            {workspaceTab === 'inspect' && (
              <InspectViewer
                currentROI={currentROI}
                onUpdateROI={(roi) => setCurrentROI(roi)}
                onNavigateToAnalyze={() => setWorkspaceTab('analyze')}
                onNavigateToValidate={() => setWorkspaceTab('validate')}
                projectCrs={activeProject.crs}
                tileId={activeProject.tileId}
              />
            )}

            {workspaceTab === 'analyze' && (
              <AnalyzeWorkspace
                currentROI={currentROI}
                onUpdateROI={(roi) => setCurrentROI(roi)}
                tileId={activeProject.tileId}
              />
            )}

            {workspaceTab === 'validate' && (
              <ValidateWorkspace project={activeProject} />
            )}
          </div>
        )}

        {currentView === 'projects' && (
          <ProjectsList
            onOpenProject={handleOpenProject}
            onNewReconstruction={handleStartReconstruction}
          />
        )}

        {currentView === 'documentation' && (
          <DocumentationView
            onBack={() => handleNavigate('landing')}
            onNavigate={handleNavigate}
          />
        )}

        {currentView === 'about-sentinel2' && (
          <AboutSentinel2 onBack={() => handleNavigate('landing')} />
        )}

        {currentView === 'validation-docs' && (
          <ValidationDocs onBack={() => handleNavigate('landing')} />
        )}
      </main>

      {/* Action / Export Drawer */}
      <ExportDrawer
        isOpen={isExportDrawerOpen}
        onClose={() => setIsExportDrawerOpen(false)}
        project={activeProject}
      />
    </div>
  );
}
