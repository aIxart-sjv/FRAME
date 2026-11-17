/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { Header } from './components/layout/Header';
import { LandingPage } from './components/landing/LandingPage';
import { SelectArea } from './components/reconstruct/SelectArea';
import { SelectScene } from './components/reconstruct/SelectScene';
import { SceneConfirmed } from './components/reconstruct/SceneConfirmed';
import { UploadScene } from './components/reconstruct/UploadScene';
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
import { useFrameSession } from './state/useFrameSession';
import type { SceneCandidate, SessionProject } from './api/types';
import { safeRetrievalResolutionM } from './lib/aoi';

type View =
  | 'landing'
  | 'reconstruct-area'
  | 'reconstruct-scene'
  | 'reconstruct-confirm'
  | 'upload'
  | 'configure'
  | 'processing'
  | 'project-workspace'
  | 'projects'
  | 'documentation'
  | 'about-sentinel2'
  | 'validation-docs';

export default function App() {
  const session = useFrameSession();
  const [currentView, setCurrentView] = useState<View>('landing');
  const [workspaceTab, setWorkspaceTab] = useState<'inspect' | 'analyze' | 'validate'>('inspect');
  const [isExportDrawerOpen, setIsExportDrawerOpen] = useState<boolean>(false);
  const [openedFromHistory, setOpenedFromHistory] = useState<boolean>(false);
  const [confirmedScene, setConfirmedScene] = useState<SceneCandidate | null>(null);

  const handleNavigate = (view: string) => {
    setCurrentView(view as View);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  /** The real map-first entry point: draw an AOI, search real Sentinel-2 scenes, pick one. Manual upload
   * (startManualUpload below) remains available as an explicit fallback. */
  const startReconstruction = () => {
    session.reset();
    setOpenedFromHistory(false);
    setConfirmedScene(null);
    handleNavigate('reconstruct-area');
  };

  const startManualUpload = () => {
    session.reset();
    setOpenedFromHistory(false);
    setConfirmedScene(null);
    handleNavigate('upload');
  };

  const handleConfirmScene = (scene: SceneCandidate) => {
    setConfirmedScene(scene);
    handleNavigate('reconstruct-confirm');
  };

  const handleProcessingComplete = () => {
    setWorkspaceTab('inspect');
    handleNavigate('project-workspace');
  };

  const handleOpenProject = (project: SessionProject) => {
    session.openProject(project);
    setOpenedFromHistory(true);
    setWorkspaceTab('inspect');
    handleNavigate('project-workspace');
  };

  return (
    <div className="min-h-screen bg-[#0B0E12] text-[#E8EBEE] flex flex-col font-sans antialiased selection:bg-[#D8B84C] selection:text-[#0B0E12]">
      <Header
        currentView={currentView}
        onNavigate={handleNavigate}
        activeProjectName={session.jobResult ? (session.file?.name ?? session.sessionProjects.find((p) => p.id === session.jobResult!.job_id)?.label ?? session.jobResult.model_name) : undefined}
        activeSceneId={session.jobResult ? session.jobResult.model_name : undefined}
        workspaceTab={workspaceTab}
        onWorkspaceTabChange={setWorkspaceTab}
        onOpenExport={() => setIsExportDrawerOpen(true)}
      />

      <main className="flex-1 flex flex-col">
        {currentView === 'landing' && (
          <LandingPage onStartReconstruction={startReconstruction} onExploreProjects={() => handleNavigate('projects')} onNavigate={handleNavigate} />
        )}

        {currentView === 'reconstruct-area' && (
          <SelectArea
            aoiBbox={session.aoiBbox}
            aoiGeojson={session.aoiGeojson}
            onSetAoi={session.setAoi}
            startDate={session.startDate}
            onChangeStartDate={session.setStartDate}
            endDate={session.endDate}
            onChangeEndDate={session.setEndDate}
            maxCloudCover={session.maxCloudCover}
            onChangeMaxCloudCover={session.setMaxCloudCover}
            searchStatus={session.searchStatus}
            searchError={session.searchError}
            onSearch={session.runAcquisitionSearch}
            onContinue={() => handleNavigate('reconstruct-scene')}
            onCancel={() => handleNavigate('landing')}
            onUseManualUpload={startManualUpload}
          />
        )}

        {currentView === 'reconstruct-scene' && session.aoiBbox && (
          <SelectScene
            aoiBbox={session.aoiBbox}
            aoiGeojson={session.aoiGeojson}
            searchStatus={session.searchStatus}
            searchResult={session.searchResult}
            searchError={session.searchError}
            selectedSceneId={session.selectedSceneId}
            onSelectScene={session.selectScene}
            onBack={() => handleNavigate('reconstruct-area')}
            onConfirm={handleConfirmScene}
          />
        )}

        {currentView === 'reconstruct-confirm' && confirmedScene && session.aoiBbox && (
          <SceneConfirmed
            scene={confirmedScene}
            retrieveStatus={session.retrieveStatus}
            retrieveResult={session.retrieveResult}
            retrieveError={session.retrieveError}
            retrieveResolutionM={safeRetrievalResolutionM(session.aoiBbox)}
            onRetrieve={() =>
              session.retrieveSelectedScene(['B02', 'B03', 'B04', 'B08'], safeRetrievalResolutionM(session.aoiBbox!), true)
            }
            onCancelRetrieve={session.cancelRetrieve}
            prepareSrStatus={session.prepareSrStatus}
            prepareSrError={session.prepareSrError}
            onRunSr={session.runSrFromAcquisition}
            onCancelPrepareSr={session.cancelPrepareSr}
            onContinueToConfigure={() => handleNavigate('configure')}
            onStartOver={startReconstruction}
          />
        )}

        {currentView === 'upload' && (
          <UploadScene
            file={session.file}
            inputScale={session.inputScale}
            onChangeInputScale={session.setInputScale}
            onSelectFile={session.selectFile}
            uploadStatus={session.uploadStatus}
            uploadResult={session.uploadResult}
            uploadError={session.uploadError}
            onContinue={() => handleNavigate('configure')}
            onCancel={() => handleNavigate('landing')}
          />
        )}

        {currentView === 'configure' && session.uploadResult && (
          <ReconstructionSetup
            uploadResult={session.uploadResult}
            health={session.health}
            model={session.model}
            onChangeModel={session.setModel}
            onStartProcessing={() => handleNavigate('processing')}
            onBack={() => handleNavigate(session.srSourceProvenance ? 'reconstruct-confirm' : 'upload')}
          />
        )}

        {currentView === 'processing' && session.uploadResult && (
          <ProcessingJob
            uploadResult={session.uploadResult}
            model={session.model}
            jobStatus={session.jobStatus}
            jobResult={session.jobResult}
            jobError={session.jobError}
            onRun={session.runReconstruction}
            onCancel={session.cancelRun}
            onComplete={handleProcessingComplete}
          />
        )}

        {currentView === 'project-workspace' && session.jobResult && (
          <div className="flex-1 flex flex-col">
            {workspaceTab === 'inspect' && (
              <InspectViewer
                file={openedFromHistory ? null : session.file ?? session.acquisitionOriginalFile}
                uploadResult={openedFromHistory ? null : session.uploadResult}
                jobResult={session.jobResult}
                ndviAnalysis={session.analysisResult}
                onNavigateToAnalyze={() => setWorkspaceTab('analyze')}
                onNavigateToValidate={() => setWorkspaceTab('validate')}
              />
            )}
            {workspaceTab === 'analyze' && (
              <AnalyzeWorkspace
                analysisStatus={session.analysisStatus}
                analysisResult={session.analysisResult}
                analysisError={session.analysisError}
                onRunAnalysis={session.runNdviAnalysis}
              />
            )}
            {workspaceTab === 'validate' && <ValidateWorkspace jobResult={session.jobResult} />}
          </div>
        )}

        {currentView === 'projects' && (
          <ProjectsList projects={session.sessionProjects} onOpenProject={handleOpenProject} onNewReconstruction={startReconstruction} />
        )}

        {currentView === 'documentation' && <DocumentationView onBack={() => handleNavigate('landing')} onNavigate={handleNavigate} />}
        {currentView === 'about-sentinel2' && <AboutSentinel2 onBack={() => handleNavigate('landing')} />}
        {currentView === 'validation-docs' && <ValidationDocs onBack={() => handleNavigate('landing')} />}
      </main>

      <ExportDrawer isOpen={isExportDrawerOpen} onClose={() => setIsExportDrawerOpen(false)} jobResult={session.jobResult} ndviAnalysis={session.analysisResult} />
    </div>
  );
}
