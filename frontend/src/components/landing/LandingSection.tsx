import { SatelliteField } from '../atmosphere/SatelliteField'
import { PipelineDiagram } from './PipelineDiagram'
import { UploadPanel } from './UploadPanel'
import { InputPreview } from './InputPreview'
import { PipelineProgress } from '../pipeline/PipelineProgress'
import { ModelSelector } from './ModelSelector'
import { ErrorNotice } from '../ui/ErrorNotice'
import { ScientificNotes } from '../notes/ScientificNotes'
import type { HealthResponse } from '../../api/types'
import type { FrameSession } from '../../state/useFrameSession'
import './LandingSection.css'

interface LandingSectionProps {
  session: FrameSession
  /** Latest GET /health result (null while offline/unknown); tells the model selector what is available. */
  health?: HealthResponse | null
}

export function LandingSection({ session, health = null }: LandingSectionProps) {
  const { file, selectFile, inputScale, setInputScale, model, setModel, uploadStatus, uploadResult, uploadError, jobStatus, jobResult, jobError, runFrame } = session

  return (
    <div className="landing-section">
      <div className="landing-section__hero">
        <SatelliteField variant="hero" />
        <div className="landing-section__hero-content">
          <p className="label">SIH 26142 · Earth observation prototype</p>
          <h2 className="landing-section__headline">Sentinel-2 super-resolution on a 2.5 m pixel grid, with its limits shown alongside.</h2>
          <p className="landing-section__tagline">FRAME turns 10 m Sentinel-2 imagery into a georeferenced SR-derived product, and adds a TTA stability diagnostic and a lightweight NDVI demonstration. The diagnostic is uncalibrated and the demonstration is not evidence of a downstream advantage.</p>
          <PipelineDiagram />
        </div>
      </div>

      <div className="landing-section__body panel">
        <div className="landing-section__body-inner">
          <UploadPanel file={file} inputScale={inputScale} onChangeInputScale={setInputScale} uploadStatus={uploadStatus} uploadError={uploadError} onSelectFile={selectFile} />

          {uploadStatus === 'success' && uploadResult && (
            <div className="landing-section__preview">
              <p className="label">Input preview</p>
              <InputPreview upload={uploadResult} />
            </div>
          )}

          {uploadStatus === 'success' && uploadResult && (
            <ModelSelector model={model} onChange={setModel} availability={health?.available_models} disabled={jobStatus === 'pending'} />
          )}

          {uploadStatus === 'success' && uploadResult && (
            <PipelineProgress uploadStatus={uploadStatus} jobStatus={jobStatus} jobResult={jobResult} canRun={uploadStatus === 'success'} onRun={runFrame} />
          )}

          {jobStatus === 'error' && jobError && <ErrorNotice title="FRAME run failed" detail={jobError} onRetry={runFrame} />}
        </div>
        <ScientificNotes />
      </div>
    </div>
  )
}
