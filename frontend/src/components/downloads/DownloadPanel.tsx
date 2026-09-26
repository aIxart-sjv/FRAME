import { useState } from 'react'
import * as api from '../../api/client'
import { downloadFile, downloadJson } from '../../lib/download'
import type { NDVIAnalysisResponse, SRResultResponse } from '../../api/types'
import './DownloadPanel.css'

interface DownloadPanelProps {
  jobResult: SRResultResponse
  analysisResult: NDVIAnalysisResponse | null
}

interface DownloadEntry {
  key: string
  label: string
  action: () => Promise<void> | void
}

/** Clean, visually secondary download controls -- kept below the imagery
 * per this phase's design direction. Every GeoTIFF download goes through
 * the real API endpoints (frame/api/routes.py); nothing is fabricated
 * client-side except the JSON serialization of a response already
 * received from the API. */
export function DownloadPanel({ jobResult, analysisResult }: DownloadPanelProps) {
  const [error, setError] = useState<string | null>(null)
  const [pendingKey, setPendingKey] = useState<string | null>(null)

  const run = async (entry: DownloadEntry) => {
    setError(null)
    setPendingKey(entry.key)
    try {
      await entry.action()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Download failed.')
    } finally {
      setPendingKey(null)
    }
  }

  const entries: DownloadEntry[] = [
    {
      key: 'sr-geotiff',
      label: 'SR GeoTIFF',
      action: () => downloadFile(api.srDownloadUrl(jobResult.job_id), `sr_${jobResult.job_id}.tif`),
    },
    {
      key: 'uncertainty-geotiff',
      label: 'Stability GeoTIFF (TTA)',
      action: () => downloadFile(api.uncertaintyDownloadUrl(jobResult.job_id), `uncertainty_${jobResult.job_id}.tif`),
    },
    {
      key: 'result-json',
      label: 'Result metadata (JSON)',
      action: () => downloadJson(jobResult, `frame_result_${jobResult.job_id}.json`),
    },
  ]

  if (analysisResult) {
    entries.push(
      {
        key: 'ndvi-json',
        label: 'NDVI analysis (JSON)',
        action: () => downloadJson(analysisResult, `frame_ndvi_${analysisResult.analysis_id}.json`),
      },
      {
        key: 'ndvi-native-geotiff',
        label: 'Native NDVI GeoTIFF',
        action: () => downloadFile(api.ndviDownloadUrl(analysisResult.analysis_id, 'native-ndvi'), `ndvi_native_${analysisResult.analysis_id}.tif`),
      },
      {
        key: 'ndvi-sr-geotiff',
        label: 'SR-derived NDVI GeoTIFF',
        action: () => downloadFile(api.ndviDownloadUrl(analysisResult.analysis_id, 'sr-ndvi'), `ndvi_sr_${analysisResult.analysis_id}.tif`),
      },
    )
  }

  return (
    <div className="download-panel">
      <p className="label">Downloads</p>
      <div className="download-panel__list">
        {entries.map((entry) => (
          <button key={entry.key} type="button" className="download-panel__button" onClick={() => run(entry)} disabled={pendingKey === entry.key} aria-label={`Download ${entry.label}`}>
            <span className="download-panel__glyph" aria-hidden="true">
              ↓
            </span>
            {pendingKey === entry.key ? 'Downloading…' : entry.label}
          </button>
        ))}
      </div>
      {error && (
        <p className="download-panel__error" role="alert">
          {error}
        </p>
      )}
    </div>
  )
}
