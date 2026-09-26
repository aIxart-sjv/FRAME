import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { DownloadPanel } from '../DownloadPanel'
import { analysisResultFixture, jobResultFixture } from '../../../test/fixtures'

vi.mock('../../../lib/download', () => ({
  downloadFile: vi.fn(),
  downloadJson: vi.fn(),
}))

import { downloadFile, downloadJson } from '../../../lib/download'

describe('DownloadPanel', () => {
  it('offers SR GeoTIFF, stability GeoTIFF, and result metadata downloads before any NDVI analysis exists', () => {
    render(<DownloadPanel jobResult={jobResultFixture} analysisResult={null} />)
    expect(screen.getByRole('button', { name: /download sr geotiff/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /download stability geotiff/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /download result metadata/i })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /ndvi/i })).not.toBeInTheDocument()
  })

  it('adds NDVI download actions once an analysis result exists', () => {
    render(<DownloadPanel jobResult={jobResultFixture} analysisResult={analysisResultFixture} />)
    expect(screen.getByRole('button', { name: /download ndvi analysis \(json\)/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /download native ndvi geotiff/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /download sr-derived ndvi geotiff/i })).toBeInTheDocument()
  })

  it('triggers the SR GeoTIFF download with the correct URL and filename', async () => {
    const user = userEvent.setup()
    render(<DownloadPanel jobResult={jobResultFixture} analysisResult={null} />)
    await user.click(screen.getByRole('button', { name: /download sr geotiff/i }))
    await waitFor(() => expect(downloadFile).toHaveBeenCalledWith(expect.stringContaining(`/sr/download/${jobResultFixture.job_id}`), `sr_${jobResultFixture.job_id}.tif`))
  })

  it('serializes the already-fetched result as downloadable JSON, not a recomputation', async () => {
    const user = userEvent.setup()
    render(<DownloadPanel jobResult={jobResultFixture} analysisResult={null} />)
    await user.click(screen.getByRole('button', { name: /download result metadata/i }))
    await waitFor(() => expect(downloadJson).toHaveBeenCalledWith(jobResultFixture, expect.stringContaining(jobResultFixture.job_id)))
  })

  it('shows a clean error message when a download fails, without a stack trace', async () => {
    vi.mocked(downloadFile).mockRejectedValueOnce(new Error('Download failed (HTTP 404). The artifact may no longer be available.'))
    const user = userEvent.setup()
    render(<DownloadPanel jobResult={jobResultFixture} analysisResult={null} />)
    await user.click(screen.getByRole('button', { name: /download sr geotiff/i }))
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('Download failed (HTTP 404). The artifact may no longer be available.'))
    expect(screen.getByRole('alert').textContent).not.toMatch(/Traceback|File "|line \d+, in /)
  })
})
