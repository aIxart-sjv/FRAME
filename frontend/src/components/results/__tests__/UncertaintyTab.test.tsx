import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { UncertaintyTab } from '../UncertaintyTab'
import { jobResultFixture } from '../../../test/fixtures'

// Raster decoding/canvas rendering is a browser-only concern (jsdom has no
// real canvas 2D context) -- these requests are left pending forever so the
// component renders its synchronous label/explainer content without ever
// resolving into RasterLoadingState's error branch.
vi.mock('../../../api/client', () => ({
  fetchSrGeotiff: vi.fn(() => new Promise(() => {})),
  fetchUncertaintyGeotiff: vi.fn(() => new Promise(() => {})),
}))

describe('UncertaintyTab', () => {
  it('labels the signal a TTA stability / reconstruction-variation diagnostic', () => {
    render(<UncertaintyTab jobResult={jobResultFixture} />)
    expect(screen.getByText('TTA stability — reconstruction-variation diagnostic')).toBeInTheDocument()
  })

  it('states what FRAME\'s own validation found: weakly informative, texture-like, not a reliability score', () => {
    render(<UncertaintyTab jobResult={jobResultFixture} />)
    const text = document.body.textContent ?? ''
    expect(text).toMatch(/only weakly associated with reconstruction error/i)
    expect(text).toMatch(/image texture alone/i)
    expect(text).toMatch(/not as a reliability score/i)
  })

  it('never claims the uncertainty is a calibrated confidence value', () => {
    render(<UncertaintyTab jobResult={jobResultFixture} />)
    const text = document.body.textContent ?? ''
    expect(text).toMatch(/not a calibrated probability of error/i)
    expect(text.toLowerCase()).not.toMatch(/is a calibrated confidence/)
  })

  it('distinguishes itself from LAM', () => {
    render(<UncertaintyTab jobResult={jobResultFixture} />)
    expect(screen.getByText(/LAM/)).toBeInTheDocument()
  })

  it('exposes a map/overlay toggle without a binary good/bad framing', () => {
    render(<UncertaintyTab jobResult={jobResultFixture} />)
    expect(screen.getByRole('button', { name: 'Stability map' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'SR image + overlay' })).toBeInTheDocument()
    expect(screen.queryByText(/good|bad|pass|fail/i)).not.toBeInTheDocument()
  })

  it('renders the real numeric distribution values from the API response, not fabricated ones', () => {
    render(<UncertaintyTab jobResult={jobResultFixture} />)
    expect(screen.getByText(jobResultFixture.uncertainty.scalar_summary.toExponential(3))).toBeInTheDocument()
    expect(screen.getByText(String(jobResultFixture.uncertainty.seed))).toBeInTheDocument()
  })
})
