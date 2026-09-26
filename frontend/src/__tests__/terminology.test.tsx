import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { GROUND_TRUTH_DISCLAIMER, LAM_DISTINCTION, SR_PRODUCT_DESCRIPTION, STATIC_SCIENTIFIC_CAVEATS, UNCERTAINTY_EXPLAINER, UNCERTAINTY_LABEL } from '../constants/terminology'
import { ScientificNotes } from '../components/notes/ScientificNotes'
import { PipelineDiagram } from '../components/landing/PipelineDiagram'
import { ModelSelector } from '../components/landing/ModelSelector'
import { MODEL_OPTIONS } from '../constants/models'

const FORBIDDEN_PHRASES = ['native 2.5 m sentinel-2', 'true 2.5 m image']

function assertNoForbiddenPhrase(text: string) {
  const lower = text.toLowerCase()
  for (const phrase of FORBIDDEN_PHRASES) {
    expect(lower).not.toContain(phrase)
  }
}

describe('scientific terminology constants', () => {
  it('uses the mandated SR product description exactly', () => {
    expect(SR_PRODUCT_DESCRIPTION).toBe('SR-derived product — 2.5 m pixel grid')
  })

  it('labels the stability signal a TTA reconstruction-variation diagnostic, not an uncertainty or confidence', () => {
    expect(UNCERTAINTY_LABEL).toBe('TTA stability — reconstruction-variation diagnostic')
    expect(UNCERTAINTY_LABEL.toLowerCase()).not.toMatch(/confidence|calibrated|probability/)
  })

  it('never uses a forbidden ground-truth phrase in any terminology constant', () => {
    assertNoForbiddenPhrase(SR_PRODUCT_DESCRIPTION)
    assertNoForbiddenPhrase(GROUND_TRUTH_DISCLAIMER)
    assertNoForbiddenPhrase(UNCERTAINTY_LABEL)
    assertNoForbiddenPhrase(UNCERTAINTY_EXPLAINER)
    for (const caveat of STATIC_SCIENTIFIC_CAVEATS) assertNoForbiddenPhrase(caveat)
  })

  it('states Sentinel-2 has never observed the ground at 2.5 m', () => {
    expect(GROUND_TRUTH_DISCLAIMER.toLowerCase()).toContain('never observed the ground at 2.5 m')
    expect(GROUND_TRUTH_DISCLAIMER).toContain('10 m')
  })

  it('explicitly denies calibrated-probability status for the uncertainty signal', () => {
    expect(UNCERTAINTY_EXPLAINER.toLowerCase()).toContain('not a calibrated probability of error')
  })

  it('distinguishes LAM from uncertainty, and only in its own dedicated sentence', () => {
    expect(LAM_DISTINCTION).toContain('LAM')
    expect(LAM_DISTINCTION.toLowerCase()).toContain('uncertainty')
    // LAM must never be conflated into the primary uncertainty label/explainer
    expect(UNCERTAINTY_LABEL.toLowerCase()).not.toContain('lam')
    expect(UNCERTAINTY_EXPLAINER.toLowerCase()).not.toContain('lam')
  })
})

describe('ScientificNotes component', () => {
  it('renders the static caveats with no forbidden phrases when no result exists yet', () => {
    render(<ScientificNotes />)
    for (const caveat of STATIC_SCIENTIFIC_CAVEATS) {
      expect(screen.getByText(caveat)).toBeInTheDocument()
    }
    assertNoForbiddenPhrase(document.body.textContent ?? '')
  })

  it('renders backend-provided caveats verbatim when a result is available', () => {
    const backendCaveats = ['Backend caveat one.', 'Backend caveat two — mentions the SR-derived product — 2.5 m pixel grid.']
    render(<ScientificNotes caveats={backendCaveats} />)
    for (const caveat of backendCaveats) {
      expect(screen.getByText(caveat)).toBeInTheDocument()
    }
  })
})

describe('PipelineDiagram', () => {
  it('communicates the input -> FRAME -> SR-derived -> stability diagnostic -> NDVI demonstration story with correct terminology', () => {
    render(<PipelineDiagram />)
    expect(screen.getByText('10 m Sentinel-2')).toBeInTheDocument()
    expect(screen.getByText('FRAME')).toBeInTheDocument()
    expect(screen.getByText(SR_PRODUCT_DESCRIPTION)).toBeInTheDocument()
    expect(screen.getByText('TTA stability (uncalibrated)')).toBeInTheDocument()
    expect(screen.getByText('NDVI (not a proof)')).toBeInTheDocument()
    assertNoForbiddenPhrase(document.body.textContent ?? '')
  })
})

describe('model selection copy', () => {
  // Users choose "SEN2SR-Mamba"; the libraries it is built on are an implementation detail.
  const RUNTIME_INTERNALS = ['triton', 'mamba_ssm', 'mamba-ssm', 'causal-conv1d', 'causal_conv1d', 'selective_scan']

  it('never names the runtime internals in any model label or hint', () => {
    for (const option of MODEL_OPTIONS) {
      const text = `${option.label} ${option.hint}`.toLowerCase()
      for (const word of RUNTIME_INTERNALS) expect(text).not.toContain(word)
    }
  })

  it('never names the runtime internals in the rendered selector, including the unavailable message', () => {
    render(
      <ModelSelector
        model="lite"
        onChange={() => {}}
        availability={[
          { id: 'lite', label: 'SEN2SR-Lite', model_name: 'x', available: true, reason: null },
          { id: 'mamba', label: 'SEN2SR-Mamba', model_name: 'y', available: false, reason: 'SEN2SR-Mamba requires a CUDA-capable GPU, which is not available.' },
        ]}
      />,
    )
    const text = (document.body.textContent ?? '').toLowerCase()
    for (const word of RUNTIME_INTERNALS) expect(text).not.toContain(word)
    assertNoForbiddenPhrase(text)
  })
})
