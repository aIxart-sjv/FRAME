import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ModelSelector } from '../ModelSelector'
import type { ModelAvailability } from '../../../api/types'

const bothAvailable: ModelAvailability[] = [
  { id: 'lite', label: 'SEN2SR-Lite', model_name: 'SEN2SRLite/NonReference_RGBN_x4', available: true, reason: null },
  { id: 'mamba', label: 'SEN2SR-Mamba', model_name: 'SEN2SR/MambaSR_RGBN_x4', available: true, reason: null },
]

const mambaUnavailable: ModelAvailability[] = [
  bothAvailable[0],
  { ...bothAvailable[1], available: false, reason: 'SEN2SR-Mamba requires a CUDA-capable GPU, which is not available.' },
]

describe('ModelSelector', () => {
  it('offers exactly the two user-facing models', () => {
    render(<ModelSelector model="lite" onChange={vi.fn()} availability={bothAvailable} />)
    expect(screen.getByRole('button', { name: 'SEN2SR-Lite' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'SEN2SR-Mamba' })).toBeInTheDocument()
    expect(screen.getAllByRole('button')).toHaveLength(2)
  })

  it('marks the selected model as pressed', () => {
    const { rerender } = render(<ModelSelector model="lite" onChange={vi.fn()} availability={bothAvailable} />)
    expect(screen.getByRole('button', { name: 'SEN2SR-Lite' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('button', { name: 'SEN2SR-Mamba' })).toHaveAttribute('aria-pressed', 'false')

    rerender(<ModelSelector model="mamba" onChange={vi.fn()} availability={bothAvailable} />)
    expect(screen.getByRole('button', { name: 'SEN2SR-Mamba' })).toHaveAttribute('aria-pressed', 'true')
  })

  it('reports the chosen model id', async () => {
    const onChange = vi.fn()
    render(<ModelSelector model="lite" onChange={onChange} availability={bothAvailable} />)
    await userEvent.setup().click(screen.getByRole('button', { name: 'SEN2SR-Mamba' }))
    expect(onChange).toHaveBeenCalledWith('mamba')
  })

  it('disables an unavailable model and explains why, using the backend reason', async () => {
    const onChange = vi.fn()
    render(<ModelSelector model="lite" onChange={onChange} availability={mambaUnavailable} />)

    const mamba = screen.getByRole('button', { name: 'SEN2SR-Mamba' })
    expect(mamba).toBeDisabled()
    expect(screen.getByRole('status')).toHaveTextContent('SEN2SR-Mamba is unavailable on this server: SEN2SR-Mamba requires a CUDA-capable GPU')

    await userEvent.setup().click(mamba)
    expect(onChange).not.toHaveBeenCalled()
    expect(screen.getByRole('button', { name: 'SEN2SR-Lite' })).toBeEnabled() // the baseline stays usable
  })

  it('keeps every model selectable when availability is unknown (older backend / health not loaded)', () => {
    render(<ModelSelector model="lite" onChange={vi.fn()} />)
    expect(screen.getByRole('button', { name: 'SEN2SR-Lite' })).toBeEnabled()
    expect(screen.getByRole('button', { name: 'SEN2SR-Mamba' })).toBeEnabled()
    expect(screen.queryByRole('status')).not.toBeInTheDocument()
  })

  it('disables selection while a run is in progress', () => {
    render(<ModelSelector model="lite" onChange={vi.fn()} availability={bothAvailable} disabled />)
    for (const button of screen.getAllByRole('button')) expect(button).toBeDisabled()
  })

  it('shows a short hint for the selected model', () => {
    render(<ModelSelector model="mamba" onChange={vi.fn()} availability={bothAvailable} />)
    expect(screen.getByText(/requires a CUDA GPU/i)).toBeInTheDocument()
  })
})
