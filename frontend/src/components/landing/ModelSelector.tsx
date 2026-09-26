import type { ModelAvailability, ModelId } from '../../api/types'
import { MODEL_OPTIONS } from '../../constants/models'
import './ModelSelector.css'

interface ModelSelectorProps {
  model: ModelId
  onChange: (model: ModelId) => void
  /** From GET /health. `undefined` = not known (older backend, or health not
   * loaded yet): every model stays selectable and the run itself reports any
   * problem clearly, rather than the UI guessing. */
  availability?: ModelAvailability[]
  disabled?: boolean
}

/** Choose which SR model the next run uses. A model the backend reports as
 * unavailable is disabled, with the backend's own user-facing reason shown. */
export function ModelSelector({ model, onChange, availability, disabled = false }: ModelSelectorProps) {
  const entryFor = (id: ModelId) => availability?.find((entry) => entry.id === id)
  const unavailable = MODEL_OPTIONS.map((option) => ({ option, entry: entryFor(option.id) })).filter(({ entry }) => entry && !entry.available)
  const selected = MODEL_OPTIONS.find((option) => option.id === model)

  return (
    <div className="model-selector">
      <div className="model-selector__row" role="group" aria-label="Super-resolution model">
        <span className="label">Model</span>
        <div className="model-selector__options">
          {MODEL_OPTIONS.map((option) => {
            const entry = entryFor(option.id)
            const isUnavailable = entry !== undefined && !entry.available
            return (
              <button
                key={option.id}
                type="button"
                className={model === option.id ? 'is-active' : ''}
                aria-pressed={model === option.id}
                disabled={disabled || isUnavailable}
                onClick={() => onChange(option.id)}
              >
                {option.label}
              </button>
            )
          })}
        </div>
      </div>

      {selected && <p className="model-selector__hint">{selected.hint}</p>}

      {unavailable.map(({ option, entry }) => (
        <p key={option.id} className="model-selector__unavailable" role="status">
          {option.label} is unavailable on this server{entry?.reason ? `: ${entry.reason}` : '.'}
        </p>
      ))}
    </div>
  )
}
