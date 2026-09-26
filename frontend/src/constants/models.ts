import type { ModelId } from '../api/types'

/** The selectable SR models, in display order. Labels and the short hints are
 * user-facing: they describe what each model is, never the runtime it is built
 * on (no library or kernel names -- see the terminology test). The canonical
 * names the backend records in results come back in `SRResultResponse.model_name`. */
export interface ModelOption {
  id: ModelId
  label: string
  hint: string
}

export const MODEL_OPTIONS: ModelOption[] = [
  { id: 'lite', label: 'SEN2SR-Lite', hint: 'Lightweight baseline. Fast; runs on CPU or GPU.' },
  { id: 'mamba', label: 'SEN2SR-Mamba', hint: 'Larger model. Slower; requires a CUDA GPU on the server.' },
]

export const DEFAULT_MODEL: ModelId = 'lite'
