import { act, renderHook, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { useFrameSession } from '../useFrameSession'
import { ApiError } from '../../api/types'
import { jobResultFixture, analysisResultFixture, uploadFixture } from '../../test/fixtures'

vi.mock('../../api/client', () => ({
  uploadScene: vi.fn(),
  runSr: vi.fn(),
  runNdviAnalysis: vi.fn(),
}))

import * as api from '../../api/client'

describe('useFrameSession', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('starts idle with no file and no result', () => {
    const { result } = renderHook(() => useFrameSession())
    expect(result.current.file).toBeNull()
    expect(result.current.uploadStatus).toBe('idle')
    expect(result.current.jobStatus).toBe('idle')
    expect(result.current.hasResult).toBe(false)
  })

  it('selecting a file immediately uploads it and surfaces the result', async () => {
    vi.mocked(api.uploadScene).mockResolvedValue(uploadFixture)
    const { result } = renderHook(() => useFrameSession())
    const file = new File(['data'], 'scene.tif')

    act(() => {
      result.current.selectFile(file)
    })

    expect(result.current.uploadStatus).toBe('pending')
    await waitFor(() => expect(result.current.uploadStatus).toBe('success'))
    expect(result.current.uploadResult).toEqual(uploadFixture)
    expect(api.uploadScene).toHaveBeenCalledWith(file, 'raw_digital_number')
  })

  it('surfaces a clean error message when upload fails (4xx)', async () => {
    vi.mocked(api.uploadScene).mockRejectedValue(new ApiError(400, { error: 'unsupported_file', code: 'unsupported_file', detail: 'Unsupported file type.' }))
    const { result } = renderHook(() => useFrameSession())

    act(() => {
      result.current.selectFile(new File(['data'], 'notes.txt'))
    })

    await waitFor(() => expect(result.current.uploadStatus).toBe('error'))
    expect(result.current.uploadError).toBe('Unsupported file type.')
  })

  it('runFrame does nothing without a successful upload (missing-result handling)', async () => {
    const { result } = renderHook(() => useFrameSession())
    await act(async () => {
      await result.current.runFrame()
    })
    expect(api.runSr).not.toHaveBeenCalled()
    expect(result.current.jobStatus).toBe('idle')
  })

  it('runFrame calls the API with the uploaded upload_id and stores the result', async () => {
    vi.mocked(api.uploadScene).mockResolvedValue(uploadFixture)
    vi.mocked(api.runSr).mockResolvedValue(jobResultFixture)
    const { result } = renderHook(() => useFrameSession())

    act(() => {
      result.current.selectFile(new File(['data'], 'scene.tif'))
    })
    await waitFor(() => expect(result.current.uploadStatus).toBe('success'))

    await act(async () => {
      await result.current.runFrame()
    })

    // the selected model (default: the Lite baseline) is always sent explicitly
    expect(api.runSr).toHaveBeenCalledWith(uploadFixture.upload_id, undefined, 'lite')
    expect(result.current.jobResult).toEqual(jobResultFixture)
    expect(result.current.hasResult).toBe(true)
  })

  it('defaults to the Lite model', () => {
    const { result } = renderHook(() => useFrameSession())
    expect(result.current.model).toBe('lite')
  })

  it('runFrame sends the model the user selected', async () => {
    vi.mocked(api.uploadScene).mockResolvedValue(uploadFixture)
    vi.mocked(api.runSr).mockResolvedValue({ ...jobResultFixture, model_id: 'mamba' })
    const { result } = renderHook(() => useFrameSession())

    act(() => {
      result.current.selectFile(new File(['data'], 'scene.tif'))
    })
    await waitFor(() => expect(result.current.uploadStatus).toBe('success'))

    act(() => {
      result.current.setModel('mamba')
    })
    expect(result.current.model).toBe('mamba')

    await act(async () => {
      await result.current.runFrame()
    })
    expect(api.runSr).toHaveBeenCalledWith(uploadFixture.upload_id, undefined, 'mamba')
  })

  it('surfaces the backend message when the selected model is unavailable, without losing the upload', async () => {
    vi.mocked(api.uploadScene).mockResolvedValue(uploadFixture)
    vi.mocked(api.runSr).mockRejectedValue(new ApiError(503, { error: 'model_unavailable', code: 'model_unavailable', detail: 'SEN2SR-Mamba requires a CUDA-capable GPU.' }))
    const { result } = renderHook(() => useFrameSession())

    act(() => {
      result.current.selectFile(new File(['data'], 'scene.tif'))
    })
    await waitFor(() => expect(result.current.uploadStatus).toBe('success'))
    act(() => {
      result.current.setModel('mamba')
    })
    await act(async () => {
      await result.current.runFrame()
    })

    expect(result.current.jobStatus).toBe('error')
    expect(result.current.jobError).toBe('SEN2SR-Mamba requires a CUDA-capable GPU.')
    expect(result.current.uploadResult).toEqual(uploadFixture) // the user can switch model and retry
  })

  it('runNdviAnalysis requires a completed job result', async () => {
    vi.mocked(api.runNdviAnalysis).mockResolvedValue(analysisResultFixture)
    const { result } = renderHook(() => useFrameSession())

    await act(async () => {
      await result.current.runNdviAnalysis()
    })
    expect(api.runNdviAnalysis).not.toHaveBeenCalled()
  })

  it('changing the input scale re-uploads the already-selected file with the new scale', async () => {
    vi.mocked(api.uploadScene).mockResolvedValue(uploadFixture)
    const { result } = renderHook(() => useFrameSession())

    act(() => {
      result.current.selectFile(new File(['data'], 'scene.tif'))
    })
    await waitFor(() => expect(result.current.uploadStatus).toBe('success'))
    expect(api.uploadScene).toHaveBeenCalledTimes(1)

    act(() => {
      result.current.setInputScale('reflectance')
    })
    expect(result.current.inputScale).toBe('reflectance')
    await waitFor(() => expect(api.uploadScene).toHaveBeenCalledTimes(2))
    expect(vi.mocked(api.uploadScene).mock.calls[1][1]).toBe('reflectance')
  })

  it('reset clears every piece of state back to idle', async () => {
    vi.mocked(api.uploadScene).mockResolvedValue(uploadFixture)
    const { result } = renderHook(() => useFrameSession())

    act(() => {
      result.current.selectFile(new File(['data'], 'scene.tif'))
    })
    await waitFor(() => expect(result.current.uploadStatus).toBe('success'))

    act(() => {
      result.current.reset()
    })

    expect(result.current.file).toBeNull()
    expect(result.current.uploadStatus).toBe('idle')
    expect(result.current.uploadResult).toBeNull()
  })
})
