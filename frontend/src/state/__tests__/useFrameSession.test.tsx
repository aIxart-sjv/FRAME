import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import * as api from '../../api/client'
import { ApiError } from '../../api/types'
import { acquisitionSearchResponseFixture, sceneCandidateFixture } from '../../test/fixtures'
import { useFrameSession } from '../useFrameSession'

beforeEach(() => {
  // Every test controls its own network behaviour explicitly -- the mount-time health
  // check must never hit a real backend during a unit test.
  vi.spyOn(api, 'fetchHealth').mockResolvedValue({
    status: 'ok',
    frame_version: '0.1.0',
    api_version: '0.1.0',
    model_name: 'SEN2SRLite/NonReference_RGBN_x4',
    default_model: 'lite',
    available_models: [],
  })
})

afterEach(() => {
  vi.restoreAllMocks()
})

const VALID_BBOX: [number, number, number, number] = [80.12, 12.85, 80.32, 13.15]

describe('acquisition search', () => {
  it('refuses to search without a valid AOI, without calling the API', async () => {
    const searchSpy = vi.spyOn(api, 'searchAcquisition')
    const { result } = renderHook(() => useFrameSession())

    await act(async () => {
      await result.current.runAcquisitionSearch()
    })

    expect(searchSpy).not.toHaveBeenCalled()
    expect(result.current.searchStatus).toBe('error')
    expect(result.current.searchError).toMatch(/area of interest/i)
  })

  it('refuses to search without dates, even with a valid AOI', async () => {
    const searchSpy = vi.spyOn(api, 'searchAcquisition')
    const { result } = renderHook(() => useFrameSession())

    act(() => result.current.setAoi(VALID_BBOX))
    await act(async () => {
      await result.current.runAcquisitionSearch()
    })

    expect(searchSpy).not.toHaveBeenCalled()
    expect(result.current.searchError).toMatch(/date/i)
  })

  it('builds the request from AOI/date/cloud state and stores the real response', async () => {
    const searchSpy = vi.spyOn(api, 'searchAcquisition').mockResolvedValue(acquisitionSearchResponseFixture)
    const { result } = renderHook(() => useFrameSession())

    act(() => {
      result.current.setAoi(VALID_BBOX)
      result.current.setStartDate('2024-01-01')
      result.current.setEndDate('2024-04-01')
      result.current.setMaxCloudCover(20)
    })
    await act(async () => {
      await result.current.runAcquisitionSearch()
    })

    expect(searchSpy).toHaveBeenCalledWith({ bbox: VALID_BBOX, start_date: '2024-01-01', end_date: '2024-04-01', max_cloud_cover: 20, max_results: 20 })
    expect(result.current.searchStatus).toBe('success')
    expect(result.current.searchResult).toEqual(acquisitionSearchResponseFixture)
  })

  it('sends aoi_geojson instead of bbox when a polygon AOI was drawn', async () => {
    const geojson = { type: 'Polygon' as const, coordinates: [[[80.1, 13.0], [80.2, 13.1], [80.3, 12.9], [80.1, 13.0]]] }
    const searchSpy = vi.spyOn(api, 'searchAcquisition').mockResolvedValue(acquisitionSearchResponseFixture)
    const { result } = renderHook(() => useFrameSession())

    act(() => {
      result.current.setAoi(VALID_BBOX, geojson)
      result.current.setStartDate('2024-01-01')
      result.current.setEndDate('2024-04-01')
    })
    await act(async () => {
      await result.current.runAcquisitionSearch()
    })

    const sentRequest = searchSpy.mock.calls[0][0]
    expect(sentRequest).toHaveProperty('aoi_geojson', geojson)
    expect(sentRequest).not.toHaveProperty('bbox')
  })

  it('reports a real zero-result search as success, not an error', async () => {
    const empty = { ...acquisitionSearchResponseFixture, scenes: [], scene_count: 0, note: 'no real scenes matched' }
    vi.spyOn(api, 'searchAcquisition').mockResolvedValue(empty)
    const { result } = renderHook(() => useFrameSession())

    act(() => {
      result.current.setAoi(VALID_BBOX)
      result.current.setStartDate('2024-01-01')
      result.current.setEndDate('2024-04-01')
    })
    await act(async () => {
      await result.current.runAcquisitionSearch()
    })

    expect(result.current.searchStatus).toBe('success')
    expect(result.current.searchResult?.scene_count).toBe(0)
  })

  it('surfaces a provider-outage error message from a real ApiError', async () => {
    vi.spyOn(api, 'searchAcquisition').mockRejectedValue(new ApiError(503, { error: 'x', code: 'acquisition_provider_unavailable', detail: 'CDSE STAC search failed' }, 'x'))
    const { result } = renderHook(() => useFrameSession())

    act(() => {
      result.current.setAoi(VALID_BBOX)
      result.current.setStartDate('2024-01-01')
      result.current.setEndDate('2024-04-01')
    })
    await act(async () => {
      await result.current.runAcquisitionSearch()
    })

    expect(result.current.searchStatus).toBe('error')
    expect(result.current.searchError).toBe('CDSE STAC search failed')
  })

  it('clearing the AOI resets any previous search result', async () => {
    vi.spyOn(api, 'searchAcquisition').mockResolvedValue(acquisitionSearchResponseFixture)
    const { result } = renderHook(() => useFrameSession())

    act(() => {
      result.current.setAoi(VALID_BBOX)
      result.current.setStartDate('2024-01-01')
      result.current.setEndDate('2024-04-01')
    })
    await act(async () => {
      await result.current.runAcquisitionSearch()
    })
    expect(result.current.searchResult).not.toBeNull()

    act(() => result.current.clearAoi())

    expect(result.current.aoiBbox).toBeNull()
    expect(result.current.searchResult).toBeNull()
    expect(result.current.searchStatus).toBe('idle')
  })
})

describe('scene selection', () => {
  async function withSearchResult() {
    vi.spyOn(api, 'searchAcquisition').mockResolvedValue(acquisitionSearchResponseFixture)
    const hook = renderHook(() => useFrameSession())
    act(() => {
      hook.result.current.setAoi(VALID_BBOX)
      hook.result.current.setStartDate('2024-01-01')
      hook.result.current.setEndDate('2024-04-01')
    })
    await act(async () => {
      await hook.result.current.runAcquisitionSearch()
    })
    return hook
  }

  it('selecting a real scene id resolves the full candidate from the search results', async () => {
    const { result } = await withSearchResult()

    act(() => result.current.selectScene(sceneCandidateFixture.id))

    expect(result.current.selectedSceneId).toBe(sceneCandidateFixture.id)
    expect(result.current.selectedScene).toEqual(sceneCandidateFixture)
  })

  it('selecting an id absent from the results yields no selected scene, not a fabricated one', async () => {
    const { result } = await withSearchResult()

    act(() => result.current.selectScene('not-a-real-scene-id'))

    expect(result.current.selectedScene).toBeNull()
  })

  it('retrieves the selected scene through the real API and stores the result', async () => {
    const { result } = await withSearchResult()
    act(() => result.current.selectScene(sceneCandidateFixture.id))

    const retrieveResponse = {
      retrieval_id: 'retrieval-1',
      search_id: acquisitionSearchResponseFixture.search_id,
      scene_id: sceneCandidateFixture.id,
      bands: ['B02', 'B03', 'B04', 'B08'],
      band_native_resolution_m: {},
      band_sources: {},
      band_radiometric_method: {},
      is_complete_l2a: false,
      validity_mask_coverage: null,
      scl_included: false,
      crs: 'EPSG:32644',
      resolution_m: 10,
      width: 64,
      height: 64,
      retrieval_provider: 'copernicus-cdse-process-api',
      retrieval_endpoint: 'https://sh.dataspace.copernicus.eu/process/v1',
      matched_acquisition_datetime: sceneCandidateFixture.datetime,
      retrieved_at_utc: '2024-03-30T05:00:00Z',
      warnings: [],
      artifacts: { bands_geotiff: '/acquisition/download/retrieval-1/bands' },
      note: 'real reflectance',
    }
    const retrieveSpy = vi.spyOn(api, 'retrieveAcquisition').mockResolvedValue(retrieveResponse)

    await act(async () => {
      await result.current.retrieveSelectedScene(['B02', 'B03', 'B04', 'B08'], 10, false)
    })

    expect(retrieveSpy).toHaveBeenCalledWith(
      {
        search_id: acquisitionSearchResponseFixture.search_id,
        scene_id: sceneCandidateFixture.id,
        bands: ['B02', 'B03', 'B04', 'B08'],
        resolution_m: 10,
        include_scl: false,
      },
      expect.any(AbortSignal),
    )
    await waitFor(() => expect(result.current.retrieveStatus).toBe('success'))
    expect(result.current.retrieveResult).toEqual(retrieveResponse)
  })

  it('cancelling an in-flight retrieval aborts the request and returns to idle, not error', async () => {
    const { result } = await withSearchResult()
    act(() => result.current.selectScene(sceneCandidateFixture.id))

    let capturedSignal: AbortSignal | undefined
    vi.spyOn(api, 'retrieveAcquisition').mockImplementation(
      (_request, signal) =>
        new Promise((_resolve, reject) => {
          capturedSignal = signal
          signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')))
        }),
    )

    let retrievePromise: Promise<void>
    act(() => {
      retrievePromise = result.current.retrieveSelectedScene(['B04'], 10, false)
    })
    await waitFor(() => expect(result.current.retrieveStatus).toBe('pending'))
    expect(capturedSignal?.aborted).toBe(false)

    act(() => result.current.cancelRetrieve())
    await act(async () => {
      await retrievePromise
    })

    expect(capturedSignal?.aborted).toBe(true)
    expect(result.current.retrieveStatus).toBe('idle')
    expect(result.current.retrieveError).toBeNull()
  })
})

const RETRIEVE_RESPONSE_FIXTURE = {
  retrieval_id: 'retrieval-1',
  search_id: acquisitionSearchResponseFixture.search_id,
  scene_id: sceneCandidateFixture.id,
  bands: ['B02', 'B03', 'B04', 'B08'],
  band_native_resolution_m: {},
  band_sources: {},
  band_radiometric_method: {},
  is_complete_l2a: false,
  validity_mask_coverage: null,
  scl_included: false,
  crs: 'EPSG:32644',
  resolution_m: 10,
  width: 64,
  height: 64,
  retrieval_provider: 'copernicus-cdse-process-api',
  retrieval_endpoint: 'https://sh.dataspace.copernicus.eu/process/v1',
  matched_acquisition_datetime: sceneCandidateFixture.datetime,
  retrieved_at_utc: '2024-03-30T05:00:00Z',
  warnings: [],
  artifacts: { bands_geotiff: '/acquisition/download/retrieval-1/bands' },
  note: 'real reflectance',
}

const PREPARED_SR_INPUT_FIXTURE = {
  upload_id: 'upload-from-acquisition-1',
  filename: 'acquisition_scene.tif',
  valid: true,
  band_names: ['B04', 'B03', 'B02', 'B08'],
  width: 64,
  height: 64,
  crs: 'EPSG:32644',
  resolution_m: 10,
  input_scale: 'reflectance' as const,
  validation_messages: ['Upload accepted: required RGBN bands present and real (frame.acquisition).'],
  retrieval_id: 'retrieval-1',
  search_id: acquisitionSearchResponseFixture.search_id,
  scene_id: sceneCandidateFixture.id,
  provider: 'copernicus-cdse-process-api',
  matched_acquisition_datetime: sceneCandidateFixture.datetime,
  excluded_bands: [] as string[],
  note: 'Real B04/B03/B02/B08 reflectance retrieved for this scene, used as-is for the existing RGBN SR pipeline.',
}

describe('SR handoff from a real acquisition retrieval', () => {
  async function withRetrieveResult() {
    vi.spyOn(api, 'searchAcquisition').mockResolvedValue(acquisitionSearchResponseFixture)
    vi.spyOn(api, 'retrieveAcquisition').mockResolvedValue(RETRIEVE_RESPONSE_FIXTURE)
    const hook = renderHook(() => useFrameSession())
    act(() => {
      hook.result.current.setAoi(VALID_BBOX)
      hook.result.current.setStartDate('2024-01-01')
      hook.result.current.setEndDate('2024-04-01')
    })
    await act(async () => {
      await hook.result.current.runAcquisitionSearch()
    })
    act(() => hook.result.current.selectScene(sceneCandidateFixture.id))
    await act(async () => {
      await hook.result.current.retrieveSelectedScene(['B02', 'B03', 'B04', 'B08'], 10, false)
    })
    return hook
  }

  it('does nothing without a completed retrieval', async () => {
    const prepareSpy = vi.spyOn(api, 'prepareSrInputFromAcquisition')
    const { result } = renderHook(() => useFrameSession())

    await act(async () => {
      await result.current.runSrFromAcquisition()
    })

    expect(prepareSpy).not.toHaveBeenCalled()
    expect(result.current.prepareSrStatus).toBe('idle')
  })

  it('populates the SAME uploadResult/uploadStatus state a manual upload fills, plus real provenance', async () => {
    const { result } = await withRetrieveResult()
    const prepareSpy = vi.spyOn(api, 'prepareSrInputFromAcquisition').mockResolvedValue(PREPARED_SR_INPUT_FIXTURE)
    vi.spyOn(api, 'fetchUploadGeotiff').mockResolvedValue(new Blob(['fake-tiff-bytes']))

    await act(async () => {
      await result.current.runSrFromAcquisition()
    })

    expect(prepareSpy).toHaveBeenCalledWith('retrieval-1', expect.any(AbortSignal))
    await waitFor(() => expect(result.current.prepareSrStatus).toBe('success'))
    // this is the exact state slot ReconstructionSetup/ProcessingJob/InspectViewer already read
    expect(result.current.uploadStatus).toBe('success')
    expect(result.current.uploadResult).toEqual({
      upload_id: 'upload-from-acquisition-1',
      filename: 'acquisition_scene.tif',
      valid: true,
      band_names: ['B04', 'B03', 'B02', 'B08'],
      width: 64,
      height: 64,
      crs: 'EPSG:32644',
      resolution_m: 10,
      input_scale: 'reflectance',
      validation_messages: PREPARED_SR_INPUT_FIXTURE.validation_messages,
    })
    expect(result.current.srSourceProvenance).toEqual({
      retrievalId: 'retrieval-1',
      sceneId: sceneCandidateFixture.id,
      provider: 'copernicus-cdse-process-api',
      matchedAcquisitionDatetime: sceneCandidateFixture.datetime,
      excludedBands: [],
    })
    expect(result.current.acquisitionOriginalFile).toBeInstanceOf(File)
    expect(result.current.acquisitionOriginalFile?.name).toBe('acquisition_scene.tif')
  })

  it('reports excluded bands honestly rather than hiding them', async () => {
    const { result } = await withRetrieveResult()
    vi.spyOn(api, 'prepareSrInputFromAcquisition').mockResolvedValue({
      ...PREPARED_SR_INPUT_FIXTURE,
      excluded_bands: ['B11'],
    })
    vi.spyOn(api, 'fetchUploadGeotiff').mockResolvedValue(new Blob(['x']))

    await act(async () => {
      await result.current.runSrFromAcquisition()
    })

    await waitFor(() => expect(result.current.prepareSrStatus).toBe('success'))
    expect(result.current.srSourceProvenance?.excludedBands).toEqual(['B11'])
  })

  it('surfaces a real 422 (insufficient bands) as a named error, without touching uploadResult', async () => {
    const { result } = await withRetrieveResult()
    vi.spyOn(api, 'prepareSrInputFromAcquisition').mockRejectedValue(
      new ApiError(422, { error: 'x', code: 'acquisition_bands_insufficient_for_model', detail: 'missing B03, B02, B08' }, 'x'),
    )

    await act(async () => {
      await result.current.runSrFromAcquisition()
    })

    expect(result.current.prepareSrStatus).toBe('error')
    expect(result.current.prepareSrError).toBe('missing B03, B02, B08')
    expect(result.current.uploadResult).toBeNull()
  })

  it('does not fail the whole hand-off if only the original-file fetch fails', async () => {
    const { result } = await withRetrieveResult()
    vi.spyOn(api, 'prepareSrInputFromAcquisition').mockResolvedValue(PREPARED_SR_INPUT_FIXTURE)
    vi.spyOn(api, 'fetchUploadGeotiff').mockRejectedValue(new ApiError(500, null, 'boom'))

    await act(async () => {
      await result.current.runSrFromAcquisition()
    })

    await waitFor(() => expect(result.current.prepareSrStatus).toBe('success'))
    expect(result.current.uploadStatus).toBe('success')
    expect(result.current.acquisitionOriginalFile).toBeNull()
  })

  it('cancelling an in-flight prepare-sr-input call returns to idle, not error', async () => {
    const { result } = await withRetrieveResult()
    let capturedSignal: AbortSignal | undefined
    vi.spyOn(api, 'prepareSrInputFromAcquisition').mockImplementation(
      (_retrievalId, signal) =>
        new Promise((_resolve, reject) => {
          capturedSignal = signal
          signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')))
        }),
    )

    let runPromise: Promise<void>
    act(() => {
      runPromise = result.current.runSrFromAcquisition()
    })
    await waitFor(() => expect(result.current.prepareSrStatus).toBe('pending'))

    act(() => result.current.cancelPrepareSr())
    await act(async () => {
      await runPromise
    })

    expect(capturedSignal?.aborted).toBe(true)
    expect(result.current.prepareSrStatus).toBe('idle')
    expect(result.current.prepareSrError).toBeNull()
  })
})

describe('reset', () => {
  it('clears all acquisition state', async () => {
    vi.spyOn(api, 'searchAcquisition').mockResolvedValue(acquisitionSearchResponseFixture)
    const { result } = renderHook(() => useFrameSession())

    act(() => {
      result.current.setAoi(VALID_BBOX)
      result.current.setStartDate('2024-01-01')
      result.current.setEndDate('2024-04-01')
    })
    await act(async () => {
      await result.current.runAcquisitionSearch()
    })
    act(() => result.current.selectScene(sceneCandidateFixture.id))

    act(() => result.current.reset())

    expect(result.current.aoiBbox).toBeNull()
    expect(result.current.searchResult).toBeNull()
    expect(result.current.searchStatus).toBe('idle')
    expect(result.current.selectedSceneId).toBeNull()
    expect(result.current.prepareSrStatus).toBe('idle')
    expect(result.current.acquisitionOriginalFile).toBeNull()
    expect(result.current.srSourceProvenance).toBeNull()
  })
})
