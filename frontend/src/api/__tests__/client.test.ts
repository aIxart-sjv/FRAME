import { afterEach, describe, expect, it, vi } from 'vitest'
import * as api from '../client'
import { ApiError } from '../types'
import { acquisitionSearchResponseFixture, sceneCandidateFixture } from '../../test/fixtures'

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? 'OK' : 'Error',
    json: async () => body,
    blob: async () => new Blob([JSON.stringify(body)]),
  } as unknown as Response
}

describe('searchAcquisition', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('POSTs the request body verbatim to /acquisition/search', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(acquisitionSearchResponseFixture))
    vi.stubGlobal('fetch', fetchMock)

    const request = { bbox: [80.12, 12.85, 80.32, 13.15] as [number, number, number, number], start_date: '2024-01-01', end_date: '2024-04-01', max_cloud_cover: 20, max_results: 10 }
    const result = await api.searchAcquisition(request)

    expect(fetchMock).toHaveBeenCalledTimes(1)
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toContain('/acquisition/search')
    expect(init.method).toBe('POST')
    expect(JSON.parse(init.body)).toEqual(request)
    expect(result).toEqual(acquisitionSearchResponseFixture)
  })

  it('maps a real zero-result response without throwing (a genuine, non-error outcome)', async () => {
    const emptyResult = { ...acquisitionSearchResponseFixture, scenes: [], scene_count: 0, note: 'No real Sentinel-2 scenes matched this AOI/date range/cloud threshold at search time.' }
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(emptyResult)))

    const result = await api.searchAcquisition({ bbox: [0, 0, 1, 1], start_date: '2024-01-01', end_date: '2024-01-02', max_cloud_cover: 100, max_results: 10 })

    expect(result.scene_count).toBe(0)
    expect(result.scenes).toEqual([])
    expect(result.note).not.toBeNull()
  })

  it('throws a typed ApiError with the backend-provided code on a 503 provider outage', async () => {
    const errorBody = { error: 'acquisition_provider_unavailable', code: 'acquisition_provider_unavailable', detail: 'CDSE STAC search failed: connection refused' }
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(errorBody, 503)))

    await expect(api.searchAcquisition({ bbox: [0, 0, 1, 1], start_date: '2024-01-01', end_date: '2024-01-02', max_cloud_cover: 100, max_results: 10 })).rejects.toMatchObject({
      status: 503,
      code: 'acquisition_provider_unavailable',
    })
  })

  it('throws a typed ApiError with the backend-provided code on a 400 invalid query', async () => {
    const errorBody = { error: 'invalid_acquisition_query', code: 'invalid_acquisition_query', detail: 'bbox longitudes are not a valid, ordered range' }
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(errorBody, 400)))

    await expect(api.searchAcquisition({ bbox: [1, 0, 0, 1], start_date: '2024-01-01', end_date: '2024-01-02', max_cloud_cover: 100, max_results: 10 })).rejects.toBeInstanceOf(ApiError)
  })

  it('reports an unreachable backend distinctly, without a real HTTP status', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')))

    try {
      await api.searchAcquisition({ bbox: [0, 0, 1, 1], start_date: '2024-01-01', end_date: '2024-01-02', max_cloud_cover: 100, max_results: 10 })
      expect.unreachable('should have thrown')
    } catch (error) {
      expect(error).toBeInstanceOf(ApiError)
      expect((error as ApiError).status).toBe(0)
      expect((error as ApiError).message).toMatch(/Could not reach the FRAME API/)
    }
  })
})

describe('fetchAcquisitionScene', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('GETs the real scene record by search_id/scene_id', async () => {
    const detail = { search_id: 'search-1', scene: sceneCandidateFixture, note: 'real asset hrefs are not returned here' }
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(detail))
    vi.stubGlobal('fetch', fetchMock)

    const result = await api.fetchAcquisitionScene('search-1', sceneCandidateFixture.id)

    const [url] = fetchMock.mock.calls[0]
    expect(url).toContain(`/acquisition/scenes/search-1/${encodeURIComponent(sceneCandidateFixture.id)}`)
    expect(result.scene).toEqual(sceneCandidateFixture)
  })

  it('surfaces a named 404 for an unknown scene id', async () => {
    const errorBody = { error: 'acquisition_scene_not_found', code: 'acquisition_scene_not_found', detail: "Scene 'x' was not part of search 'search-1'." }
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(errorBody, 404)))

    await expect(api.fetchAcquisitionScene('search-1', 'x')).rejects.toMatchObject({ status: 404, code: 'acquisition_scene_not_found' })
  })
})

describe('retrieveAcquisition', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('POSTs the retrieval request and maps real artifact download paths', async () => {
    const response = {
      retrieval_id: 'retrieval-1',
      search_id: 'search-1',
      scene_id: sceneCandidateFixture.id,
      bands: ['B02', 'B03', 'B04', 'B08'],
      band_native_resolution_m: { B02: 10, B03: 10, B04: 10, B08: 10 },
      band_sources: { B02: 'copernicus-cdse-process-api', B03: 'copernicus-cdse-process-api', B04: 'copernicus-cdse-process-api', B08: 'copernicus-cdse-process-api' },
      scl_included: true,
      crs: 'EPSG:32644',
      resolution_m: 10,
      width: 64,
      height: 64,
      retrieval_provider: 'copernicus-cdse-process-api',
      retrieval_endpoint: 'https://sh.dataspace.copernicus.eu/process/v1',
      matched_acquisition_datetime: sceneCandidateFixture.datetime,
      retrieved_at_utc: '2024-03-30T05:00:00Z',
      warnings: [],
      artifacts: { bands_geotiff: '/acquisition/download/retrieval-1/bands', scl_geotiff: '/acquisition/download/retrieval-1/scl' },
      note: 'real reflectance, not an SR product',
    }
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(response))
    vi.stubGlobal('fetch', fetchMock)

    const result = await api.retrieveAcquisition({ search_id: 'search-1', scene_id: sceneCandidateFixture.id, bands: ['B02', 'B03', 'B04', 'B08'], resolution_m: 10, include_scl: true })

    expect(result.artifacts.bands_geotiff).toBe('/acquisition/download/retrieval-1/bands')
    expect(api.acquisitionDownloadUrl(result.artifacts.bands_geotiff)).toBe(`${api.API_BASE_URL}/acquisition/download/retrieval-1/bands`)
  })

  it('surfaces an authentication failure distinctly from a generic provider outage', async () => {
    const errorBody = { error: 'acquisition_authentication_failed', code: 'acquisition_authentication_failed', detail: 'CDSE rejected the configured credentials' }
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(errorBody, 401)))

    await expect(
      api.retrieveAcquisition({ search_id: 'search-1', scene_id: sceneCandidateFixture.id, bands: ['B04'], resolution_m: 10, include_scl: false }),
    ).rejects.toMatchObject({ status: 401, code: 'acquisition_authentication_failed' })
  })
})
