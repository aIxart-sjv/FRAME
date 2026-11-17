import type { AcquisitionSearchResponse, SceneCandidate, UploadResponse } from '../api/types'

/** A real-shaped scene candidate, copied from the real CDSE STAC response this project's own backend tests
 * (frame/tests/test_acquisition_live.py) verified against the live API -- not invented for the frontend. */
export const sceneCandidateFixture: SceneCandidate = {
  id: 'S2B_MSIL2A_20240330T045659_N0510_R119_T44PMV_20240330T072508',
  provider: 'copernicus-cdse',
  collection: 'sentinel-2-l2a',
  datetime: '2024-03-30T04:56:59.024000Z',
  cloud_cover: 2.43,
  bbox: [80.075, 12.574, 81.09, 13.568],
  geometry: {
    type: 'Polygon',
    coordinates: [
      [
        [80.075, 13.567],
        [80.079, 12.574],
        [81.09, 12.576],
        [81.083, 13.568],
        [80.075, 13.567],
      ],
    ],
  },
  mgrs_tile: 'MGRS-44PMV',
  platform: 'sentinel-2b',
  processing_baseline: '05.10',
  asset_count: 40,
}

export const acquisitionSearchResponseFixture: AcquisitionSearchResponse = {
  search_id: 'search-abc123',
  query: {
    bbox: [80.12, 12.85, 80.32, 13.15],
    start_date: '2024-01-01',
    end_date: '2024-04-01',
    max_cloud_cover: 20,
    max_results: 10,
  },
  provider: 'copernicus-cdse',
  stac_endpoint: 'https://stac.dataspace.copernicus.eu/v1/search',
  scenes: [sceneCandidateFixture],
  scene_count: 1,
  note: null,
}

export const uploadFixture: UploadResponse = {
  upload_id: 'upload-abc123',
  filename: 'scene.tif',
  valid: true,
  band_names: ['B04', 'B03', 'B02', 'B08'],
  width: 128,
  height: 128,
  crs: 'EPSG:32630',
  resolution_m: 10.0,
  input_scale: 'raw_digital_number',
  validation_messages: ['Upload accepted: required bands and shape present, CRS valid.'],
}
