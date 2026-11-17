/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Types for the real FRAME API (frame/api/schemas.py). Every field here exists in the backend's actual response
 * models -- nothing is invented to make a UI easier to write. Field names and shapes match schemas.py exactly.
 */

export type ModelId = 'lite' | 'mamba';

export interface ModelAvailability {
  id: ModelId;
  label: string;
  model_name: string;
  available: boolean;
  reason: string | null;
}

export interface HealthResponse {
  status: string;
  frame_version: string | null;
  api_version: string;
  model_name: string;
  default_model: ModelId;
  available_models: ModelAvailability[];
}

export interface UploadResponse {
  upload_id: string;
  filename: string;
  valid: boolean;
  band_names: string[];
  width: number;
  height: number;
  crs: string | null;
  resolution_m: number | null;
  input_scale: InputScale;
  validation_messages: string[];
}

export type InputScale = 'raw_digital_number' | 'reflectance';

export interface ResolutionDescription {
  native_resolution_m: number;
  sr_resolution_m: number;
  scale_factor: number;
  description: string;
}

export interface UncertaintyDistribution {
  mean: number;
  median: number;
  std: number;
  p90: number;
  p95: number;
  min: number;
  max: number;
  n_pixels: number;
}

export interface UncertaintySummary {
  label: string;
  scalar_summary: number;
  scalar_summary_definition: string;
  overall_distribution: UncertaintyDistribution;
  n: number;
  seed: number;
  transform_names: string[];
  disclaimer: string;
}

export interface SelfConsistencySummary {
  downsample_rmse: number | null;
  ndvi_discrepancy_mean_abs: number | null;
  b08_b04_ratio_discrepancy_mean_abs: number | null;
  note: string;
}

export interface SeamDiagnostic {
  status: 'COMPUTABLE' | 'NOT_COMPUTABLE';
  pair_count: number;
  overlap_pixel_count: number;
  mean_abs_difference: number | null;
  rmse: number | null;
  max_abs_difference: number | null;
  note: string;
}

export interface TilingSummary {
  tile_size: number;
  overlap: number;
  stride: number;
  scale: number;
  padding_mode: string;
  blend_mode: string;
  scene_shape?: [number, number];
  sr_shape?: [number, number];
  tile_grid?: [number, number];
  tile_count?: number;
  padded_tiles?: number;
  scene_passes?: number;
  tile_inferences?: number;
  mean_tile_seconds?: number;
  tile_seconds_total?: number;
  seam_diagnostic?: SeamDiagnostic | null;
  seam_diagnostic_definition?: string;
}

export interface OutputGeospatial {
  crs: string | null;
  transform: number[];
  bounds: number[];
  width: number;
  height: number;
}

export interface SRResultMetadata {
  inference_seconds: number;
  device: string;
  tiling: TilingSummary;
  model_runtime?: Record<string, unknown>;
  output_geospatial: OutputGeospatial;
  preprocessing_mask_coverage: number;
}

export interface SRResultResponse {
  job_id: string;
  status: string;
  upload_id: string;
  model_name: string;
  model_id: ModelId;
  input_shape: number[];
  output_shape: number[];
  resolution: ResolutionDescription;
  bands: string[];
  crs: string | null;
  uncertainty: UncertaintySummary;
  self_consistency: SelfConsistencySummary;
  metadata: SRResultMetadata;
  scientific_caveats: string[];
  artifacts: { sr_geotiff: string; uncertainty_geotiff: string };
  created_at: string;
}

export interface NDVIComparisonSummary {
  status: string;
  valid_pixel_count: number;
  mean_abs_difference: number | null;
  rmse: number | null;
  max_abs_difference: number | null;
  resampling_method: string;
}

export interface UncertaintyWeightedSummary {
  status: string;
  correlation_uncertainty_vs_abs_diff: number | null;
  uncertainty_weighted_mean_abs_diff: number | null;
  unweighted_mean_abs_diff: number | null;
}

export interface NDVIAnalysisResponse {
  analysis_id: string;
  job_id: string;
  status: string;
  ndvi_formula: string;
  native_resolution_m: number;
  sr_resolution_m: number;
  comparison: NDVIComparisonSummary;
  uncertainty_weighted_summary: UncertaintyWeightedSummary;
  scientific_caveats: string[];
  metadata: Record<string, unknown>;
  artifacts: {
    native_ndvi_geotiff: string;
    sr_ndvi_geotiff: string;
    ndvi_diff_geotiff: string;
  };
  created_at: string;
}

// ---------------------------------------------------------------------------
// POST /acquisition/search, GET /acquisition/scenes/{search_id}/{scene_id},
// POST /acquisition/retrieve, GET /acquisition/download/{retrieval_id}/{bands,scl}
//
// Real Sentinel-2 discovery/retrieval (frame/acquisition/, frame/api/schemas.py).
// Every field here is copied verbatim from a real backend response -- never
// fabricated. See docs/ACQUISITION.md.
// ---------------------------------------------------------------------------

export interface GeoJsonPolygon {
  type: 'Polygon' | 'MultiPolygon';
  coordinates: unknown;
}

export interface AcquisitionSearchRequest {
  bbox?: [number, number, number, number];
  aoi_geojson?: GeoJsonPolygon;
  start_date: string;
  end_date: string;
  max_cloud_cover: number;
  max_results: number;
}

export interface SceneCandidate {
  id: string;
  provider: string;
  collection: string;
  datetime: string;
  cloud_cover: number | null;
  bbox: [number, number, number, number];
  geometry: GeoJsonPolygon | null;
  mgrs_tile: string | null;
  platform: string | null;
  processing_baseline: string | null;
  asset_count: number;
}

export interface AcquisitionSearchResponse {
  search_id: string;
  query: AcquisitionSearchRequest;
  provider: string;
  stac_endpoint: string;
  scenes: SceneCandidate[];
  scene_count: number;
  note: string | null;
}

export interface AcquisitionSceneDetailResponse {
  search_id: string;
  scene: SceneCandidate;
  note: string;
}

export interface AcquisitionRetrieveRequest {
  search_id: string;
  scene_id: string;
  bands: string[];
  resolution_m: number;
  include_scl: boolean;
}

export interface AcquisitionSrInputResponse {
  upload_id: string;
  filename: string;
  valid: boolean;
  band_names: string[];
  width: number;
  height: number;
  crs: string | null;
  resolution_m: number | null;
  input_scale: InputScale;
  validation_messages: string[];
  retrieval_id: string;
  search_id: string;
  scene_id: string;
  provider: string;
  matched_acquisition_datetime: string;
  excluded_bands: string[];
  note: string;
}

export interface AcquisitionRetrieveResponse {
  retrieval_id: string;
  search_id: string;
  scene_id: string;
  bands: string[];
  band_native_resolution_m: Record<string, number>;
  band_sources: Record<string, string>;
  /** Step F (frame.acquisition.harmonise): the real radiometric conversion each band's actual
   * source provider applied -- never one assumed formula for the whole stack. */
  band_radiometric_method: Record<string, string>;
  /** Whether this retrieval's bands cover all 12 required Sentinel-2 L2A bands (B10 excluded). */
  is_complete_l2a: boolean;
  /** Fraction of pixels the real SCL-derived validity mask marks valid, or null if SCL was not retrieved. */
  validity_mask_coverage: number | null;
  scl_included: boolean;
  crs: string;
  resolution_m: number;
  width: number;
  height: number;
  retrieval_provider: string;
  retrieval_endpoint: string;
  matched_acquisition_datetime: string;
  retrieved_at_utc: string;
  warnings: string[];
  artifacts: Record<string, string>;
  note: string;
}

/** The shape of every 4xx/5xx JSON error body (frame/api/errors.py, frame/api/app.py). */
export interface ApiErrorBody {
  error: string;
  code: string;
  detail: string;
}

/** Thrown by the client for any non-2xx response; carries the parsed error body when the server sent one. */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string | null;

  constructor(status: number, body: ApiErrorBody | null, fallbackMessage: string) {
    super(body?.detail ?? fallbackMessage);
    this.name = 'ApiError';
    this.status = status;
    this.code = body?.code ?? null;
  }
}

/** A completed reconstruction, kept in this browser session only (the backend has no persistence -- see frame/api/README.md, "Prototype behavior"). */
export interface SessionProject {
  id: string;
  label: string;
  sourceFilename: string;
  createdAt: string;
  result: SRResultResponse;
  ndviAnalysis: NDVIAnalysisResponse | null;
}
