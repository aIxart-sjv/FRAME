/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * A thin client of the real FRAME API (frame/api/, FastAPI). Every function here maps to exactly one endpoint
 * documented in frame/api/README.md -- nothing is fabricated, no endpoint is guessed, and no response is mocked.
 */

import type {
  AcquisitionRetrieveRequest,
  AcquisitionRetrieveResponse,
  AcquisitionSceneDetailResponse,
  AcquisitionSearchRequest,
  AcquisitionSearchResponse,
  AcquisitionSrInputResponse,
  HealthResponse,
  InputScale,
  ModelId,
  NDVIAnalysisResponse,
  SRResultResponse,
  UploadResponse,
} from './types';
import { ApiError } from './types';

export const API_BASE_URL = (import.meta.env.VITE_FRAME_API_URL as string | undefined)?.replace(/\/$/, '') || 'http://127.0.0.1:8000';

async function parseErrorBody(response: Response): Promise<ApiError> {
  try {
    const body = await response.json();
    return new ApiError(response.status, body, response.statusText);
  } catch {
    return new ApiError(response.status, null, response.statusText || 'Request failed');
  }
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init);
  } catch (cause) {
    // fetch() itself throws only for network-level failures (backend unreachable, CORS, DNS, aborted) -- there is no HTTP status to read.
    if (cause instanceof DOMException && cause.name === 'AbortError') throw cause;
    throw new ApiError(0, null, `Could not reach the FRAME API at ${API_BASE_URL}. Is the backend running?`);
  }
  if (!response.ok) throw await parseErrorBody(response);
  return (await response.json()) as T;
}

async function requestBlob(path: string, init?: RequestInit): Promise<Blob> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init);
  } catch {
    throw new ApiError(0, null, `Could not reach the FRAME API at ${API_BASE_URL}. Is the backend running?`);
  }
  if (!response.ok) throw await parseErrorBody(response);
  return response.blob();
}

/** POST /acquisition/search -- real Copernicus Data Space STAC search. Returns real scene metadata/footprints,
 * cached server-side under `search_id`; a zero-result search is a normal 200 response (`scene_count: 0`), not
 * an error. */
export function searchAcquisition(request: AcquisitionSearchRequest, signal?: AbortSignal): Promise<AcquisitionSearchResponse> {
  return requestJson<AcquisitionSearchResponse>('/acquisition/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
    signal,
  });
}

/** GET /acquisition/scenes/{search_id}/{scene_id} -- the real scene record from a previous search, by id. */
export function fetchAcquisitionScene(searchId: string, sceneId: string, signal?: AbortSignal): Promise<AcquisitionSceneDetailResponse> {
  return requestJson<AcquisitionSceneDetailResponse>(
    `/acquisition/scenes/${encodeURIComponent(searchId)}/${encodeURIComponent(sceneId)}`,
    { signal },
  );
}

/** POST /acquisition/retrieve -- real multi-resolution pixel retrieval + harmonisation for a previously-searched
 * scene (frame.acquisition.retrieve). Not yet wired to /sr/run -- see docs/ACQUISITION.md. */
export function retrieveAcquisition(request: AcquisitionRetrieveRequest, signal?: AbortSignal): Promise<AcquisitionRetrieveResponse> {
  return requestJson<AcquisitionRetrieveResponse>('/acquisition/retrieve', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
    signal,
  });
}

export function acquisitionDownloadUrl(path: string): string {
  return `${API_BASE_URL}${path}`;
}

/** GET /upload/{upload_id}/download -- the exact real file behind an upload_id (added for Step E:
 * an acquisition-derived upload has no browser File object already in memory the way a manual
 * /upload does, so the "original" comparison view fetches the real bytes /sr/run actually used). */
export function fetchUploadGeotiff(uploadId: string, signal?: AbortSignal): Promise<Blob> {
  return requestBlob(`/upload/${encodeURIComponent(uploadId)}/download`, { signal });
}

/** POST /acquisition/retrieve/{retrieval_id}/prepare-sr-input -- builds a real RGBN GeoTIFF from a
 * completed retrieval and registers it exactly like a manual /upload; the returned `upload_id` is
 * then used with the existing, unmodified `runReconstruction`/`POST /sr/run`. */
export function prepareSrInputFromAcquisition(retrievalId: string, signal?: AbortSignal): Promise<AcquisitionSrInputResponse> {
  return requestJson<AcquisitionSrInputResponse>(`/acquisition/retrieve/${encodeURIComponent(retrievalId)}/prepare-sr-input`, {
    method: 'POST',
    signal,
  });
}

/** GET /health -- never starts a model; only reports what is selectable right now. */
export function fetchHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return requestJson<HealthResponse>('/health', { signal });
}

/** POST /upload -- validates a GeoTIFF (bands, CRS, shape, declared scale vs. its values). Does not run SR. */
export function uploadScene(file: File, inputScale: InputScale, signal?: AbortSignal): Promise<UploadResponse> {
  const form = new FormData();
  form.append('file', file);
  form.append('input_scale', inputScale);
  return requestJson<UploadResponse>('/upload', { method: 'POST', body: form, signal });
}

/** POST /sr/run -- one synchronous request: preprocessing, the selected model through the tile engine, the TTA stability ensemble, and geospatial export. No polling endpoint exists because there is no asynchronous execution to poll. */
export function runReconstruction(uploadId: string, model: ModelId, seed?: number, signal?: AbortSignal): Promise<SRResultResponse> {
  return requestJson<SRResultResponse>('/sr/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ upload_id: uploadId, model, ...(seed !== undefined ? { seed } : {}) }),
    signal,
  });
}

/** POST /analysis/ndvi -- the downstream NDVI demonstration against a completed job's already-saved tensors (no re-inference). */
export function runNdviAnalysis(jobId: string, signal?: AbortSignal): Promise<NDVIAnalysisResponse> {
  return requestJson<NDVIAnalysisResponse>('/analysis/ndvi', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ job_id: jobId }),
    signal,
  });
}

export function srDownloadUrl(jobId: string): string {
  return `${API_BASE_URL}/sr/download/${jobId}`;
}

export function uncertaintyDownloadUrl(jobId: string): string {
  return `${API_BASE_URL}/uncertainty/download/${jobId}`;
}

export function ndviDownloadUrl(analysisId: string, layer: 'native-ndvi' | 'sr-ndvi' | 'ndvi-diff'): string {
  return `${API_BASE_URL}/analysis/download/${analysisId}/${layer}`;
}

export function fetchSrGeotiff(jobId: string, signal?: AbortSignal): Promise<Blob> {
  return requestBlob(`/sr/download/${jobId}`, { signal });
}

export function fetchUncertaintyGeotiff(jobId: string, signal?: AbortSignal): Promise<Blob> {
  return requestBlob(`/uncertainty/download/${jobId}`, { signal });
}

export function fetchNdviGeotiff(analysisId: string, layer: 'native-ndvi' | 'sr-ndvi' | 'ndvi-diff', signal?: AbortSignal): Promise<Blob> {
  return requestBlob(`/analysis/download/${analysisId}/${layer}`, { signal });
}
