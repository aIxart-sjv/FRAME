/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import * as api from '../api/client';
import { ApiError } from '../api/types';
import type {
  AcquisitionRetrieveResponse,
  AcquisitionSearchResponse,
  HealthResponse,
  InputScale,
  ModelId,
  NDVIAnalysisResponse,
  SceneCandidate,
  SessionProject,
  SRResultResponse,
  UploadResponse,
} from '../api/types';

export interface SrSourceProvenance {
  retrievalId: string;
  sceneId: string;
  provider: string;
  matchedAcquisitionDatetime: string;
  excludedBands: string[];
}
import type { AoiPolygonGeojson, Bbox } from '../lib/aoi';
import { isValidBbox } from '../lib/aoi';

export type AsyncStatus = 'idle' | 'pending' | 'success' | 'error';

function messageOf(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof DOMException && error.name === 'AbortError') return 'Cancelled.';
  if (error instanceof Error) return error.message;
  return 'Something unexpected went wrong.';
}

/**
 * All state for one FRAME session: the backend health check, the upload -> run -> NDVI flow against the REAL
 * API, and this browser's own in-memory list of completed reconstructions (the backend keeps no history of its
 * own -- frame/api/README.md, "Prototype behavior"). No field here is ever filled with fabricated data: a status
 * of 'idle' or 'error' with no result is the honest state until a real response arrives.
 */
export function useFrameSession() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthStatus, setHealthStatus] = useState<AsyncStatus>('idle');
  const [healthError, setHealthError] = useState<string | null>(null);

  const [file, setFile] = useState<File | null>(null);
  const [inputScale, setInputScale] = useState<InputScale>('raw_digital_number');
  const [uploadStatus, setUploadStatus] = useState<AsyncStatus>('idle');
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const [model, setModel] = useState<ModelId>('lite');
  const [jobStatus, setJobStatus] = useState<AsyncStatus>('idle');
  const [jobResult, setJobResult] = useState<SRResultResponse | null>(null);
  const [jobError, setJobError] = useState<string | null>(null);

  const [analysisStatus, setAnalysisStatus] = useState<AsyncStatus>('idle');
  const [analysisResult, setAnalysisResult] = useState<NDVIAnalysisResponse | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  const [sessionProjects, setSessionProjects] = useState<SessionProject[]>([]);

  // Real Sentinel-2 acquisition (map/AOI -> POST /acquisition/search -> POST /acquisition/retrieve). See
  // docs/ACQUISITION.md -- manual upload above remains the fallback path, untouched.
  const [aoiBbox, setAoiBboxState] = useState<Bbox | null>(null);
  const [aoiGeojson, setAoiGeojsonState] = useState<AoiPolygonGeojson | null>(null);
  const [startDate, setStartDate] = useState<string>('');
  const [endDate, setEndDate] = useState<string>('');
  const [maxCloudCover, setMaxCloudCover] = useState<number>(20);

  const [searchStatus, setSearchStatus] = useState<AsyncStatus>('idle');
  const [searchResult, setSearchResult] = useState<AcquisitionSearchResponse | null>(null);
  const [searchError, setSearchError] = useState<string | null>(null);

  const [selectedSceneId, setSelectedSceneId] = useState<string | null>(null);

  const [retrieveStatus, setRetrieveStatus] = useState<AsyncStatus>('idle');
  const [retrieveResult, setRetrieveResult] = useState<AcquisitionRetrieveResponse | null>(null);
  const [retrieveError, setRetrieveError] = useState<string | null>(null);

  // Step E: handing a real retrieval off into the existing, unmodified /sr/run (POST
  // /acquisition/retrieve/{id}/prepare-sr-input). Populates the SAME uploadResult/uploadStatus
  // state a manual /upload fills, so the existing configure/processing/result views run unchanged.
  const [prepareSrStatus, setPrepareSrStatus] = useState<AsyncStatus>('idle');
  const [prepareSrError, setPrepareSrError] = useState<string | null>(null);
  const [acquisitionOriginalFile, setAcquisitionOriginalFile] = useState<File | null>(null);
  const [srSourceProvenance, setSrSourceProvenance] = useState<SrSourceProvenance | null>(null);

  const uploadRequestId = useRef(0);
  const runAbortController = useRef<AbortController | null>(null);
  const searchRequestId = useRef(0);
  const retrieveAbortController = useRef<AbortController | null>(null);
  const prepareSrAbortController = useRef<AbortController | null>(null);

  const checkHealth = useCallback(async () => {
    setHealthStatus('pending');
    try {
      const result = await api.fetchHealth();
      setHealth(result);
      setHealthStatus('success');
      setHealthError(null);
    } catch (error) {
      setHealthStatus('error');
      setHealthError(messageOf(error));
    }
  }, []);

  useEffect(() => {
    checkHealth();
  }, [checkHealth]);

  const setAoi = useCallback((bbox: Bbox | null, geojson: AoiPolygonGeojson | null = null) => {
    setAoiBboxState(bbox);
    setAoiGeojsonState(geojson);
    // A new AOI invalidates any scene search made for the previous one.
    setSearchStatus('idle');
    setSearchResult(null);
    setSearchError(null);
    setSelectedSceneId(null);
  }, []);

  const clearAoi = useCallback(() => setAoi(null, null), [setAoi]);

  const runAcquisitionSearch = useCallback(async () => {
    if (!aoiBbox || !isValidBbox(aoiBbox)) {
      setSearchStatus('error');
      setSearchError('Draw or select a valid area of interest before searching.');
      return;
    }
    if (!startDate || !endDate) {
      setSearchStatus('error');
      setSearchError('Choose a start and end date before searching.');
      return;
    }
    const requestId = ++searchRequestId.current;
    setSearchStatus('pending');
    setSearchError(null);
    setSelectedSceneId(null);
    try {
      const result = await api.searchAcquisition(
        aoiGeojson
          ? { aoi_geojson: aoiGeojson, start_date: startDate, end_date: endDate, max_cloud_cover: maxCloudCover, max_results: 20 }
          : { bbox: aoiBbox, start_date: startDate, end_date: endDate, max_cloud_cover: maxCloudCover, max_results: 20 },
      );
      if (searchRequestId.current !== requestId) return; // superseded by a newer search
      setSearchResult(result);
      setSearchStatus('success');
    } catch (error) {
      if (searchRequestId.current !== requestId) return;
      setSearchStatus('error');
      setSearchError(messageOf(error));
    }
  }, [aoiBbox, aoiGeojson, startDate, endDate, maxCloudCover]);

  const selectScene = useCallback((sceneId: string | null) => {
    setSelectedSceneId(sceneId);
    setRetrieveStatus('idle');
    setRetrieveResult(null);
    setRetrieveError(null);
  }, []);

  const selectedScene: SceneCandidate | null =
    (searchResult?.scenes.find((s) => s.id === selectedSceneId) ?? null) || null;

  const retrieveSelectedScene = useCallback(
    async (bands: string[], resolutionM: number, includeScl: boolean) => {
      if (!searchResult || !selectedSceneId) return;
      retrieveAbortController.current?.abort();
      const controller = new AbortController();
      retrieveAbortController.current = controller;
      setRetrieveStatus('pending');
      setRetrieveError(null);
      try {
        const result = await api.retrieveAcquisition(
          {
            search_id: searchResult.search_id,
            scene_id: selectedSceneId,
            bands,
            resolution_m: resolutionM,
            include_scl: includeScl,
          },
          controller.signal,
        );
        setRetrieveResult(result);
        setRetrieveStatus('success');
      } catch (error) {
        if (error instanceof DOMException && error.name === 'AbortError') {
          setRetrieveStatus('idle');
          return;
        }
        setRetrieveStatus('error');
        setRetrieveError(messageOf(error));
      }
    },
    [searchResult, selectedSceneId],
  );

  const cancelRetrieve = useCallback(() => {
    retrieveAbortController.current?.abort();
  }, []);

  const runSrFromAcquisition = useCallback(async () => {
    if (!retrieveResult) return;
    prepareSrAbortController.current?.abort();
    const controller = new AbortController();
    prepareSrAbortController.current = controller;
    setPrepareSrStatus('pending');
    setPrepareSrError(null);
    try {
      const prepared = await api.prepareSrInputFromAcquisition(retrieveResult.retrieval_id, controller.signal);

      setUploadResult({
        upload_id: prepared.upload_id,
        filename: prepared.filename,
        valid: prepared.valid,
        band_names: prepared.band_names,
        width: prepared.width,
        height: prepared.height,
        crs: prepared.crs,
        resolution_m: prepared.resolution_m,
        input_scale: prepared.input_scale,
        validation_messages: prepared.validation_messages,
      });
      setUploadStatus('success');
      setSrSourceProvenance({
        retrievalId: prepared.retrieval_id,
        sceneId: prepared.scene_id,
        provider: prepared.provider,
        matchedAcquisitionDatetime: prepared.matched_acquisition_datetime,
        excludedBands: prepared.excluded_bands,
      });

      // The "original" comparison view is a real convenience, not a requirement of SR itself --
      // its failure must never block a successful hand-off into the SR pipeline above.
      try {
        const blob = await api.fetchUploadGeotiff(prepared.upload_id, controller.signal);
        setAcquisitionOriginalFile(new File([blob], prepared.filename, { type: 'image/tiff' }));
      } catch {
        setAcquisitionOriginalFile(null);
      }

      setPrepareSrStatus('success');
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        setPrepareSrStatus('idle');
        return;
      }
      setPrepareSrStatus('error');
      setPrepareSrError(messageOf(error));
    }
  }, [retrieveResult]);

  const cancelPrepareSr = useCallback(() => {
    prepareSrAbortController.current?.abort();
  }, []);

  const uploadWithScale = useCallback(async (nextFile: File, scale: InputScale) => {
    const requestId = ++uploadRequestId.current;
    setUploadStatus('pending');
    setUploadError(null);
    setUploadResult(null);
    try {
      const result = await api.uploadScene(nextFile, scale);
      if (uploadRequestId.current !== requestId) return; // a newer selection/scale change superseded this request
      setUploadResult(result);
      setUploadStatus('success');
    } catch (error) {
      if (uploadRequestId.current !== requestId) return;
      setUploadStatus('error');
      setUploadError(messageOf(error));
    }
  }, []);

  const selectFile = useCallback(
    (next: File | null) => {
      setFile(next);
      setJobStatus('idle');
      setJobResult(null);
      setJobError(null);
      setAnalysisStatus('idle');
      setAnalysisResult(null);
      setAnalysisError(null);
      if (!next) {
        uploadRequestId.current += 1;
        setUploadStatus('idle');
        setUploadResult(null);
        setUploadError(null);
        return;
      }
      uploadWithScale(next, inputScale);
    },
    [inputScale, uploadWithScale],
  );

  const changeInputScale = useCallback(
    (scale: InputScale) => {
      setInputScale(scale);
      if (file) uploadWithScale(file, scale);
    },
    [file, uploadWithScale],
  );

  const runReconstruction = useCallback(async () => {
    if (!uploadResult) return;
    runAbortController.current?.abort();
    const controller = new AbortController();
    runAbortController.current = controller;
    setJobStatus('pending');
    setJobError(null);
    try {
      const result = await api.runReconstruction(uploadResult.upload_id, model, undefined, controller.signal);
      setJobResult(result);
      setJobStatus('success');
      setSessionProjects((prev) => [
        {
          id: result.job_id,
          label: file?.name ?? uploadResult.filename,
          sourceFilename: uploadResult.filename,
          createdAt: result.created_at,
          result,
          ndviAnalysis: null,
        },
        ...prev,
      ]);
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        setJobStatus('idle');
        return;
      }
      setJobStatus('error');
      setJobError(messageOf(error));
    }
  }, [uploadResult, model, file]);

  const cancelRun = useCallback(() => {
    runAbortController.current?.abort();
  }, []);

  const runNdviAnalysis = useCallback(async () => {
    if (!jobResult) return;
    setAnalysisStatus('pending');
    setAnalysisError(null);
    try {
      const result = await api.runNdviAnalysis(jobResult.job_id);
      setAnalysisResult(result);
      setAnalysisStatus('success');
      setSessionProjects((prev) => prev.map((p) => (p.id === jobResult.job_id ? { ...p, ndviAnalysis: result } : p)));
    } catch (error) {
      setAnalysisStatus('error');
      setAnalysisError(messageOf(error));
    }
  }, [jobResult]);

  const reset = useCallback(() => {
    uploadRequestId.current += 1;
    searchRequestId.current += 1;
    runAbortController.current?.abort();
    retrieveAbortController.current?.abort();
    prepareSrAbortController.current?.abort();
    setFile(null);
    setUploadStatus('idle');
    setUploadResult(null);
    setUploadError(null);
    setJobStatus('idle');
    setJobResult(null);
    setJobError(null);
    setAnalysisStatus('idle');
    setAnalysisResult(null);
    setAnalysisError(null);
    setAoiBboxState(null);
    setAoiGeojsonState(null);
    setSearchStatus('idle');
    setSearchResult(null);
    setSearchError(null);
    setSelectedSceneId(null);
    setRetrieveStatus('idle');
    setRetrieveResult(null);
    setRetrieveError(null);
    setPrepareSrStatus('idle');
    setPrepareSrError(null);
    setAcquisitionOriginalFile(null);
    setSrSourceProvenance(null);
  }, []);

  const openProject = useCallback((project: SessionProject) => {
    setJobResult(project.result);
    setJobStatus('success');
    setAnalysisResult(project.ndviAnalysis);
    setAnalysisStatus(project.ndviAnalysis ? 'success' : 'idle');
    setUploadResult(null);
    setFile(null);
  }, []);

  return {
    health,
    healthStatus,
    healthError,
    checkHealth,
    file,
    selectFile,
    inputScale,
    setInputScale: changeInputScale,
    uploadStatus,
    uploadResult,
    uploadError,
    model,
    setModel,
    jobStatus,
    jobResult,
    jobError,
    runReconstruction,
    cancelRun,
    analysisStatus,
    analysisResult,
    analysisError,
    runNdviAnalysis,
    sessionProjects,
    openProject,
    reset,
    aoiBbox,
    aoiGeojson,
    setAoi,
    clearAoi,
    startDate,
    setStartDate,
    endDate,
    setEndDate,
    maxCloudCover,
    setMaxCloudCover,
    searchStatus,
    searchResult,
    searchError,
    runAcquisitionSearch,
    selectedSceneId,
    selectedScene,
    selectScene,
    retrieveStatus,
    retrieveResult,
    retrieveError,
    retrieveSelectedScene,
    cancelRetrieve,
    prepareSrStatus,
    prepareSrError,
    runSrFromAcquisition,
    cancelPrepareSr,
    acquisitionOriginalFile,
    srSourceProvenance,
  };
}

export type FrameSession = ReturnType<typeof useFrameSession>;
