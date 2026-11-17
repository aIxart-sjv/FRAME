/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useRef, useState } from 'react';
import { decodeGeotiffBlob, decodeGeotiffFile, type DecodedRaster } from '../lib/geotiff';

export interface RasterState {
  status: 'idle' | 'pending' | 'success' | 'error';
  raster: DecodedRaster | null;
  error: string | null;
}

/**
 * Fetches and decodes a GeoTIFF the backend serves (`fetcher` is one of the `api.fetch*Geotiff` client calls) and
 * caches the decoded result per `cacheKey` so switching tabs back and forth does not re-download it. Every raster
 * shown in the app -- the input, the SR output, the stability layer, the NDVI layers -- goes through this hook.
 */
export function useRemoteRaster(fetcher: () => Promise<Blob>, cacheKey: string | null): RasterState {
  const [state, setState] = useState<RasterState>({ status: 'idle', raster: null, error: null });
  const cache = useRef<Map<string, DecodedRaster>>(new Map());

  useEffect(() => {
    if (!cacheKey) {
      setState({ status: 'idle', raster: null, error: null });
      return;
    }
    const cached = cache.current.get(cacheKey);
    if (cached) {
      setState({ status: 'success', raster: cached, error: null });
      return;
    }
    let cancelled = false;
    setState({ status: 'pending', raster: null, error: null });
    (async () => {
      try {
        const blob = await fetcher();
        const raster = await decodeGeotiffBlob(blob);
        if (cancelled) return;
        cache.current.set(cacheKey, raster);
        setState({ status: 'success', raster, error: null });
      } catch (error) {
        if (cancelled) return;
        setState({ status: 'error', raster: null, error: error instanceof Error ? error.message : 'Could not decode this GeoTIFF.' });
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cacheKey]);

  return state;
}

/** Decodes a GeoTIFF the user picked locally (before it is uploaded) -- used to show the real "Original" image. */
export function useDecodedFile(file: File | null): RasterState {
  const [state, setState] = useState<RasterState>({ status: 'idle', raster: null, error: null });

  useEffect(() => {
    if (!file) {
      setState({ status: 'idle', raster: null, error: null });
      return;
    }
    let cancelled = false;
    setState({ status: 'pending', raster: null, error: null });
    (async () => {
      try {
        const raster = await decodeGeotiffFile(file);
        if (cancelled) return;
        setState({ status: 'success', raster, error: null });
      } catch (error) {
        if (cancelled) return;
        setState({ status: 'error', raster: null, error: error instanceof Error ? error.message : 'Could not decode this GeoTIFF.' });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [file]);

  return state;
}
