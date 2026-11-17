/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

/** Trigger a real browser download of a Blob (a GeoTIFF fetched from the backend, or JSON built from an already-received response). */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  // Revoke on the next tick: some browsers need the anchor's click to fully resolve first.
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

/** Download an in-memory object (the actual API response, not a fabricated report) as a formatted JSON file. */
export function downloadJson(data: unknown, filename: string): void {
  downloadBlob(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }), filename);
}
