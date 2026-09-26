/** Client-side download helpers. GeoTIFFs are fetched (rather than a plain
 * `<a href download>`) specifically so a 404/5xx can be caught and shown
 * inline instead of the browser silently navigating to an error page. */

export async function downloadFile(url: string, filename: string): Promise<void> {
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`Download failed (HTTP ${response.status}). The artifact may no longer be available.`)
  }
  const blob = await response.blob()
  triggerBlobDownload(blob, filename)
}

export function downloadJson(data: unknown, filename: string): void {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
  triggerBlobDownload(blob, filename)
}

function triggerBlobDownload(blob: Blob, filename: string): void {
  const objectUrl = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = objectUrl
  link.download = filename
  document.body.appendChild(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(objectUrl)
}
