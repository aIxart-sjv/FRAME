# FRAME frontend

A console for the FRAME pipeline: upload a Sentinel-2 L2A GeoTIFF, run SEN2SR-Lite or SEN2SR-Mamba, and inspect the SR-derived output, its TTA stability diagnostic, and the NDVI downstream demonstration —
entirely through the real FastAPI backend (`frame/api/`, see its own README for the endpoint contract). Nothing here is computed in the browser except decoding a GeoTIFF for display and a percentile-stretch render.

## Running it

Requires the FRAME API running first (from the repository root):

```bash
sen2sr_venv/bin/uvicorn frame.api.app:app --port 8000
```

Then, in this directory:

```bash
npm install
npm run dev
```

Open the printed `http://localhost:3000` URL.

## Backend URL configuration

Set with `VITE_FRAME_API_URL` (see `.env.example`); defaults to `http://127.0.0.1:8000`. Copy `.env.example` to `.env.local` to override it:

```bash
cp .env.example .env.local
# edit .env.local
```

The backend's CORS allow-list (`FRAME_API_CORS_ORIGINS`) must include this frontend's origin — its default already includes `http://localhost:3000`.

## Supported input

A Sentinel-2 L2A GeoTIFF with exactly the four bands the backend supports — **B04, B03, B02, B08** (order-independent) — of any height and width up to the backend's pixel cap (larger scenes are tiled server-side), with an embedded CRS. Pixel values must be declared as either raw digital numbers or already-scaled reflectance (0–1); the wrong declaration is refused with a message explaining why, not silently mis-processed.

## What is, and is not, connected to the backend

Everything shown in the app is either a real value from a `frame/api` response, or explicitly labelled as unavailable. In particular:

- **Upload → run → inspect → NDVI → download** is the real, only product path this backend exposes. There is no scene catalogue, no AOI search, and no asynchronous job queue on the server — `/sr/run` is one synchronous request, and this app waits on it rather than polling or animating a fabricated multi-stage progress bar.
- Only **SEN2SR-Lite** and **SEN2SR-Mamba** exist; there is no third model.
- The stability view is a **TTA reconstruction-variation diagnostic**, not a calibrated uncertainty — the app repeats the backend's own disclaimer verbatim rather than a paraphrase.
- The NDVI view is a **downstream demonstration** on the one uploaded scene, not proof that super-resolution improves an NDVI-derived decision.
- There is no independent high-resolution reference for an arbitrary uploaded scene, so PSNR/SSIM/SAM/ERGAS and any hallucination/omission rate are never shown for a live reconstruction — the Validate workspace says so, using the backend's own framing, instead of inventing a number.
- Land-cover classification, multi-temporal change detection, NDBI/NDWI/BSI, and any claim about Indian validation are **not implemented anywhere in this app**, because the backend does not compute them for a live job (see `docs/CLAIMS.md` in the repository root for what the project's own offline research separately measured, and did not measure).
- "Projects" are this browser session's completed reconstructions only — the backend keeps job records in memory for the life of the process (see `frame/api/README.md`, "Prototype behavior"), so nothing here is persisted across a backend restart.

## Development commands

```bash
npm run dev       # start the Vite dev server
npm run build     # production build
npm run preview   # preview the production build locally
npm run lint       # tsc --noEmit
```
