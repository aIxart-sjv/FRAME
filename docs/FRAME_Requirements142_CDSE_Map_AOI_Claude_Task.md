# FRAME — Requirements 142 + CDSE Map/AOI Integration Task for Claude

## Purpose

You are working on **FRAME** for **SIH 2026 Problem Statement 26142**:

> **Deep Learning Based Super Resolution Mapping (SRM) from Medium Resolution Satellite Imageries**

The target is not a simple "upload a 10 m image and upscale it to 2.5 m" demo.

**FRAME must become a real map-first geospatial application** in which the user can select an area of interest (AOI), discover real Sentinel-2 imagery, select a scene, retrieve the required multispectral data, preprocess it correctly, run the verified SR pipeline, and inspect the resulting geospatial products on the map.

---

# 1. SOURCE OF TRUTH — READ THIS FIRST

Before modifying code, you MUST read:

```text
Requirements 142.txt
```

This is the main project requirements/research specification for SIH26142.

There is also a duplicate/current-library copy named:

```text
Requirements 142(1).txt
```

Use the file that exists in the repository/workspace. Do not replace the requirements with your own interpretation.

The requirements document is organized around:

1. Sentinel-2 Data Acquisition
2. Multispectral Preprocessing & Harmonisation
3. Paired LR–HR Dataset Creation
4. Super-Resolution Model
5. Spectral Consistency
6. Geospatial Consistency
7. Uncertainty / Model-Stability / Error Awareness
8. SR Accuracy & Validation
9. High-Resolution Reference Validation
10. Downstream Analytical Utility
11. Operationalisation / GIS-Ready Product Generation / System Deployment
12. Geographic / Indian / Cross-Domain Generalisation

Treat those requirements as the **target product/system scope**.

Do NOT treat our existing status documents as a substitute for the requirements.

---

# 2. READ THE EXISTING FRAME DOCUMENTATION TOO

After reading `Requirements 142.txt`, audit the current implementation against the existing project docs.

At minimum inspect:

```text
docs/README.md
RELEASE_READINESS.md
CLAIMS.md
TRACEABILITY.md
REGISTRY.md
REPRODUCIBILITY.md
DEMO_RUNBOOK.md
DATA.md
TRAINING.md
EVALUATION.md
RELIABILITY.md
DOWNSTREAM.md
MAMBA_INTEGRATION.md
TILING.md
```

Also inspect the relevant frontend documentation and source.

Important interpretation:

- `Requirements 142.txt` = **target requirements**
- current docs = **what the code currently does / previous implementation status**
- `CLAIMS.md` = **scientific honesty / claims guardrail**

If a current document describes a narrower 4-band/two-model product, that is NOT permission to redefine the project as 4-band/two-model.

Instead, use it to understand the current gap.

Do not silently modify the requirements document.

---

# 3. CRITICAL CURRENT-PROJECT PROBLEM

The frontend integration that happened recently narrowed the user-facing product to roughly:

```text
upload GeoTIFF
    ↓
run SR
    ↓
inspect/download result
```

This is NOT the intended final FRAME workflow.

The requirements explicitly envision a system that starts with an interactive geographic workflow:

```text
Interactive map
    ↓
AOI selection
    ↓
date/date-range
    ↓
cloud filtering
    ↓
real Sentinel-2 scene discovery
    ↓
scene selection
    ↓
real satellite data acquisition
    ↓
preprocessing/harmonisation
    ↓
SR
    ↓
geospatial result
    ↓
map visualization / analysis / evidence
```

**Do not remove the map in favor of manual upload.**

Manual GeoTIFF upload may remain as a secondary/fallback/testing pathway.

The **map/AOI workflow must become the primary pathway**.

---

# 4. CURRENT GIT / FRONTEND STATE — BE CAREFUL

The repository currently contains a large set of frontend deletions/modifications from an earlier integration.

There is also an untracked/reference frontend directory:

```text
Frontend-refrence/
```

and the current frontend contains newly added components such as:

```text
frontend/src/components/landing/LandingPage.tsx
frontend/src/components/layout/Header.tsx
frontend/src/components/workspace/
frontend/src/components/reconstruct/
frontend/src/components/hero/
frontend/src/components/projects/
frontend/src/components/export/
frontend/src/components/docs/
...
```

Before making changes:

```bash
git status --short
```

Understand what has been deleted, what is the current implementation, and what exists in `Frontend-refrence/`.

**Do not blindly delete existing frontend components.**

**Do not replace the complete frontend with a smaller upload-only interface.**

Use the existing/reference frontend to recover and preserve useful UI/features while connecting them to real backend capabilities.

Do not make a wholesale visual redesign unless necessary.

---

# 5. COPERNICUS CREDENTIALS ARE ALREADY CONFIGURED

The user has already created a Copernicus Data Space OAuth client using:

```text
Supported flow: Client Credentials
Single-page app: OFF
```

The root project now has a local `.env`.

The real credentials are already present in that `.env`.

Expected variables:

```env
FRAME_CDSE_CLIENT_ID=
FRAME_CDSE_CLIENT_SECRET=
FRAME_CDSE_STAC_URL=https://stac.dataspace.copernicus.eu/v1
FRAME_CDSE_PROCESS_URL=https://sh.dataspace.copernicus.eu/process/v1
FRAME_CDSE_TOKEN_URL=https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token
```

### SECURITY RULES

- NEVER commit `.env`.
- NEVER copy `FRAME_CDSE_CLIENT_SECRET` into frontend code.
- NEVER expose the client secret to the browser.
- NEVER print the secret in logs.
- NEVER return OAuth credentials through a public API endpoint.
- Backend-only authentication.
- Use environment-variable loading such as `python-dotenv` or the project's existing configuration mechanism.
- Preserve a safe `.env.example` with variable names only.
- Do not hard-code credentials.
- Do not put secrets in Git, TypeScript, Vite client variables, public assets, or markdown documentation.

The root `.gitignore` should keep `.env` ignored.

---

# 6. THE COPERNICUS INTEGRATION HAS ALREADY BEEN VERIFIED

Do not waste time re-proving the APIs unless your implementation needs a diagnostic test.

The following have already been tested successfully from the user's machine:

### OAuth

Client Credentials authentication returned a valid token.

### STAC collection

The following works:

```text
https://stac.dataspace.copernicus.eu/v1/collections/sentinel-2-l2a
```

and returns:

```text
sentinel-2-l2a
Sentinel-2 Level-2A
```

### Real AOI/date/cloud search

A real STAC search using:

- AOI bbox
- date range
- `eo:cloud_cover <= 20`

returned real Sentinel-2 L2A scenes.

### Real Process API retrieval

The following works:

```text
https://sh.dataspace.copernicus.eu/process/v1
```

using:

```text
sentinel-2-l2a
```

and returns real GeoTIFF pixels.

### Real 12-band test

A 256 × 256 Float32 GeoTIFF was successfully retrieved with these requested Sentinel-2 bands:

```text
B01
B02
B03
B04
B05
B06
B07
B08
B8A
B09
B11
B12
```

with:

```text
B10 excluded
```

Rasterio confirmed:

```text
bands: 12
dtype: Float32
CRS: EPSG:4326
```

This proves the CDSE data-access layer is technically available.

Do not recreate this as a fake/mock API.

---

# 7. REQUIREMENT 1 — REAL SENTINEL-2 ACQUISITION

Implement a real acquisition layer.

The requirements recommend:

```text
Copernicus Data Space STAC
        ↓
PySTAC Client / HTTP STAC access
        ↓
Sentinel-2 L2A scene discovery
        ↓
Rasterio/GDAL ingestion
```

The recommended primary source is:

```text
Copernicus Data Space
```

Bhoonidhi is an additional Indian provider/access route, but do not fake it if credentials/access are not available.

Design the acquisition layer so another provider can be added later.

A provider abstraction is desirable, for example:

```text
SatelliteProvider
    ├── CopernicusProvider
    └── BhoonidhiProvider (only when real access is implemented)
```

---

# 8. PRIMARY USER WORKFLOW — MAP FIRST

The frontend must support this as the main workflow:

```text
Open FRAME
    ↓
Interactive map
    ↓
Draw/edit AOI
    ↓
Choose date or date range
    ↓
Choose cloud threshold
    ↓
Search Sentinel-2 L2A
    ↓
Display real candidate scenes
    ↓
Display scene footprints on map
    ↓
Select a scene
    ↓
Show real scene metadata
    ↓
Acquire required data
    ↓
Preprocess / harmonise
    ↓
Choose verified SR model
    ↓
Run SR
    ↓
Show original + SR on map
    ↓
Show analysis/evidence layers
    ↓
Download GIS-ready output
```

### AOI controls

Support:

- pan/zoom
- rectangle AOI
- polygon AOI
- edit AOI
- clear AOI
- coordinate display
- date/date-range
- cloud threshold
- scene search
- candidate scene selection

Do not use fake scene IDs, fake footprints, fake catalogue data, or hard-coded example scenes.

Every candidate scene visible in the product must come from an actual backend API response.

---

# 9. STAC SEARCH API TO BUILD

Create a real backend endpoint, using the existing FastAPI architecture rather than creating a second unrelated server.

Conceptually:

```http
POST /satellite/search
```

Request:

```json
{
  "aoi": {
    "type": "Polygon",
    "coordinates": []
  },
  "datetime": {
    "from": "2026-09-01T00:00:00Z",
    "to": "2026-09-27T23:59:59Z"
  },
  "cloud_cover_max": 20
}
```

or an equivalent clean schema.

Backend should query:

```text
STAC:
https://stac.dataspace.copernicus.eu/v1

Collection:
sentinel-2-l2a
```

Use spatial filtering via `intersects`/GeoJSON when polygon AOIs are supplied.

Use `bbox` only when appropriate.

Support:

- spatial filtering
- temporal filtering
- cloud-cover filtering
- collection filtering
- sorting/selection
- pagination where useful

---

# 10. SCENE RESULT DATA

Each returned scene should preserve enough metadata for the frontend and provenance system.

At minimum:

```text
STAC item ID
provider
collection
product ID where available
acquisition datetime
tile ID where available
orbit where available
geometry / footprint
bbox
cloud cover
processing baseline/version where available
asset references
```

The frontend should display real metadata, not invented values.

---

# 11. TWO-STAGE QUALITY FILTERING

Requirements 142 explicitly distinguish scene-level filtering from AOI-level quality.

Do NOT treat:

```text
eo:cloud_cover <= X
```

as the final cloud mask.

Implement conceptually:

```text
Stage 1
scene-level cloud filtering
        ↓
Stage 2
AOI-specific quality evaluation
        ↓
Stage 3
SCL/cloud/shadow/etc. masking
```

Where the selected L2A product provides relevant quality layers, use them.

Preserve the validity mask instead of physically deleting invalid pixels from spatial arrays.

---

# 12. ACQUISITION ENDPOINT

Add a real scene acquisition endpoint, e.g.:

```http
POST /satellite/acquire
```

It should take the selected scene and the requested AOI/processing specification.

The implementation should avoid downloading a complete Sentinel-2 product when a smaller AOI/band-specific request is sufficient.

The intended structure is:

```text
selected STAC Item
        ↓
determine required data
        ↓
authenticated retrieval
        ↓
temporary/local raster
        ↓
validation
        ↓
preprocessing
```

Add:

- retry/error handling
- timeout handling
- useful error responses
- cleanup of temporary files
- provenance information

Do not make the browser call the authenticated satellite API directly.

---

# 13. REQUIREMENT 2 — 12-BAND MULTISPECTRAL INPUT

The required L2A band set from Requirements 142 is:

### Native 10 m

```text
B02
B03
B04
B08
```

### Native 20 m

```text
B05
B06
B07
B8A
B11
B12
```

### Native 60 m

```text
B01
B09
```

### Explicitly excluded

```text
B10
```

The system must preserve explicit Sentinel-2 band names/order.

Do not call the raster bands `band1`, `band2`, etc. in the domain model without preserving their Sentinel-2 identities.

---

# 14. VERY IMPORTANT — 12-BAND INPUT IS NOT THE SAME AS 12-BAND LEARNED SR

Do NOT make this scientific mistake:

```text
We retrieved 12 bands
        ↓
therefore our SR model super-resolves 12 bands
```

That is false unless the actual executable model contract supports it.

The current integrated Mamba/Lite SR path in FRAME is known to operate on the RGBN group:

```text
B04
B03
B02
B08
```

with a 4× SR output.

Therefore explicitly separate:

```text
12-band acquisition
        ↓
multispectral preprocessing/harmonisation
        ↓
model-specific supported channels
        ↓
actual learned SR outputs
        ↓
contextual/resampled bands if applicable
```

Do not fake B01/B09 2.5 m learned outputs.

Do not modify a model's input channel count without proving the architecture and weights actually support it.

Do not claim "12-band SR" just because a 12-band tensor exists.

---

# 15. REQUIREMENT 2 — PREPROCESSING / HARMONISATION

Implement or integrate the actual preprocessing pipeline for Sentinel-2 L2A.

Requirements include:

```text
radiometric scaling
processing-baseline-aware offset handling
cloud/shadow/quality masking
10 m / 20 m / 60 m resolution handling
registration/co-registration
band ordering
validity masks
provenance
```

### Radiometry

Do not blindly assume:

```text
DN / 10000
```

is universally correct.

Inspect product metadata / processing baseline and apply the correct L2A radiometric conversion.

Keep the conversion explicit and testable.

### Native resolution handling

Do not pretend:

```text
20 m → 10 m interpolation
```

creates genuine 10 m information.

The data model should preserve native-resolution provenance.

### Harmonised representation

Choose the common tensor grid only after auditing the existing preprocessing/model requirements.

Document exactly which operations are:

- true retrieval
- resampling
- harmonisation
- learned super-resolution

These are not interchangeable terms.

---

# 16. MODEL AUDIT — DO NOT INVENT THE MODEL SET

Requirements 142 investigated:

```text
SEN2SR / Mamba
SEN2SR-Lite
LDSR-S2
Residual CNN baseline
Bicubic baseline
```

The current repository historically exposed only the verified Lite and Mamba systems.

There is a local environment:

```text
ldsrs2_venv
```

Therefore you MUST inspect the repository before deciding what is actually available.

Search for:

```text
ldsrs2_venv
LDSR
LDSR-S2
opensr-model
model weights
checkpoint files
loader implementations
existing baseline code
```

If LDSR-S2 is actually available and can be integrated correctly:

- add it as a real selectable model/system
- verify its input/output contract
- verify its weights
- verify its environment
- verify inference

If it is NOT actually available:

- do not create a fake LDSR button
- document the missing artifact/environment
- leave the capability explicitly unavailable

Likewise preserve:

```text
Bicubic
```

as a real non-neural baseline.

Do not label a model 12-band unless it truly is.

---

# 17. CURRENT MAMBA / LITE IMPLEMENTATION MUST BE PRESERVED

The existing Mamba path is an actual executable integrated worker.

It uses a strict architecture/checkpoint contract.

Do not replace it with a placeholder.

The existing Mamba/Lite functionality should remain available while the new satellite acquisition layer is added.

Do not break the current manual GeoTIFF route.

---

# 18. DO NOT TOUCH `sen2sr/` UNNECESSARILY

The original upstream/package area:

```text
sen2sr/
```

has intentionally been kept untouched.

Avoid modifying it.

If an unavoidable change is truly required, first prove why.

Prefer wrapping/integrating existing functionality from outside that directory.

Do not rewrite upstream code to simplify your implementation.

---

# 19. MAP RENDERING ARCHITECTURE

Requirements 142 explicitly propose:

```text
React + TypeScript + Vite
MapLibre GL JS
FastAPI + Pydantic
COG
TiTiler / raster tile service
GDAL
Rasterio
GeoPandas / Shapely
STAC metadata
```

Use the existing frontend stack where possible.

For large raster visualization:

```text
FRAME output
    ↓
COG
    ↓
TiTiler / raster tile endpoint
    ↓
PNG/WebP map tiles
    ↓
MapLibre
```

A browser should NOT load a huge 12-band Float32 GeoTIFF directly just to display the map.

If an existing map/tile mechanism already exists, reuse it.

If it does not, implement a minimal real mechanism.

Do not claim "COG/TiTiler support" unless it actually works.

---

# 20. MAP LAYERS / VIEWER MODES

Requirements 142 define four primary viewer modes:

### Mode 1 — Original

```text
10 m Sentinel-2
```

### Mode 2 — Super-resolved

```text
2.5 m reconstructed product
```

### Mode 3 — Analytical

Potentially:

```text
NDVI
NDBI
NDWI
BSI
land cover
change
```

### Mode 4 — Evidence

Potentially:

```text
model stability
high-risk regions
validation
```

The evidence layer is important because FRAME should not communicate reconstructed details as if every pixel were equally certain.

---

# 21. ANALYTICAL PRODUCTS

Requirements 142 ask whether SR improves actual geospatial analysis rather than merely visual sharpness.

Potential layers include:

```text
NDVI
NDBI
NDWI
BSI
change detection
crop mapping
urban mapping
land-cover analysis
flood analysis
burn-scar analysis
```

Implement a derived layer only when the required data and formula are actually available.

For example:

```text
NDVI = (B08 - B04) / (B08 + B04)
NDBI = (B11 - B08) / (B11 + B08)
```

Use correct band provenance, masks, nodata handling and the documented formula.

Do not fabricate classification labels or model accuracy tables.

If a task requires a real labeled dataset that is not present, expose the capability as pending/unavailable rather than inventing results.

---

# 22. STAC-STYLE PROVENANCE FOR EVERY RESULT

Requirements 142 emphasize traceability.

For each processed result preserve at least:

### Source

```text
provider
STAC item ID
collection
product ID
tile ID
acquisition datetime
AOI
```

### Input

```text
band set
native resolution
processing baseline/version
CRS
dimensions
asset identifiers
```

### Processing

```text
preprocessing version
masking method
resampling method
harmonisation method
SR model
model version
weights/checkpoint hash
SR factor
```

### Output

```text
output resolution
CRS
transform
bounds
dimensions
nodata
band names
```

### Validation / evidence

```text
metrics where available
stability method
number of passes
stability metric
```

This should be reflected in the API result and persisted metadata.

---

# 23. MANUAL UPLOAD MUST REMAIN A FALLBACK

The existing upload route is still useful for:

- local test TIFFs
- benchmark datasets
- offline evaluation
- debugging
- demonstrations without network access

Keep it.

But it must NOT be the default product experience.

The default entry should be:

```text
MAP / AOI → SCENE SEARCH → SCENE SELECTION
```

---

# 24. FRONTEND RESTORATION RULE

Review `Frontend-refrence/` and the current frontend before adding new components.

Restore useful features that were previously removed by the narrowed integration when they can be connected to real backend capabilities.

Do not restore fake functionality.

In particular:

- real tools should stay
- fake data should not return
- fake scene catalogues must not return
- fake PSNR tables must not return
- fake land-cover metrics must not return
- fake analysis values must not return
- fake queue/progress data must not return

The goal is:

> **full product experience backed by real APIs and real data, not a visually impressive fake.**

---

# 25. API DESIGN SHOULD BE CLEAN

Prefer a clear separation:

```text
/satellite/*
    discovery / metadata / acquisition

/sr/*
    inference

/analysis/*
    derived products

/maps/* or /tiles/*
    visualization

/models
    verified model registry

/health
    system status
```

Do not create parallel competing APIs for the same operation.

Reuse existing endpoint conventions where practical.

---

# 26. UI STATE MUST REFLECT REAL BACKEND STATE

Do not fake:

```text
Loading
Running
Complete
Failed
```

unless those states correspond to actual backend operations.

For long-running processing, use a real job/status architecture if needed.

A synchronous implementation is acceptable for a small prototype if honestly documented, but don't pretend a queue exists if it doesn't.

---

# 27. PERFORMANCE / RESOURCE MANAGEMENT

The user's machine is an RTX 3050 Laptop GPU with 4 GB VRAM.

Do not design the first implementation assuming an 80 GB GPU.

Use:

- bounded AOI requests
- tiled inference
- memory-aware processing
- CPU fallback where the existing code supports it
- cleanup of temporary files
- request size limits
- sensible caching

Do not automatically download entire Sentinel-2 SAFE archives unless genuinely needed.

Do not download large training datasets just to prove the API connection.

---

# 28. VALIDATION REQUIREMENTS

Once the acquisition/map integration is complete, create real tests for:

### Acquisition

- valid AOI
- invalid AOI
- date filtering
- cloud filtering
- no-result case
- real scene metadata
- provider failure

### Raster

- valid 12-band input
- expected band names/order
- B10 excluded
- native-resolution provenance
- CRS
- transform
- bounds
- nodata/validity mask

### Map

- AOI draw/edit/clear
- search
- scene footprints
- scene selection
- original layer
- SR layer
- layer switching

### Security

- `.env` ignored
- credentials not present in frontend bundle
- credentials not returned by API
- credentials not written to logs

### End-to-end

At minimum:

```text
Map
 ↓
AOI
 ↓
STAC
 ↓
real scene
 ↓
real acquisition
 ↓
validation
 ↓
preprocessing
 ↓
SR
 ↓
georeferenced result
 ↓
map visualization
```

Use a small AOI for real integration tests.

---

# 29. ACCEPTANCE CRITERIA — PHASE 1 OF THIS TASK

Do not declare the work complete until these work for a real test scene:

### A. Map opens

FRAME opens to a real interactive map.

### B. AOI works

User can draw/edit/clear an AOI.

### C. Search is real

User sets:

```text
AOI
date range
cloud threshold
```

and FRAME returns real Sentinel-2 L2A results from CDSE.

### D. Footprints are real

Returned scene geometries are shown on the map.

### E. Scene selection is real

Selecting a scene displays its real metadata.

### F. Acquisition is real

FRAME retrieves actual pixels through the backend.

### G. Multispectral validation is real

Required 12-band acquisition is represented correctly:

```text
B01 B02 B03 B04
B05 B06 B07 B08A
B08 B09 B11 B12
```

with:

```text
B10 excluded
```

and native-resolution provenance retained.

### H. No fake model claims

Model capabilities shown in the UI must match actual verified executables.

### I. Existing SR remains functional

The existing manual RGBN SR workflow must still work.

### J. Map shows actual data

The map can display the actual acquired original raster, not a fake placeholder.

---

# 30. ACCEPTANCE CRITERIA — PHASE 2

After Phase 1 is stable, extend the same real scene into:

```text
selected Sentinel-2 scene
        ↓
preprocessing/harmonisation
        ↓
verified SR model
        ↓
2.5 m output where the model actually supports it
        ↓
georeferenced result
        ↓
COG/tile layer
        ↓
MapLibre viewer
```

Then add:

```text
analytical layers
stability/evidence
metadata/provenance
downloads
```

---

# 31. DO NOT MAKE THESE CLAIMS

Do NOT claim any of the following unless verified by executable code + tests:

```text
"12-band 2.5 m learned SR"
"all Sentinel-2 bands super-resolved"
"uncertainty calibrated"
"hallucination-free"
"three SR models integrated"
"real-time processing"
"large-scale production deployment"
"Bhoonidhi supported"
"TiTiler supported"
"batch processing supported"
```

Scientific wording must match actual capability.

---

# 32. IMPORTANT DISTINCTION ABOUT REQUIREMENTS VS IMPLEMENTATION

Requirements 142 intentionally covers more than what has already been implemented.

You must NOT "complete" the requirements by fabricating features.

Instead:

```text
Requirement
    ↓
Current implementation audit
    ↓
Real implementation
    ↓
Test
    ↓
Documentation update
```

If a feature cannot yet be completed, record it explicitly as a gap.

Do not quietly downgrade the requirement.

---

# 33. IMPLEMENTATION ORDER

Work in this order to avoid repeating the previous integration mistake.

## Phase A — Audit

1. Read `Requirements 142.txt`.
2. Audit current backend/frontend.
3. Audit current docs.
4. Audit `Frontend-refrence/`.
5. Audit current model registry.
6. Identify which code/features were removed.
7. Write a concrete gap list.

Do not modify anything until the audit is understood.

## Phase B — Satellite acquisition

Implement:

```text
/satellite/search
/satellite/scenes/{id} or equivalent
/satellite/acquire
```

using real CDSE access.

## Phase C — Map/AOI

Connect the frontend:

```text
MapLibre
 ↓
AOI
 ↓
date/cloud controls
 ↓
real search
 ↓
real footprints
 ↓
scene selection
```

## Phase D — Multispectral input

Implement the real 12-band ingestion/validation/harmonisation pathway.

## Phase E — SR integration

Connect verified model(s) without overstating band capability.

## Phase F — Raster result map

Serve original/SR/stability/analysis layers through a suitable raster tile mechanism.

## Phase G — Evidence / analysis / provenance

Add the requirements-driven analytical and evidence layers.

## Phase H — Tests and documentation

Only after functionality works:

- update documentation
- update traceability
- update release readiness
- update claims
- add reproducibility notes

---

# 34. IMPORTANT: DO NOT COMMIT

Do not commit or push changes.

After implementation/testing, leave the working tree for the user to inspect.

Always report:

```text
git status --short
```

and summarize:

- files added
- files modified
- files deleted
- tests executed
- known limitations

Do not force-push anything.

Do not rewrite `main`.

Do not delete the existing backup branch.

---

# 35. FINAL DIRECTIVE

The objective is NOT:

> "Make the current upload page prettier."

The objective is:

> **Turn FRAME into the actual map-first Sentinel-2 SRM system described by Requirements 142, using real Copernicus Data Space APIs, real AOI/scene discovery, real multispectral acquisition, correct resolution-aware preprocessing, verified SR models, geospatially correct outputs, and an honest analytical/evidence layer.**

The desired high-level architecture is:

```text
                        FRAME
                          │
                 ┌────────▼────────┐
                 │  Interactive Map │
                 │    MapLibre      │
                 └────────┬────────┘
                          │
                   AOI / date / cloud
                          │
                          ▼
                ┌───────────────────┐
                │   FRAME FastAPI   │
                └─────────┬─────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
         CDSE STAC              Process API
              │                       │
        scene discovery         real pixels
              │                       │
              └──────────┬────────────┘
                         ▼
                  Sentinel-2 L2A
                         │
                         ▼
              12-band validation
                         │
                         ▼
            preprocessing/harmonisation
                         │
                         ▼
                 verified SR model
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
           2.5m       stability    analysis
             │           │           │
             └───────────┼───────────┘
                         ▼
                    COG / tiles
                         │
                         ▼
                   MapLibre viewer
                         │
                         ▼
                 GIS-ready export
```

**Start with the audit. Do not strip the frontend down again.**
