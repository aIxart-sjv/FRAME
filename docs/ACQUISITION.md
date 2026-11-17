# Sentinel-2 acquisition (`frame/acquisition/`, requirements-recovery pass)

Real Sentinel-2 scene discovery and pixel retrieval, as a tested library **and** a real FastAPI HTTP layer. This
module answers Requirement 1 (`docs/Requirements 142.txt`): AOI + date range + cloud threshold → real STAC scene
metadata → a scene the caller selects → real, provenance-tracked pixels harmonised onto one grid. There is still no
frontend UI for any of this — see §6.

```
POST /acquisition/search {bbox | aoi_geojson, start_date, end_date, max_cloud_cover}
        │
        ▼
frame.acquisition.search.search_scenes            -- real CDSE STAC search, collection "sentinel-2-l2a"
        │
        ▼
[SceneCandidate, ...]  -- cached under a search_id; real id, footprint, cloud cover, tile, datetime, asset hrefs
        │  GET /acquisition/scenes/{search_id}/{scene_id}  (caller selects one)
        ▼
POST /acquisition/retrieve {search_id, scene_id, bands, resolution_m, include_scl}
        │
        ▼
frame.acquisition.retrieve.retrieve_scene
        │
        ├─▶ frame.acquisition.providers.cdse            (primary: Sentinel Hub Process API, OAuth2, real credentialed pixels)
        └─▶ frame.acquisition.providers.planetary_computer (fallback: cubo / Microsoft Planetary Computer, credential-free)
        │
        ▼
RetrievedScene -- bands harmonised onto one grid, native resolution + provider kept per band, full provenance
        │
        ▼
real GeoTIFF artifact(s) on disk, downloadable via GET /acquisition/download/{retrieval_id}/{bands,scl}
```

## 1. What is and is not in scope

| Item | Status |
|---|---|
| Real CDSE STAC search (AOI/date/collection), client-side cloud-cover filtering, a named error rather than a silent empty result | implemented |
| Real, credentialed pixel retrieval via the CDSE Sentinel Hub Process API (primary provider) | implemented |
| Real, credential-free pixel retrieval via Microsoft Planetary Computer (`cubo`), as a fallback | implemented |
| Multi-resolution harmonisation (10 m / 20 m / 60 m bands + 20 m SCL onto one grid); bilinear for reflectance, nearest-neighbour for SCL | implemented |
| Per-band native-resolution and per-band provider provenance, with resampling/provider-discrepancy warnings disclosed, not absorbed | implemented |
| Real FastAPI layer: `POST /acquisition/search` (bbox or GeoJSON AOI), `GET /acquisition/scenes/{search_id}/{scene_id}`, `POST /acquisition/retrieve`, `GET /acquisition/download/{retrieval_id}/{bands,scl}` | implemented |
| Clean, typed HTTP error codes for invalid AOI/query, provider outage, authentication failure, no-scenes, unknown scene/search/retrieval id | implemented |
| Frontend map/AOI UI | **not done** |
| Feeding a retrieved scene into the existing `/sr/run` pipeline (a real 4-band RGBN subset, registered as an `UploadRecord`) | **not done** |
| Bhoonidhi (the Indian-specific acquisition route named in Requirements 142 §1) | **not attempted** — requires separate Indian-government registration/credentials this environment does not have |
| Cloud/shadow masking beyond the scene-level `eo:cloud_cover` filter (AOI-specific SCL-based quality evaluation) | partially implemented — SCL is retrieved and harmonised, but no AOI-level quality-decision logic consumes it yet |

## 2. Why two providers, and why CDSE is primary

CDSE's own STAC catalogue is the real, authoritative discovery source (`frame.acquisition.search`); its own asset
hrefs, however, are `s3://eodata/...` paths that need separate S3 credentials this project does not have, so they
are never fetched directly. Real pixel retrieval instead goes through the Sentinel Hub **Process API**
(`https://sh.dataspace.copernicus.eu/process/v1`), authenticated with the OAuth2 client-credentials the project's
`.env` provides (`FRAME_CDSE_CLIENT_ID`/`FRAME_CDSE_CLIENT_SECRET`, git-ignored, never logged, never returned by any
endpoint, never sent to the frontend). Requesting bands with `units: "REFLECTANCE"` also means Sentinel Hub applies
the correct per-processing-baseline radiometric offset itself — this sidesteps the flat-DN/10000 pitfall Requirements
142 §14 warns against, for CDSE-served bands specifically.

Microsoft Planetary Computer, via the already-installed `cubo` dependency (previously used only in
`experiments/baseline/run_baseline.py`), is the fallback when CDSE is unconfigured or a request fails. It serves the
same underlying Sentinel-2 archive but returns **raw digital numbers**, not reflectance (verified empirically:
values in the thousands) — `frame.acquisition.providers.planetary_computer` converts these with this project's one
existing, tested DN-to-reflectance convention (`frame.preprocessing.reflectance.to_reflectance`, a flat divide by
10,000, no baseline-offset correction). This is a real, small, disclosed discrepancy against CDSE-served bands for
scenes processed under baseline ≥ 04.00: `RetrievedScene.provenance.warnings` names it explicitly whenever the
fallback provider serves a reflectance band group. It is never silently absorbed into a single "the pixels" number.

`RetrievedScene.provenance.band_sources` names, per band, exactly which provider served it — a scene retrieved
partly from CDSE and partly from the fallback (e.g. CDSE succeeded for the 10 m group but its token request failed
for the 20 m group) is never presented as if one provider served the whole thing.

## 3. Real acquisition window, not the full search range

A candidate is retrieved from a ±12 hour window around **its own** `datetime`, never the full search date range —
Sentinel-2's ~5-day revisit means this window contains at most the one real acquisition the candidate names, so
retrieval can never silently substitute a different date's scene for the one actually selected.

## 4. Running the tests

```bash
pytest frame/tests/test_acquisition_contract.py frame/tests/test_acquisition_search.py \
       frame/tests/test_acquisition_cdse.py frame/tests/test_acquisition_geometry.py \
       frame/tests/test_acquisition_retrieve.py frame/tests/test_acquisition_planetary_computer.py -q
# 78 tests, all fixture-backed -- no network access or credentials required.

# Live, network-touching, against the real Copernicus API -- skipped automatically unless
# FRAME_CDSE_CLIENT_ID/SECRET are set (this project's .env is git-ignored, never assumed in CI):
set -a && source .env && set +a && pytest frame/tests/test_acquisition_live.py -q
```

The live test was run in this pass against a real ~330 m AOI near Chennai: real STAC search returned a real scene
(`cloud_cover <= 20`), and real retrieval + harmonisation produced B01/B02/B03/B04/B08/B11 plus SCL on one grid with
correct per-band native resolutions and plausible reflectance values (0 ≤ mean(B04) < 1) — see the module's own test
file for the exact assertions; the numbers themselves are not reported here because they depend on whatever scene is
returned at run time (the catalogue can drift — see `experiments/end_to_end/README.md` for a prior, unrelated
instance of the same drift).

## 5. The HTTP layer (`frame/api/routes.py`, `frame/api/services/acquisition.py`)

Thin glue over the library above, following the same conventions as every other FRAME API route (`frame.api.schemas`
Pydantic models, `frame.api.services.storage.JobStore` for process-lifetime state, errors mapped centrally in
`frame.api.errors.status_code_for`/`app.py`'s single generic exception handler — no second API architecture).

| Endpoint | What it does |
|---|---|
| `POST /acquisition/search` | Real CDSE STAC search. Accepts `bbox` **or** `aoi_geojson` (a GeoJSON Feature/Polygon/MultiPolygon, reduced to its bounding box — see §2's caveat), `start_date`/`end_date`, `max_cloud_cover`, `max_results`. Returns real `SceneCandidate`s (id, provider, collection, datetime, cloud cover, bbox, **real GeoJSON footprint geometry**, MGRS tile, processing baseline, asset count) cached under a `search_id`. Zero real matches is a normal **200** response (`scene_count: 0`, a `note` explaining why), never an error. |
| `GET /acquisition/scenes/{search_id}/{scene_id}` | The same real scene record, looked up by id. Never returns the raw `asset_hrefs` (those are CDSE's own `s3://eodata/...` paths — irrelevant to a caller that can't use S3 credentials this server doesn't have). |
| `POST /acquisition/retrieve` | Real multi-resolution retrieval + harmonisation for a previously-searched scene (`search_id` + `scene_id` + `bands` + `resolution_m` + `include_scl`). Writes a real multi-band reflectance GeoTIFF (and a real single-band SCL GeoTIFF, class codes, if requested) to the workspace directory, and returns per-band native resolution, per-band provider (`band_sources`), resampling/provider-discrepancy warnings, and download links. **Not** wired to `/sr/run` — see §6. |
| `GET /acquisition/download/{retrieval_id}/bands` \| `/scl` | The real GeoTIFF artifact(s) from a retrieval. |

Named error codes (`{"error", "code", "detail"}`, matching every other FRAME endpoint):

| Condition | HTTP status | `code` |
|---|---|---|
| Bad AOI/date/cloud-cover value (e.g. `min_lon > max_lon`) | 400 | `invalid_acquisition_query` |
| Neither `bbox` nor `aoi_geojson` given | 422 | `request_validation_error` (FastAPI's own body-validation error) |
| A real provider is unreachable, unconfigured, or returned a non-2xx response | 503 | `acquisition_provider_unavailable` |
| A provider rejected the configured credentials (HTTP 401/403) | 401 | `acquisition_authentication_failed` |
| An unknown `search_id` | 404 | `acquisition_search_not_found` |
| A `scene_id` not part of that search's real results | 404 | `acquisition_scene_not_found` |
| A retrieval provider's response did not match the requested scene | 502 | `acquisition_retrieval_mismatch` |
| An unknown `retrieval_id`, or an artifact (e.g. SCL) that retrieval did not produce | 404 | `acquisition_retrieval_not_found` / `acquisition_artifact_not_available` |

A zero-result search is deliberately **not** in this table: `NoScenesFoundError` is caught inside
`frame.api.services.acquisition.search` and turned into a normal, real, empty `AcquisitionSearchResponse` — a search
that legitimately finds nothing is not the same kind of event as a provider outage, and conflating the two would make
a frontend show an error banner for what is often just "try a wider date range."

Tests: `frame/tests/test_api_acquisition.py` (15 HTTP-layer tests, FastAPI `TestClient`, fixture-backed — the two
real provider calls are monkeypatched at `frame.api.services.acquisition.real_search_scenes`/`real_retrieve_scene`,
the names that module binds them to at import time). `frame/tests/test_api_errors.py` covers the new status/code
mapping in isolation.

### 5.1 Verified sufficient for Step D (the map/AOI frontend)

Checked directly against a real live CDSE search result before Step C was called done: `SceneCandidateSchema`
carries a real, valid GeoJSON `Polygon` footprint (5-point closed ring, real coordinates), a real bbox, a real MGRS
tile (`"MGRS-44PMV"`, copied verbatim — not reformatted), a real processing baseline (`"05.10"`), a real platform
(`"sentinel-2b"`), and a real cloud-cover percentage. `AcquisitionSearchRequest` accepts either a bbox or an
arbitrary GeoJSON AOI. Provider/provenance (which provider discovered the scene, which provider(s) retrieved which
bands, the real STAC/Process API endpoints, resampling and cross-provider-discrepancy warnings) is carried through
`AcquisitionSearchResponse`/`AcquisitionRetrieveResponse`. This is what Step D's map needs to draw AOIs, list real
scenes with real footprints, and show real provenance — nothing here was added speculatively for the map; it is
what the library already had.

## 6. The frontend (`frontend/src/components/reconstruct/{AcquisitionMap,SelectArea,SelectScene,SceneConfirmed}.tsx`)

A real MapLibre GL JS map (`AcquisitionMap.tsx`) drives the product's primary flow: draw a rectangle or polygon
AOI, choose a date range and cloud-cover ceiling, call the real `POST /acquisition/search`, see real scene
footprints and metadata on the map and in a list, select one, and optionally call the real
`POST /acquisition/retrieve` to fetch real pixels (with a bounded, safe target resolution -- see §5.1's note on
`safeRetrievalResolutionM`). Manual GeoTIFF upload (`UploadScene.tsx`) remains fully intact as an explicit
fallback, reachable from the AOI page, never removed or weakened.

Two real bugs surfaced by live testing against the production Copernicus API, both fixed (not just noted):
retrieval used to request every band group at its **native** resolution regardless of the target grid, so a large
AOI at a coarse target resolution still triggered a huge, slow real request; `frame.acquisition.retrieve` now
requests at `max(native_resolution, target_resolution)` per group (and for SCL), so a real request is never
larger than the analyst's chosen output grid actually needs. The frontend also clamps its own suggested
resolution to the backend's accepted range (`AcquisitionRetrieveRequest.resolution_m`, `le=100`).

## 7. Real acquisition → the existing SR pipeline (Step E)

`POST /acquisition/retrieve/{retrieval_id}/prepare-sr-input` (`frame/api/services/acquisition.py`) is the one
adapter between a real retrieval and the existing, **completely unmodified** `POST /sr/run`. It reads the
retrieval's real bands GeoTIFF, validates and subsets it to the 4 RGBN bands (`frame.data.bands.select_bands`,
naming any extra real bands in `excluded_bands` rather than discarding them silently), writes a real RGBN GeoTIFF,
and registers it exactly like a manually-uploaded file (`frame.api.services.pipeline.process_upload`, given one new
optional field, `source_acquisition_retrieval_id`, so the resulting `UploadRecord` still names its real source).
The frontend (`SceneConfirmed.tsx`'s "Run Super Resolution") drops the response straight into the same
`uploadResult`/`uploadStatus` state a manual upload fills, so `ReconstructionSetup`/`ProcessingJob`/`InspectViewer`
needed zero changes. Verified live end-to-end, including real inference on the real RTX 3050 GPU (SEN2SR-Lite,
CUDA, 0.42 s) with correct geospatial pass-through (CRS unchanged, transform pixel size divided by exactly 4,
bounds consistent). See `test_api_acquisition_sr_handoff.py`.

## 8. 12-band L2A harmonisation (Step F, `frame/acquisition/harmonise.py`) -- data correctness, not a new model

Requirements 142 Requirement 2 draws a hard line: **harmonisation is not the same thing as super-resolution.**
`frame.acquisition.harmonise` makes three things explicit that were previously only implicit inside a
`RetrievedScene`:

1. **The real 12-band contract.** `validate_full_l2a_bands` checks a band set against all 12 required L2A bands
   (`frame.data.bands.L2A_BANDS`: B01–B09, B8A, B11, B12) -- B10 is correctly never checked for, since L2A excludes
   it by definition, not by this project's choice.
2. **A real validity mask, never a deletion.** `harmonise()` builds a `frame.preprocessing.masks.ValidityMask` from
   the scene's own real SCL layer (reusing the existing `from_scl`/`DEFAULT_SCL_INVALID_CLASSES` -- no second
   masking convention). Masked pixels stay in the multispectral stack; only a separate boolean mask says which
   ones they are, so spatial correspondence is never destroyed.
3. **Honest per-band radiometric provenance.** `band_radiometric_method` names, per band, the real conversion its
   real source provider actually applied: CDSE's own Process API reflectance units (server-side, correctly
   baseline-aware) versus Planetary Computer's flat DN/10000 (not baseline-aware) -- the same real discrepancy §2
   already discloses, now attached to the harmonised product itself rather than only to a prose warning. A product
   built from more than one provider (e.g. CDSE succeeded for some bands, Planetary Computer for others) reports a
   **different** method per band; nothing here presents one assumed formula for the whole stack.

`select_rgbn_adapter(product)` is the one explicit, tested boundary from this (potentially 12-band) representation
into the existing RGBN SR pipeline -- it returns the 4 RGBN bands in canonical order plus the list of every other
real band the product has, and refuses (raising `ContractError`, never fabricating a band) if even the 4 RGBN
bands are not all present. `POST /acquisition/retrieve`'s response now also carries this harmonisation metadata
directly (`band_radiometric_method`, `is_complete_l2a`, `validity_mask_coverage`), so a caller can see it without
a separate call.

**What this genuinely is:** ordinary, documented, **non-learned** resampling (bilinear for reflectance bands,
nearest-neighbour for the categorical SCL -- never the reverse) onto one common computational grid, exactly
Requirements 142's "Option A -- explicit harmonisation" (§15 there), explicitly **not** the "learned/controlled
harmonisation" or multi-resolution SR architecture ("Option B") the same section also describes. Per §25 of that
requirement: this is a **"12-band Sentinel-2 L2A stack resampled/aligned to a common computational grid"**, never
"12 native N m bands" -- a 20 m or 60 m band resampled onto the 10 m grid does not thereby become a genuine 10 m
observation, and `band_native_resolution_m` keeps saying so per band after harmonisation.

**What this is not:** a second SR implementation, a claim that 12 bands can be learned-reconstructed to 2.5 m, or
a decision about whether B01/B09 should participate in a future multi-resolution reconstruction (Requirements 142
§16–17 explicitly say that needs an experiment, not a default -- none was run here; out of scope for a
data-correctness step that trains no model).

Tests: `test_acquisition_harmonise.py` -- 34 fixture-backed tests (band contract, per-band native-resolution
provenance, resampling discipline including a check that harmonise() contains no interpolation call of its own,
mixed-provider radiometric tagging, mask construction/coverage/non-destructiveness, geospatial pass-through, and a
real integration test: a full 12-band harmonised product → `select_rgbn_adapter` → a real, unmodified `/sr/run`)
plus 1 live test against the real Copernicus API.

## 9. What "Sentinel-2 acquisition" does not yet mean here

Real scenes, real pixels, a real map-first UI, a real handoff into the existing SR pipeline, and real 12-band
harmonisation all now exist, with tests proving each. What still does not exist: a genuine multi-resolution or
12-band-to-2.5m learned SR model (LDSR-S2 is not integrated; the RGBN models are unchanged); a learned/controlled
20 m/60 m→10 m harmonisation step (Requirements 142's "Option A" second half, or "Option B"); SCL-based masking in
the manual `/upload` path (only the acquisition path has real SCL to mask with); and a resolved BOA-offset
correction for the Planetary Computer fallback path. See `docs/RELEASE_READINESS.md` and `docs/TRACEABILITY.md`
Requirements 1–2 for the current, honest state of what remains, and `docs/CLAIMS.md` for the wording rules this
product must keep following as it grows (no claim that all bands are learned at 2.5 m, no claim that Bhoonidhi is
supported, no claim that a polygon AOI is filtered by its real shape rather than its bounding box).
