import { 
  Sentinel2Band, 
  Sentinel2Scene, 
  Project, 
  LandCoverClassDistribution,
  PixelSpectralSample 
} from '../types';

/**
 * Sentinel-2 MSI Bands Specification
 * Per Requirements 142.txt:
 * 12 usable Level-2A surface-reflectance bands: B1–B9, B8A, B11, B12.
 * B10 (1375 nm cirrus absorption band) is excluded from L2A surface reflectance.
 */
export const SENTINEL2_BANDS: Sentinel2Band[] = [
  { id: 'B01', name: 'Coastal aerosol', centralWavelength: 443, nativeResolution: 60, isAnalyticalL2A: true, description: 'Aerosol detection and coastal water mapping' },
  { id: 'B02', name: 'Blue', centralWavelength: 490, nativeResolution: 10, isAnalyticalL2A: true, description: 'Vegetation characteristics, soil-vegetation differentiation' },
  { id: 'B03', name: 'Green', centralWavelength: 560, nativeResolution: 10, isAnalyticalL2A: true, description: 'Green peak reflection, water body detection' },
  { id: 'B04', name: 'Red', centralWavelength: 665, nativeResolution: 10, isAnalyticalL2A: true, description: 'Chlorophyll absorption band, NDVI component' },
  { id: 'B05', name: 'Red Edge 1', centralWavelength: 705, nativeResolution: 20, isAnalyticalL2A: true, description: 'Vegetation classification and stress evaluation' },
  { id: 'B06', name: 'Red Edge 2', centralWavelength: 740, nativeResolution: 20, isAnalyticalL2A: true, description: 'Leaf area index (LAI) and canopy density' },
  { id: 'B07', name: 'Red Edge 3', centralWavelength: 783, nativeResolution: 20, isAnalyticalL2A: true, description: 'Canopy chlorophyll content estimation' },
  { id: 'B08', name: 'NIR', centralWavelength: 842, nativeResolution: 10, isAnalyticalL2A: true, description: 'Near-Infrared canopy plateau, biomass, NDVI' },
  { id: 'B8A', name: 'NIR Narrow', centralWavelength: 865, nativeResolution: 20, isAnalyticalL2A: true, description: 'Atmospheric water vapour absorption reference' },
  { id: 'B09', name: 'Water vapour', centralWavelength: 945, nativeResolution: 60, isAnalyticalL2A: true, description: 'Atmospheric water vapour correction' },
  { id: 'B10', name: 'Cirrus', centralWavelength: 1375, nativeResolution: 60, isAnalyticalL2A: false, description: 'Atmospheric absorption; excluded from L2A surface reflectance' },
  { id: 'B11', name: 'SWIR 1', centralWavelength: 1610, nativeResolution: 20, isAnalyticalL2A: true, description: 'Snow/ice vs cloud differentiation, soil moisture, NDBI' },
  { id: 'B12', name: 'SWIR 2', centralWavelength: 2190, nativeResolution: 20, isAnalyticalL2A: true, description: 'Geological feature identification, built-up surfaces' }
];

export const AVAILABLE_SCENES: Sentinel2Scene[] = [
  {
    id: 'S2B_MSIL2A_20260218T050909_N0512_R019_T44RPR_20260218T081522',
    tileId: 'T44RPR',
    acquisitionDatetime: '2026-02-18 05:09:09 UTC',
    cloudCoverPercentage: 1.8,
    dataCoveragePercentage: 100.0,
    processingBaseline: '05.12',
    crs: 'EPSG:32644 (UTM zone 44N)',
    bbox: [80.12, 12.85, 80.32, 13.15],
    provider: 'Bhoonidhi NRSC/ISRO',
    stacItemId: 'sentinel2-l2a-t44rpr-20260218',
    surfaceConditions: 'Clear sky, dry season, peri-urban corridor'
  },
  {
    id: 'S2A_MSIL2A_20260124T054941_N0511_R090_T43SBT_20260124T091012',
    tileId: 'T43SBT',
    acquisitionDatetime: '2026-01-24 05:49:41 UTC',
    cloudCoverPercentage: 3.4,
    dataCoveragePercentage: 100.0,
    processingBaseline: '05.11',
    crs: 'EPSG:32643 (UTM zone 43N)',
    bbox: [75.65, 30.82, 75.95, 31.08],
    provider: 'Bhoonidhi NRSC/ISRO',
    stacItemId: 'sentinel2-l2a-t43sbt-20260124',
    surfaceConditions: 'Rabi agricultural cycle, smallholder parcel boundaries'
  },
  {
    id: 'S2B_MSIL2A_20251104T104849_N0510_R008_T30SYJ_20251104T142510',
    tileId: 'T30SYJ',
    acquisitionDatetime: '2025-11-04 10:48:49 UTC',
    cloudCoverPercentage: 6.2,
    dataCoveragePercentage: 98.4,
    processingBaseline: '05.10',
    crs: 'EPSG:32630 (UTM zone 30N)',
    bbox: [-0.45, 39.38, -0.22, 39.58],
    provider: 'Copernicus Data Space',
    stacItemId: 'sentinel2-l2a-t30syj-20251104',
    surfaceConditions: 'Post-precipitation wetland water bodies, Albufera lagoon'
  },
  {
    id: 'S2A_MSIL2A_20240812T155821_N0509_R011_T18STJ_20240812T200234',
    tileId: 'T18STJ',
    acquisitionDatetime: '2024-08-12 15:58:21 UTC',
    cloudCoverPercentage: 0.9,
    dataCoveragePercentage: 100.0,
    processingBaseline: '05.09',
    crs: 'EPSG:32618 (UTM zone 18N)',
    bbox: [-78.22, 38.85, -78.08, 38.96],
    provider: 'Copernicus Data Space',
    stacItemId: 'sentinel2-l2a-t18stj-20240812',
    surfaceConditions: 'SEN2NEON SCBI calibration tile; AVIRIS-NG 2.5m reference paired'
  }
];

export const DEFAULT_PROJECTS: Project[] = [
  {
    id: 'proj-chennai-periurban',
    name: 'Chennai Peri-Urban Corridor',
    location: 'Tamil Nadu, India',
    sceneId: 'S2B_MSIL2A_20260218T050909_T44RPR',
    tileId: 'T44RPR',
    acquisitionDate: '2026-02-18',
    crs: 'EPSG:32644',
    status: 'Ready',
    lastModified: '2026-02-20',
    referenceAvailable: false,
    referenceSensorName: 'No HR reference paired (operational evaluation)',
    metrics: {
      psnr: null, // No independent HR reference; displaying null per scientific integrity rules
      ssim: null,
      rmse: null,
      samDegrees: null,
      ergas: null,
      reflectanceBias: -0.002, // From observation-grid reduction (10m downsample comparison)
      synthesisScore: 0.018,
      hallucinationRate: null, // requires independent HR reference
      omissionRate: null,
      improvementRate: null,
      crsPreserved: true,
      affineResidualPx: 0.04,
      boundaryDisplacementMeters: 0.62,
      ttaVariance: 0.0034,
      perturbationConsistencyPercent: 96.8
    }
  },
  {
    id: 'proj-sen2neon-scbi',
    name: 'SEN2NEON Validation Site NEON-SCBI',
    location: 'Virginia, USA (Scientific Benchmark)',
    sceneId: 'S2A_MSIL2A_20240812T155821_T18STJ',
    tileId: 'T18STJ',
    acquisitionDate: '2024-08-12',
    crs: 'EPSG:32618',
    status: 'Ready',
    lastModified: '2026-01-15',
    referenceAvailable: true,
    referenceSensorName: 'AVIRIS-NG Hyperspectral (Harmonized to S2 SRF at 2.5m)',
    metrics: {
      psnr: 32.44, // Benchmark mean from SEN2NEON publication
      ssim: 0.81,
      rmse: 0.024,
      samDegrees: 2.86,
      ergas: 1.48,
      reflectanceBias: 0.0081,
      synthesisScore: 0.021,
      hallucinationRate: 0.32,
      omissionRate: 0.44,
      improvementRate: 0.24,
      crsPreserved: true,
      affineResidualPx: 0.02,
      boundaryDisplacementMeters: 0.41,
      ttaVariance: 0.0028,
      perturbationConsistencyPercent: 97.4
    }
  },
  {
    id: 'proj-punjab-agriculture',
    name: 'Punjab Smallholder Agricultural Parcel',
    location: 'Punjab, India',
    sceneId: 'S2A_MSIL2A_20260124T054941_T43SBT',
    tileId: 'T43SBT',
    acquisitionDate: '2026-01-24',
    crs: 'EPSG:32643',
    status: 'Ready',
    lastModified: '2026-02-10',
    referenceAvailable: false,
    metrics: {
      psnr: null,
      ssim: null,
      rmse: null,
      samDegrees: null,
      ergas: null,
      reflectanceBias: 0.0018,
      synthesisScore: 0.016,
      hallucinationRate: null,
      omissionRate: null,
      improvementRate: null,
      crsPreserved: true,
      affineResidualPx: 0.03,
      boundaryDisplacementMeters: 0.58,
      ttaVariance: 0.0041,
      perturbationConsistencyPercent: 95.2
    }
  },
  {
    id: 'proj-valencia-wetland',
    name: 'Valencia Wetland & Flood Plain',
    location: 'Valencia, Spain (OpenSR-Test Site)',
    sceneId: 'S2B_MSIL2A_20251104T104849_T30SYJ',
    tileId: 'T30SYJ',
    acquisitionDate: '2025-11-04',
    crs: 'EPSG:32630',
    status: 'Ready',
    lastModified: '2025-12-01',
    referenceAvailable: true,
    referenceSensorName: 'PNOA Orthophoto (Resampled to 2.5m)',
    metrics: {
      psnr: 31.92,
      ssim: 0.79,
      rmse: 0.028,
      samDegrees: 3.12,
      ergas: 1.62,
      reflectanceBias: 0.0094,
      synthesisScore: 0.025,
      hallucinationRate: 0.38,
      omissionRate: 0.40,
      improvementRate: 0.22,
      crsPreserved: true,
      affineResidualPx: 0.03,
      boundaryDisplacementMeters: 0.49,
      ttaVariance: 0.0039,
      perturbationConsistencyPercent: 94.6
    }
  }
];

export const MOCK_LAND_COVER_DISTRIBUTION: LandCoverClassDistribution[] = [
  { className: 'Built-up', color: '#E05252', original10mAreaHa: 412.4, sr25mAreaHa: 428.1, deltaPercent: +3.8, boundaryConfidence: 'Moderate' },
  { className: 'Vegetation', color: '#35C98A', original10mAreaHa: 684.2, sr25mAreaHa: 672.6, deltaPercent: -1.7, boundaryConfidence: 'High' },
  { className: 'Cropland', color: '#D69E2E', original10mAreaHa: 890.5, sr25mAreaHa: 894.2, deltaPercent: +0.4, boundaryConfidence: 'High' },
  { className: 'Water', color: '#3182CE', original10mAreaHa: 142.8, sr25mAreaHa: 139.7, deltaPercent: -2.1, boundaryConfidence: 'High' },
  { className: 'Bare soil', color: '#A0AEC0', original10mAreaHa: 218.6, sr25mAreaHa: 213.9, deltaPercent: -2.1, boundaryConfidence: 'Scrutiny Recommended' }
];

export const SAMPLE_PIXEL_SPECTRA: Record<string, PixelSpectralSample> = {
  vegetation: {
    coordinates: [80.214, 13.012],
    landCoverClass: 'Vegetation (Dense Canopy)',
    original10mBands: {
      B01: 0.024, B02: 0.028, B03: 0.048, B04: 0.032, B05: 0.092, B06: 0.284,
      B07: 0.362, B08: 0.395, B8A: 0.408, B09: 0.412, B11: 0.174, B12: 0.082
    },
    sr25mBands: {
      B01: 0.024, B02: 0.027, B03: 0.047, B04: 0.031, B05: 0.095, B06: 0.291,
      B07: 0.370, B08: 0.402, B8A: 0.415, B09: 0.412, B11: 0.170, B12: 0.080
    },
    stabilityScore: 0.94,
    ndviOriginal: 0.85,
    ndviSR: 0.86
  },
  builtUpEdge: {
    coordinates: [80.228, 13.025],
    landCoverClass: 'Built-up / Road Boundary (Sub-pixel mixed cell)',
    original10mBands: {
      B01: 0.118, B02: 0.134, B03: 0.145, B04: 0.158, B05: 0.172, B06: 0.188,
      B07: 0.201, B08: 0.214, B8A: 0.222, B09: 0.228, B11: 0.264, B12: 0.245
    },
    sr25mBands: {
      B01: 0.115, B02: 0.130, B03: 0.142, B04: 0.162, B05: 0.178, B06: 0.194,
      B07: 0.208, B08: 0.220, B8A: 0.230, B09: 0.228, B11: 0.272, B12: 0.252
    },
    stabilityScore: 0.68,
    ndviOriginal: 0.15,
    ndviSR: 0.15
  },
  waterBody: {
    coordinates: [80.205, 12.995],
    landCoverClass: 'Deep Water Surface',
    original10mBands: {
      B01: 0.045, B02: 0.040, B03: 0.032, B04: 0.018, B05: 0.012, B06: 0.008,
      B07: 0.006, B08: 0.005, B8A: 0.004, B09: 0.002, B11: 0.002, B12: 0.001
    },
    sr25mBands: {
      B01: 0.044, B02: 0.039, B03: 0.031, B04: 0.017, B05: 0.011, B06: 0.008,
      B07: 0.006, B08: 0.005, B8A: 0.004, B09: 0.002, B11: 0.002, B12: 0.001
    },
    stabilityScore: 0.98,
    ndviOriginal: -0.56,
    ndviSR: -0.55
  }
};
