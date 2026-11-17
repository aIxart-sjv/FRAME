/**
 * FRAME — Scientific Type Definitions
 * Source of Truth: Requirements 142.txt (SIH26142)
 */

export type ViewerMode = 'original' | 'super-resolved' | 'analytical' | 'evidence';

export type AnalyticalLayer = 'ndvi' | 'ndbi' | 'ndwi' | 'bsi' | 'land-cover' | 'change';

export type ValidationSection = 
  | 'spectral' 
  | 'spatial' 
  | 'detail' 
  | 'stability' 
  | 'reference' 
  | 'generalisation';

export type JobStatus = 
  | 'QUEUED' 
  | 'PREPROCESSING' 
  | 'RECONSTRUCTING' 
  | 'POSTPROCESSING' 
  | 'READY' 
  | 'FAILED';

export interface Sentinel2Band {
  id: string; // e.g. 'B01', 'B02', ... 'B12'
  name: string; // e.g. 'Blue', 'Red Edge 1', 'SWIR 1'
  centralWavelength: number; // in nm
  nativeResolution: 10 | 20 | 60; // in meters
  isAnalyticalL2A: boolean; // B10 is false
  description: string;
}

export interface Sentinel2Scene {
  id: string;
  tileId: string;
  acquisitionDatetime: string;
  cloudCoverPercentage: number;
  dataCoveragePercentage: number;
  processingBaseline: string;
  crs: string;
  bbox: [number, number, number, number]; // [minLon, minLat, maxLon, maxLat]
  provider: 'Copernicus Data Space' | 'Bhoonidhi NRSC/ISRO';
  stacItemId: string;
  previewUrl?: string;
  surfaceConditions?: string;
}

export interface RegionOfInterest {
  id: string;
  name: string;
  bounds: {
    north: number;
    south: number;
    east: number;
    west: number;
  };
  areaHectares: number;
  createdAt: string;
  color?: string;
}

export interface ReconstructionJob {
  id: string;
  sceneId: string;
  tileId: string;
  aoiName: string;
  status: JobStatus;
  currentStep: 'ACQUIRE' | 'PREPROCESS' | 'RECONSTRUCT' | 'EVALUATE' | 'READY';
  submittedAt: string;
  completedAt?: string;
  model: 'SEN2SR-Mamba' | 'SEN2SR-Lite' | 'LDSR-S2 (Reference)' | 'Residual-CNN Baseline';
  targetResolution: '2.5m';
  bandsCount: 12; // B1-B9, B8A, B11, B12 (B10 excluded)
  logs: Array<{
    timestamp: string;
    stage: string;
    message: string;
  }>;
}

export interface PixelSpectralSample {
  coordinates: [number, number];
  landCoverClass: string;
  original10mBands: Record<string, number>; // reflectance values 0.0 - 1.0
  sr25mBands: Record<string, number>;
  referenceBands?: Record<string, number>;
  stabilityScore: number; // 0.0 (unstable) to 1.0 (highly consistent across perturbations)
  ndviOriginal: number;
  ndviSR: number;
}

export interface LandCoverClassDistribution {
  className: 'Built-up' | 'Vegetation' | 'Cropland' | 'Water' | 'Bare soil';
  color: string;
  original10mAreaHa: number;
  sr25mAreaHa: number;
  deltaPercent: number;
  boundaryConfidence: 'High' | 'Moderate' | 'Scrutiny Recommended';
}

export interface ValidationMetrics {
  // Global Reconstruction
  psnr: number | null; // e.g. 32.44 dB or null if pending
  ssim: number | null; // e.g. 0.81 or null
  rmse: number | null;
  samDegrees: number | null; // Spectral Angle Mapper in deg
  ergas: number | null; // Relative dimensionless global error
  reflectanceBias: number | null;
  
  // Detail & Correctness (OpenSR-Test framework)
  synthesisScore: number | null;
  hallucinationRate: number | null;
  omissionRate: number | null;
  improvementRate: number | null;

  // Spatial & Geospatial Alignment
  crsPreserved: boolean;
  affineResidualPx: number | null;
  boundaryDisplacementMeters: number | null;

  // Stability
  ttaVariance: number | null;
  perturbationConsistencyPercent: number | null;
}

export interface Project {
  id: string;
  name: string;
  location: string;
  sceneId: string;
  tileId: string;
  acquisitionDate: string;
  crs: string;
  status: 'Ready' | 'Processing' | 'Draft';
  lastModified: string;
  metrics: ValidationMetrics;
  referenceAvailable: boolean;
  referenceSensorName?: string;
  thumbnailUrl?: string;
}
