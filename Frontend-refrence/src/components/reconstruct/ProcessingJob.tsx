import React, { useState, useEffect } from 'react';
import { CheckCircle2, Clock, Terminal, ArrowRight, AlertTriangle } from 'lucide-react';
import { ReconstructionJob, JobStatus } from '../../types';

interface ProcessingJobProps {
  jobConfig: {
    model: string;
    tileSize: number;
    haloSize: number;
    enableStabilityEvaluation: boolean;
    aoiName: string;
    sceneId: string;
    tileId: string;
  };
  onComplete: (projectData: any) => void;
  onCancel: () => void;
}

export const ProcessingJob: React.FC<ProcessingJobProps> = ({
  jobConfig,
  onComplete,
  onCancel,
}) => {
  const [job, setJob] = useState<ReconstructionJob>({
    id: `JOB-${Math.random().toString(36).substring(2, 9).toUpperCase()}`,
    sceneId: jobConfig.sceneId,
    tileId: jobConfig.tileId,
    aoiName: jobConfig.aoiName,
    status: 'QUEUED',
    currentStep: 'ACQUIRE',
    submittedAt: new Date().toISOString().replace('T', ' ').substring(0, 19) + ' UTC',
    model: jobConfig.model as any,
    targetResolution: '2.5m',
    bandsCount: 12,
    logs: [
      {
        timestamp: '00:00:01',
        stage: 'INIT',
        message: `Registered asynchronous super-resolution task for ${jobConfig.tileId}. Model: ${jobConfig.model}.`,
      },
    ],
  });

  const STEPS: Array<'ACQUIRE' | 'PREPROCESS' | 'RECONSTRUCT' | 'EVALUATE' | 'READY'> = [
    'ACQUIRE',
    'PREPROCESS',
    'RECONSTRUCT',
    'EVALUATE',
    'READY',
  ];

  // Automated step progression simulating real job execution
  useEffect(() => {
    const timer1 = setTimeout(() => {
      setJob((prev) => ({
        ...prev,
        status: 'PREPROCESSING',
        currentStep: 'PREPROCESS',
        logs: [
          ...prev.logs,
          {
            timestamp: '00:00:04',
            stage: 'ACQUIRE',
            message: 'STAC catalogue asset resolved: 12 Level-2A surface-reflectance bands downloaded.',
          },
          {
            timestamp: '00:00:06',
            stage: 'PREPROCESS',
            message: 'Applying SCL mask. Radiometric baseline conversion: (DN - 1000) / 10000. B10 excluded.',
          },
          {
            timestamp: '00:00:08',
            stage: 'HARMONISE',
            message: 'Interpolating 20m & 60m bands to common 10m grid. Tiling into 128x128 patches with 16px halo.',
          },
        ],
      }));
    }, 2000);

    const timer2 = setTimeout(() => {
      setJob((prev) => ({
        ...prev,
        status: 'RECONSTRUCTING',
        currentStep: 'RECONSTRUCT',
        logs: [
          ...prev.logs,
          {
            timestamp: '00:00:12',
            stage: 'INFERENCE',
            message: `Dispatching patches to ${jobConfig.model}. Enforcing low-frequency Fourier hard constraint.`,
          },
          {
            timestamp: '00:00:16',
            stage: 'STITCHING',
            message: 'Reconstructing 512x512 outputs. Weighted distance linear blending applied across overlaps.',
          },
        ],
      }));
    }, 5500);

    const timer3 = setTimeout(() => {
      setJob((prev) => ({
        ...prev,
        status: 'POSTPROCESSING',
        currentStep: 'EVALUATE',
        logs: [
          ...prev.logs,
          {
            timestamp: '00:00:20',
            stage: 'STABILITY',
            message: 'Executing 6-pass Test-Time Augmentation (TTA) perturbation matrix for stability evidence.',
          },
          {
            timestamp: '00:00:23',
            stage: 'GEOSPATIAL',
            message: 'Preserving EPSG:32644 CRS & Affine Transform. Writing Cloud-Optimized GeoTIFF raster.',
          },
        ],
      }));
    }, 9000);

    const timer4 = setTimeout(() => {
      setJob((prev) => ({
        ...prev,
        status: 'READY',
        currentStep: 'READY',
        completedAt: new Date().toISOString().replace('T', ' ').substring(0, 19) + ' UTC',
        logs: [
          ...prev.logs,
          {
            timestamp: '00:00:26',
            stage: 'READY',
            message: 'Reconstruction pipeline completed successfully. 2.5m product, stability layer, and STAC record indexed.',
          },
        ],
      }));
    }, 12500);

    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
      clearTimeout(timer4);
    };
  }, [jobConfig]);

  const getStepState = (step: typeof STEPS[number]) => {
    const currentIndex = STEPS.indexOf(job.currentStep);
    const stepIndex = STEPS.indexOf(step);

    if (stepIndex < currentIndex || job.status === 'READY') {
      return 'completed';
    }
    if (stepIndex === currentIndex) {
      return 'current';
    }
    return 'pending';
  };

  return (
    <div className="w-full min-h-[calc(100vh-3.5rem)] bg-[#0B0E12] text-white flex flex-col justify-center items-center p-6">
      <div className="max-w-3xl w-full bg-[#151A20] border border-[#1D242C] rounded-xl overflow-hidden shadow-2xl">
        {/* Job Header */}
        <div className="p-6 border-b border-[#1D242C] bg-[#151A20] flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs font-mono text-[#D8B84C] font-bold">{job.id}</span>
              <span className="text-[#343C45]">·</span>
              <span className="text-xs font-mono text-[#9AA3AD]">{job.tileId}</span>
            </div>
            <h2 className="text-lg font-bold text-white tracking-tight">
              {job.status === 'READY' ? 'Reconstruction Complete' : 'Super-Resolution Job in Progress'}
            </h2>
          </div>

          <div className="text-right">
            <span
              className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-mono font-medium ${
                job.status === 'READY'
                  ? 'bg-[#35C98A]/10 text-[#35C98A] border border-[#35C98A]/30'
                  : 'bg-[#D8B84C]/10 text-[#D8B84C] border border-[#D8B84C]/30'
              }`}
            >
              {job.status === 'READY' ? <CheckCircle2 className="w-3.5 h-3.5" /> : <Clock className="w-3.5 h-3.5 animate-spin" />}
              <span>{job.status}</span>
            </span>
          </div>
        </div>

        {/* Visual Workflow Stepper */}
        {/* Requirement: Completed = Blue/check, Current = Yellow, Pending = Silver */}
        <div className="p-6 border-b border-[#1D242C] bg-[#151A20]">
          <div className="flex items-center justify-between relative">
            <div className="absolute top-1/2 left-0 right-0 h-0.5 bg-[#1D242C] -translate-y-1/2 z-0" />
            {STEPS.map((step, idx) => {
              const state = getStepState(step);
              return (
                <div key={step} className="relative z-10 flex flex-col items-center">
                  <div
                    className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-mono transition-colors ${
                      state === 'completed'
                        ? 'bg-[#D8B84C] text-[#0B0E12] border-2 border-[#D8B84C]'
                        : state === 'current'
                        ? 'bg-[#D8B84C] text-[#0B0E12] font-bold ring-4 ring-[#D8B84C]/20'
                        : 'bg-[#151A20] text-[#9AA3AD] border border-[#343C45]'
                    }`}
                  >
                    {state === 'completed' ? (
                      <CheckCircle2 className="w-4 h-4 text-white" />
                    ) : (
                      <span>{idx + 1}</span>
                    )}
                  </div>
                  <span
                    className={`text-[10px] font-mono mt-2 tracking-wider uppercase ${
                      state === 'current'
                        ? 'text-[#D8B84C] font-bold'
                        : state === 'completed'
                        ? 'text-white'
                        : 'text-[#9AA3AD]'
                    }`}
                  >
                    {step}
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Terminal Execution Log Stream */}
        <div className="p-5 bg-[#0e0e0e] border-b border-[#1D242C]">
          <div className="flex items-center justify-between mb-3 text-xs text-[#9AA3AD]">
            <div className="flex items-center gap-1.5 font-mono">
              <Terminal className="w-3.5 h-3.5 text-[#D8B84C]" />
              <span>EXECUTION TELEMETRY &amp; HARMONISATION LOGS</span>
            </div>
            <span className="font-mono text-[11px] text-[#9AA3AD]">Model: {jobConfig.model}</span>
          </div>

          <div className="font-mono text-xs space-y-1.5 h-44 overflow-y-auto pr-2 select-text">
            {job.logs.map((log, i) => (
              <div key={i} className="flex items-start gap-3">
                <span className="text-[#666666] shrink-0 text-[11px]">{log.timestamp}</span>
                <span className="text-[#D8B84C] shrink-0 text-[11px]">[{log.stage}]</span>
                <span className="text-[#9AA3AD] leading-relaxed text-[11px]">{log.message}</span>
              </div>
            ))}
            {job.status !== 'READY' && (
              <div className="flex items-center gap-2 text-[#9AA3AD] text-[11px]">
                <span className="inline-block w-1.5 h-3 bg-[#D8B84C] animate-pulse" />
                <span>Computing layer transformation matrices...</span>
              </div>
            )}
          </div>
        </div>

        {/* Action Bottom Bar */}
        <div className="p-4 bg-[#151A20] flex items-center justify-between">
          <div className="text-xs text-[#9AA3AD]">
            <span>Target Grid: </span>
            <span className="font-mono text-white">2.5 m</span>
            <span className="mx-2">·</span>
            <span>Bands: </span>
            <span className="font-mono text-white">12 L2A (B10 excluded)</span>
          </div>

          <div className="flex items-center gap-3">
            {job.status !== 'READY' ? (
              <button
                onClick={onCancel}
                className="px-4 py-2 text-xs font-medium text-[#9AA3AD] hover:text-white bg-[#1D242C] hover:bg-[#343C45] rounded transition-colors cursor-pointer"
              >
                Cancel Job
              </button>
            ) : (
              <button
                onClick={() =>
                  onComplete({
                    id: 'proj-active-reconstruction',
                    name: jobConfig.aoiName,
                    sceneId: jobConfig.sceneId,
                    tileId: jobConfig.tileId,
                  })
                }
                className="flex items-center gap-2 px-5 py-2.5 text-xs font-bold rounded bg-[#D8B84C] hover:bg-[#E4C96A] text-[#151A20] transition-all cursor-pointer shadow-lg animate-pulse"
              >
                <span>Inspect Reconstruction Workspace</span>
                <ArrowRight className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
