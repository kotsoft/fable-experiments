import { refineDiskCrossing, type ThinDisk } from './disk';
import {
  coordinateVelocity,
  hamiltonian,
  stepNullGeodesic,
  traceNullGeodesic,
  type GeodesicState,
  type TraceOptions,
} from './geodesic';
import { horizonRadius, kerrSchildRadius, type KerrSchildParams, type Vec3, type Vec4 } from './kerrSchild';
import { sampleDiskRadiance, type DiskRadianceModel, type DiskRadianceSample } from './radiance';
import { launchPhotonFromTetrad, type GrTetrad } from './tetrad';

export interface ProbeCamera {
  position: Vec3;
  tetrad: GrTetrad;
  verticalFovRadians: number;
}

export interface ProbeRay {
  pixelX: number;
  pixelY: number;
  localDirection: Vec3;
  status: 'escaped' | 'horizon' | 'singularity' | 'max-steps' | 'disk';
  steps: number;
  finalRadius: number;
  maxHamiltonianDrift: number;
  diskHit?: {
    radius: number;
    affineParameter: number;
    radiance: DiskRadianceSample | null;
  };
  color: [number, number, number];
}

export interface ProbeGrid {
  width: number;
  height: number;
  rays: ProbeRay[];
}

export function renderProbeGrid(
  params: KerrSchildParams,
  camera: ProbeCamera,
  width: number,
  height: number,
  traceOptions: TraceOptions,
  disk?: ThinDisk,
  radianceModel?: DiskRadianceModel,
): ProbeGrid {
  const rays: ProbeRay[] = [];
  const aspect = width / height;
  const tanHalfFov = Math.tan(camera.verticalFovRadians * 0.5);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const ndcX = (2 * (x + 0.5) / width - 1) * aspect;
      const ndcY = 1 - 2 * (y + 0.5) / height;
      const localDirection = normalize3({ x: ndcX * tanHalfFov, y: ndcY * tanHalfFov, z: 1 });
      const initial: GeodesicState = {
        position: { t: 0, ...camera.position },
        momentum: launchPhotonFromTetrad(camera.position, params, camera.tetrad, localDirection),
      };
      const result = disk
        ? traceProbeRayWithDisk(initial, params, traceOptions, disk, camera.tetrad.eTime, radianceModel)
        : traceProbeRay(initial, params, traceOptions);
      const finalRadius = kerrSchildRadius(position3(result.state.position), params);
      rays.push({
        pixelX: x,
        pixelY: y,
        localDirection,
        status: result.status,
        steps: result.steps,
        finalRadius,
        maxHamiltonianDrift: result.maxHamiltonianDrift,
        diskHit: result.diskHit,
        color: result.diskHit?.radiance?.observedRgb ??
          diagnosticColor(result.status, coordinateVelocity(result.state.position, result.state.momentum, params)),
      });
    }
  }

  return { width, height, rays };
}

interface ProbeTraceResult {
  state: GeodesicState;
  steps: number;
  status: ProbeRay['status'];
  maxHamiltonianDrift: number;
  diskHit?: ProbeRay['diskHit'];
}

function traceProbeRay(initial: GeodesicState, params: KerrSchildParams, options: TraceOptions): ProbeTraceResult {
  return traceNullGeodesic(initial, params, options);
}

function traceProbeRayWithDisk(
  initial: GeodesicState,
  params: KerrSchildParams,
  options: TraceOptions,
  disk: ThinDisk,
  observerVelocity: Vec4,
  radianceModel?: DiskRadianceModel,
): ProbeTraceResult {
  let state = initial;
  const h0 = hamiltonian(initial, params);
  let maxHamiltonianDrift = 0;
  const horizon = horizonRadius(params);

  for (let steps = 0; steps < options.maxSteps; steps++) {
    const radius = kerrSchildRadius(position3(state.position), params);
    if (radius <= options.singularityRadius) {
      return { state, steps, status: 'singularity', maxHamiltonianDrift };
    }
    if (horizon > 0 && radius <= horizon) {
      return { state, steps, status: 'horizon', maxHamiltonianDrift };
    }
    if (radius >= options.escapeRadius && radialCoordinateSpeed(state, params) > 0) {
      return { state, steps, status: 'escaped', maxHamiltonianDrift };
    }

    const crossing = refineDiskCrossing(state, params, options.stepSize, disk);
    if (crossing && crossing.affineParameter > 1e-7) {
      return {
        state: crossing.state,
        steps,
        status: 'disk',
        maxHamiltonianDrift,
        diskHit: {
          radius: crossing.radius,
          affineParameter: crossing.affineParameter,
          radiance: radianceModel
            ? sampleDiskRadiance(crossing.state, params, radianceModel, observerVelocity)
            : null,
        },
      };
    }

    state = stepNullGeodesic(state, params, options.stepSize);
    maxHamiltonianDrift = Math.max(maxHamiltonianDrift, Math.abs(hamiltonian(state, params) - h0));
  }

  return { state, steps: options.maxSteps, status: 'max-steps', maxHamiltonianDrift };
}

function diagnosticColor(status: ProbeRay['status'], velocity: Vec4): [number, number, number] {
  if (status === 'disk') return [1, 0.72, 0.35];
  if (status === 'horizon') return [0, 0, 0];
  if (status === 'singularity') return [0.3, 0, 0.5];
  if (status === 'max-steps') return [0.9, 0.1, 0.1];
  const dir = normalize3({ x: velocity.x, y: velocity.y, z: velocity.z });
  return [0.35 + 0.35 * dir.x, 0.35 + 0.35 * dir.y, 0.55 + 0.3 * dir.z];
}

function radialCoordinateSpeed(state: GeodesicState, params: KerrSchildParams): number {
  const p = position3(state.position);
  const velocity = coordinateVelocity(state.position, state.momentum, params);
  const radius = Math.max(kerrSchildRadius(p, params), 1e-8);
  return (p.x * velocity.x + p.y * velocity.y + p.z * velocity.z) / radius;
}

function position3(v: Vec4): Vec3 {
  return { x: v.x, y: v.y, z: v.z };
}

function normalize3(v: Vec3): Vec3 {
  const length = Math.hypot(v.x, v.y, v.z) || 1;
  return { x: v.x / length, y: v.y / length, z: v.z / length };
}
