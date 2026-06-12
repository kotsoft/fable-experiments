import { type ThinDisk } from './disk';
import {
  coordinateVelocity,
  hamiltonian,
  stepNullGeodesic,
  traceNullGeodesic,
  type GeodesicState,
  type TraceOptions,
} from './geodesic';
import { horizonRadius, kerrSchildRadius, type KerrSchildParams, type Vec3, type Vec4 } from './kerrSchild';
import {
  sampleDiskVolumeEmission,
  type DiskRadianceModel,
  type DiskRadianceSample,
} from './radiance';
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

const VOLUME_SAMPLE_STRIDE = 2;

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
  const accumulatedColor: [number, number, number] = [0, 0, 0];
  let accumulatedWeight = 0;
  let brightestDiskRadius = -1;
  let brightestDiskRadiance: DiskRadianceSample | null = null;
  const volumeModel = radianceModel
    ? { ...radianceModel, innerRadius: disk.innerRadius, outerRadius: disk.outerRadius }
    : undefined;

  for (let steps = 0; steps < options.maxSteps; steps++) {
    const radius = kerrSchildRadius(position3(state.position), params);
    if (radius <= options.singularityRadius) {
      return finishAccumulatedTrace(
        state,
        steps,
        'singularity',
        maxHamiltonianDrift,
        accumulatedColor,
        accumulatedWeight,
        brightestDiskRadius,
        brightestDiskRadiance,
        params,
      );
    }
    if (horizon > 0 && radius <= horizon) {
      return finishAccumulatedTrace(
        state,
        steps,
        'horizon',
        maxHamiltonianDrift,
        accumulatedColor,
        accumulatedWeight,
        brightestDiskRadius,
        brightestDiskRadiance,
        params,
      );
    }
    if (radius >= options.escapeRadius && radialCoordinateSpeed(state, params) > 0) {
      return finishAccumulatedTrace(
        state,
        steps,
        'escaped',
        maxHamiltonianDrift,
        accumulatedColor,
        accumulatedWeight,
        brightestDiskRadius,
        brightestDiskRadiance,
        params,
      );
    }

    if (volumeModel && steps % VOLUME_SAMPLE_STRIDE === 0) {
      const emission = sampleDiskVolumeEmission(
        state,
        params,
        volumeModel,
        observerVelocity,
        options.stepSize * VOLUME_SAMPLE_STRIDE,
      );
      if (emission) {
        accumulatedColor[0] += emission.rgb[0];
        accumulatedColor[1] += emission.rgb[1];
        accumulatedColor[2] += emission.rgb[2];
        const weight = emission.rgb[0] + emission.rgb[1] + emission.rgb[2];
        if (weight > accumulatedWeight) {
          accumulatedWeight = weight;
          brightestDiskRadius = emission.radius;
          brightestDiskRadiance = emission.radiance;
        }
      }
    }

    state = stepNullGeodesic(state, params, options.stepSize);
    maxHamiltonianDrift = Math.max(maxHamiltonianDrift, Math.abs(hamiltonian(state, params) - h0));
  }

  return finishAccumulatedTrace(
    state,
    options.maxSteps,
    'max-steps',
    maxHamiltonianDrift,
    accumulatedColor,
    accumulatedWeight,
    brightestDiskRadius,
    brightestDiskRadiance,
    params,
  );
}

function finishAccumulatedTrace(
  state: GeodesicState,
  steps: number,
  terminalStatus: ProbeRay['status'],
  maxHamiltonianDrift: number,
  accumulatedColor: [number, number, number],
  accumulatedWeight: number,
  diskRadius: number,
  diskRadiance: DiskRadianceSample | null,
  params: KerrSchildParams,
): ProbeTraceResult {
  if (accumulatedWeight <= 1e-8) {
    return { state, steps, status: terminalStatus, maxHamiltonianDrift };
  }

  const background = diagnosticColor(
    terminalStatus,
    coordinateVelocity(state.position, state.momentum, params),
  );

  return {
    state,
    steps,
    status: 'disk',
    maxHamiltonianDrift,
    diskHit: {
      radius: diskRadius,
      affineParameter: steps,
      radiance: diskRadiance
        ? { ...diskRadiance, observedRgb: addRgb(accumulatedColor, background) }
        : null,
    },
  };
}

function diagnosticColor(status: ProbeRay['status'], velocity: Vec4): [number, number, number] {
  if (status === 'disk') return [1, 0.72, 0.35];
  if (status === 'horizon') return [0, 0, 0];
  if (status === 'singularity') return [0.3, 0, 0.5];
  if (status === 'max-steps') return [0.9, 0.1, 0.1];
  const dir = normalize3({ x: velocity.x, y: velocity.y, z: velocity.z });
  return escapedBackgroundColor(dir);
}

export function escapedBackgroundColor(direction: Vec3): [number, number, number] {
  const dir = normalize3(direction);
  const bandCoordinate = Math.abs(0.72 * dir.y + 0.24 * dir.x - 0.1 * dir.z);
  const band = Math.max(0, 1 - bandCoordinate * 4.5) ** 2;
  const base: [number, number, number] = [
    0.006 + 0.012 * Math.max(dir.z, 0),
    0.008 + 0.01 * Math.max(dir.y, 0),
    0.016 + 0.018 * Math.max(-dir.z, 0),
  ];
  const color: [number, number, number] = [
    base[0] + 0.035 * band,
    base[1] + 0.032 * band,
    base[2] + 0.048 * band,
  ];
  addStar(color, dir, normalize3({ x: 0.42, y: 0.16, z: 0.89 }), 220, [1.9, 1.65, 1.2]);
  addStar(color, dir, normalize3({ x: -0.68, y: -0.08, z: 0.73 }), 180, [1.15, 1.35, 1.9]);
  addStar(color, dir, normalize3({ x: 0.09, y: 0.82, z: -0.56 }), 260, [1.7, 1.45, 1.05]);
  addStar(color, dir, normalize3({ x: -0.24, y: -0.74, z: -0.63 }), 200, [1.25, 1.55, 1.8]);
  return color;
}

function addStar(
  color: [number, number, number],
  direction: Vec3,
  starDirection: Vec3,
  sharpness: number,
  tint: [number, number, number],
): void {
  const strength = Math.max(0, dot3(direction, starDirection)) ** sharpness;
  color[0] += tint[0] * strength;
  color[1] += tint[1] * strength;
  color[2] += tint[2] * strength;
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

function dot3(a: Vec3, b: Vec3): number {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

function addRgb(a: [number, number, number], b: [number, number, number]): [number, number, number] {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}
