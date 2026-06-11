import { coordinateVelocity, traceNullGeodesic, type GeodesicState, type TraceOptions } from './geodesic';
import { kerrSchildRadius, type KerrSchildParams, type Vec3, type Vec4 } from './kerrSchild';
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
  status: 'escaped' | 'horizon' | 'singularity' | 'max-steps';
  steps: number;
  finalRadius: number;
  maxHamiltonianDrift: number;
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
      const result = traceNullGeodesic(initial, params, traceOptions);
      const finalRadius = kerrSchildRadius(position3(result.state.position), params);
      rays.push({
        pixelX: x,
        pixelY: y,
        localDirection,
        status: result.status,
        steps: result.steps,
        finalRadius,
        maxHamiltonianDrift: result.maxHamiltonianDrift,
        color: diagnosticColor(result.status, coordinateVelocity(result.state.position, result.state.momentum, params)),
      });
    }
  }

  return { width, height, rays };
}

function diagnosticColor(status: ProbeRay['status'], velocity: Vec4): [number, number, number] {
  if (status === 'horizon') return [0, 0, 0];
  if (status === 'singularity') return [0.3, 0, 0.5];
  if (status === 'max-steps') return [0.9, 0.1, 0.1];
  const dir = normalize3({ x: velocity.x, y: velocity.y, z: velocity.z });
  return [0.35 + 0.35 * dir.x, 0.35 + 0.35 * dir.y, 0.55 + 0.3 * dir.z];
}

function position3(v: Vec4): Vec3 {
  return { x: v.x, y: v.y, z: v.z };
}

function normalize3(v: Vec3): Vec3 {
  const length = Math.hypot(v.x, v.y, v.z) || 1;
  return { x: v.x / length, y: v.y / length, z: v.z / length };
}
