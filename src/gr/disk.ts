import { stepNullGeodesic, type GeodesicState } from './geodesic';
import { kerrSchildRadius, type KerrSchildParams, type Vec3 } from './kerrSchild';

export interface ThinDisk {
  innerRadius: number;
  outerRadius: number;
  normal?: Vec3;
}

export interface DiskCrossing {
  state: GeodesicState;
  affineParameter: number;
  radius: number;
  height: number;
}

export function diskHeight(state: GeodesicState, disk: ThinDisk): number {
  const n = normalize3(disk.normal ?? { x: 0, y: 0, z: 1 });
  return state.position.x * n.x + state.position.y * n.y + state.position.z * n.z;
}

export function refineDiskCrossing(
  start: GeodesicState,
  params: KerrSchildParams,
  stepSize: number,
  disk: ThinDisk,
  iterations = 36,
): DiskCrossing | null {
  const h0 = diskHeight(start, disk);
  const end = stepNullGeodesic(start, params, stepSize);
  const h1 = diskHeight(end, disk);

  if (Math.abs(h0) < 1e-12) return crossingAt(start, params, disk, 0);
  if (h0 * h1 > 0) return null;

  let lo = 0;
  let hi = stepSize;
  let loHeight = h0;
  let best = end;
  for (let i = 0; i < iterations; i++) {
    const mid = 0.5 * (lo + hi);
    const state = stepNullGeodesic(start, params, mid);
    const height = diskHeight(state, disk);
    best = state;
    if (Math.abs(height) < 1e-12) {
      lo = mid;
      hi = mid;
      break;
    }
    if (loHeight * height <= 0) {
      hi = mid;
    } else {
      lo = mid;
      loHeight = height;
    }
  }

  return crossingAt(best, params, disk, 0.5 * (lo + hi));
}

function crossingAt(
  state: GeodesicState,
  params: KerrSchildParams,
  disk: ThinDisk,
  affineParameter: number,
): DiskCrossing | null {
  const radius = kerrSchildRadius({ x: state.position.x, y: state.position.y, z: state.position.z }, params);
  if (radius < disk.innerRadius || radius > disk.outerRadius) return null;
  return {
    state,
    affineParameter,
    radius,
    height: diskHeight(state, disk),
  };
}

function normalize3(v: Vec3): Vec3 {
  const length = Math.hypot(v.x, v.y, v.z) || 1;
  return { x: v.x / length, y: v.y / length, z: v.z / length };
}
