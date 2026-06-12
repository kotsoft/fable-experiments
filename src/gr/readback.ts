import { type ProbeGrid, type ProbeRay } from './referenceProbe';

export const READBACK_FLOATS_PER_RAY = 16;

export const ReadbackStatus = {
  Escaped: 0,
  Horizon: 1,
  Singularity: 2,
  MaxSteps: 3,
  Disk: 4,
} as const;

export type ReadbackStatus = typeof ReadbackStatus[keyof typeof ReadbackStatus];

export interface RayReadback {
  status: ReadbackStatus;
  steps: number;
  finalRadius: number;
  maxHamiltonianDrift: number;
  diskRadius: number;
  redshift: number;
  bolometricIntensity: number;
  color: [number, number, number];
}

export function probeGridToReadback(grid: ProbeGrid): Float32Array {
  const out = new Float32Array(grid.rays.length * READBACK_FLOATS_PER_RAY);
  grid.rays.forEach((ray, index) => {
    const base = index * READBACK_FLOATS_PER_RAY;
    const readback = probeRayToReadback(ray);
    out[base] = readback.status;
    out[base + 1] = readback.steps;
    out[base + 2] = readback.finalRadius;
    out[base + 3] = readback.maxHamiltonianDrift;
    out[base + 4] = readback.diskRadius;
    out[base + 5] = readback.redshift;
    out[base + 6] = readback.bolometricIntensity;
    out[base + 7] = readback.color[0];
    out[base + 8] = readback.color[1];
    out[base + 9] = readback.color[2];
    out[base + 10] = ray.localDirection.x;
    out[base + 11] = ray.localDirection.y;
    out[base + 12] = ray.localDirection.z;
    out[base + 13] = ray.pixelX;
    out[base + 14] = ray.pixelY;
    out[base + 15] = 0;
  });
  return out;
}

export function probeRayToReadback(ray: ProbeRay): RayReadback {
  return {
    status: statusToReadback(ray.status),
    steps: ray.steps,
    finalRadius: ray.finalRadius,
    maxHamiltonianDrift: ray.maxHamiltonianDrift,
    diskRadius: ray.diskHit?.radius ?? -1,
    redshift: ray.diskHit?.radiance?.redshift ?? 0,
    bolometricIntensity: ray.diskHit?.radiance?.bolometricIntensity ?? 0,
    color: ray.color,
  };
}

export function statusToReadback(status: ProbeRay['status']): ReadbackStatus {
  switch (status) {
    case 'escaped':
      return ReadbackStatus.Escaped;
    case 'horizon':
      return ReadbackStatus.Horizon;
    case 'singularity':
      return ReadbackStatus.Singularity;
    case 'max-steps':
      return ReadbackStatus.MaxSteps;
    case 'disk':
      return ReadbackStatus.Disk;
  }
}
