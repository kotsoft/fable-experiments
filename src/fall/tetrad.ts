import { metricDot, type Vec3, type Vec4 } from '../gr/kerrSchild';
import { buildObserverTetrad, tetradResidual as grTetradResidual, type GrTetrad } from '../gr/tetrad';
import {
  FALL_PARAMS,
  fourVelocityFromState,
  localVelocity,
  spatialPositionFromState,
  type FallState,
} from './physics';

export type { Vec3, Vec4 };

export type Tetrad = GrTetrad;

export interface PlayerCamera {
  yaw: number;
  pitch: number;
  roll: number;
}

export interface ObserverFrame {
  tetrad: Tetrad;
  beta: Vec3;
  speed: number;
  flatForward: { x: number; z: number };
  fourVelocity: Vec4;
}

const MAX_PITCH = 1.35;

export function createPlayerCamera(x = -1, z = 0): PlayerCamera {
  const flat = normalize2(x, z);
  return { yaw: Math.atan2(flat.z, flat.x), pitch: 0, roll: 0 };
}

export function setCameraLookDirection(camera: PlayerCamera, x: number, z: number): void {
  const flat = normalize2(x, z);
  camera.yaw = Math.atan2(flat.z, flat.x);
  camera.pitch = 0;
  camera.roll = 0;
}

export function rotatePlayerCamera(camera: PlayerCamera, deltaYaw: number, deltaPitch: number): void {
  camera.yaw += deltaYaw;
  camera.pitch = clamp(camera.pitch + deltaPitch, -MAX_PITCH, MAX_PITCH);
}

export function cameraFlatForward(camera: PlayerCamera): { x: number; z: number } {
  return { x: Math.cos(camera.yaw), z: Math.sin(camera.yaw) };
}

export function observerFrameFromState(state: FallState, camera: PlayerCamera): ObserverFrame {
  const position = spatialPositionFromState(state);
  const fourVelocity = fourVelocityFromState(state);
  const local = localVelocity(state);
  const hints = cameraAxisHints(camera);
  const tetrad = buildObserverTetrad(position, FALL_PARAMS, fourVelocity, hints);
  return {
    tetrad,
    beta: { x: local.x, y: local.z, z: 0 },
    speed: local.speed,
    flatForward: cameraFlatForward(camera),
    fourVelocity,
  };
}

export function tetradResidual(tetrad: Tetrad, state?: FallState): number {
  if (!state) {
    return flatTetradResidual(tetrad);
  }
  return grTetradResidual(spatialPositionFromState(state), FALL_PARAMS, tetrad);
}

function cameraAxisHints(camera: PlayerCamera): { right: Vec4; up: Vec4; forward: Vec4 } {
  const cp = Math.cos(camera.pitch);
  const sp = Math.sin(camera.pitch);
  const flat = cameraFlatForward(camera);
  const forward = normalize3({ x: flat.x * cp, y: flat.z * cp, z: sp });
  const right = normalize3({ x: flat.z, y: -flat.x, z: 0 });
  const up = normalize3(cross(right, forward));
  return {
    forward: { t: 0, ...forward },
    right: { t: 0, ...right },
    up: { t: 0, ...up },
  };
}

function flatTetradResidual(tetrad: Tetrad): number {
  const position = { x: 20, y: 0, z: 0 };
  const axes = [tetrad.eTime, tetrad.eRight, tetrad.eUp, tetrad.eForward];
  const expected = [-1, 1, 1, 1];
  let residual = 0;
  for (let i = 0; i < axes.length; i++) {
    residual = Math.max(residual, Math.abs(metricDot(position, { mass: 0, spin: 0 }, axes[i], axes[i]) - expected[i]));
    for (let j = i + 1; j < axes.length; j++) {
      residual = Math.max(residual, Math.abs(metricDot(position, { mass: 0, spin: 0 }, axes[i], axes[j])));
    }
  }
  return residual;
}

function normalize2(x: number, z: number): { x: number; z: number } {
  const len = Math.hypot(x, z) || 1;
  return { x: x / len, z: z / len };
}

function normalize3(v: Vec3): Vec3 {
  const len = Math.hypot(v.x, v.y, v.z) || 1;
  return { x: v.x / len, y: v.y / len, z: v.z / len };
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  };
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}
