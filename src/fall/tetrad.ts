import { localVelocity, type FallState } from './physics';

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface Vec4 {
  t: number;
  x: number;
  y: number;
  z: number;
}

export interface Tetrad {
  eTime: Vec4;
  eRight: Vec4;
  eUp: Vec4;
  eForward: Vec4;
}

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
  const velocity = localVelocity(state);
  const beta = { x: velocity.x, y: 0, z: velocity.z };
  const localBasis = cameraLocalBasis(camera);
  return {
    tetrad: boostTetrad(beta, localBasis),
    beta,
    speed: velocity.speed,
    flatForward: cameraFlatForward(camera),
  };
}

export function minkowskiDot(a: Vec4, b: Vec4): number {
  return -a.t * b.t + a.x * b.x + a.y * b.y + a.z * b.z;
}

export function tetradResidual(tetrad: Tetrad): number {
  const axes = [tetrad.eTime, tetrad.eRight, tetrad.eUp, tetrad.eForward];
  const expected = [-1, 1, 1, 1];
  let residual = 0;
  for (let i = 0; i < axes.length; i++) {
    residual = Math.max(residual, Math.abs(minkowskiDot(axes[i], axes[i]) - expected[i]));
    for (let j = i + 1; j < axes.length; j++) {
      residual = Math.max(residual, Math.abs(minkowskiDot(axes[i], axes[j])));
    }
  }
  return residual;
}

function cameraLocalBasis(camera: PlayerCamera): { right: Vec3; up: Vec3; forward: Vec3 } {
  const cp = Math.cos(camera.pitch);
  const sp = Math.sin(camera.pitch);
  const flat = cameraFlatForward(camera);
  const forward = normalize3({ x: flat.x * cp, y: sp, z: flat.z * cp });
  const right = normalize3({ x: flat.z, y: 0, z: -flat.x });
  const up = normalize3(cross(right, forward));
  return { right, up, forward };
}

function boostTetrad(beta: Vec3, basis: { right: Vec3; up: Vec3; forward: Vec3 }): Tetrad {
  const b2 = dot3(beta, beta);
  const gamma = 1 / Math.sqrt(Math.max(1 - b2, 1e-5));
  return {
    eTime: { t: gamma, x: gamma * beta.x, y: gamma * beta.y, z: gamma * beta.z },
    eRight: boostSpatialAxis(basis.right, beta, gamma, b2),
    eUp: boostSpatialAxis(basis.up, beta, gamma, b2),
    eForward: boostSpatialAxis(basis.forward, beta, gamma, b2),
  };
}

function boostSpatialAxis(axis: Vec3, beta: Vec3, gamma: number, b2: number): Vec4 {
  if (b2 < 1e-10) return { t: 0, x: axis.x, y: axis.y, z: axis.z };
  const bd = dot3(beta, axis);
  const scale = ((gamma - 1) * bd) / b2;
  return {
    t: gamma * bd,
    x: axis.x + scale * beta.x,
    y: axis.y + scale * beta.y,
    z: axis.z + scale * beta.z,
  };
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

function dot3(a: Vec3, b: Vec3): number {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}
