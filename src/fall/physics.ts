import {
  coordinateVelocity,
  hamiltonian,
  stepNullGeodesic,
  type GeodesicState,
} from '../gr/geodesic';
import {
  horizonRadius,
  kerrSchildParams,
  kerrSchildNullSpatial,
  kerrSchildRadius,
  lowerVector,
  metricDot,
  type KerrSchildParams,
  type Vec3,
  type Vec4,
} from '../gr/kerrSchild';
import { buildObserverTetrad, staticObserverFourVelocity } from '../gr/tetrad';

export interface FallState {
  r: number;
  phi: number;
  tau: number;
  t: number;
  energy: number;
  angularMomentum: number;
  radialSign: number;
  position: Vec4;
  momentum: Vec4;
}

export interface LocalLaunch {
  r: number;
  phi: number;
  betaRadial: number;
  betaTangential: number;
}

export interface PreviewPoint {
  x: number;
  z: number;
  r: number;
}

export interface LocalVelocity {
  betaRadial: number;
  betaTangential: number;
  x: number;
  z: number;
  speed: number;
}

export const FALL_PARAMS: KerrSchildParams = kerrSchildParams(0.35, 0.5);
export const HORIZON_RADIUS = horizonRadius(FALL_PARAMS);
export const SINGULARITY_CUTOFF = 0.045;

export function clampRadius(r: number): number {
  return Math.max(HORIZON_RADIUS + 0.08, Math.min(18, r));
}

export function positionFromState(state: FallState): PreviewPoint {
  return {
    x: state.position.x,
    z: state.position.y,
    r: state.r,
  };
}

export function spatialPositionFromState(state: FallState): Vec3 {
  return {
    x: state.position.x,
    y: state.position.y,
    z: state.position.z,
  };
}

export function fourVelocityFromState(state: FallState): Vec4 {
  return coordinateVelocity(state.position, state.momentum, FALL_PARAMS);
}

export function launchFromLocal(local: LocalLaunch): FallState {
  const r = clampRadius(local.r);
  const phi = local.phi;
  const position = { t: 0, x: r * Math.cos(phi), y: r * Math.sin(phi), z: 0 };
  const betaRadial = clamp(local.betaRadial, -0.88, 0.88);
  const betaTangential = clamp(local.betaTangential, -0.88, 0.88);
  const beta2 = Math.min(betaRadial * betaRadial + betaTangential * betaTangential, 0.88 * 0.88);
  const betaScale = beta2 > 0 ? Math.sqrt(beta2) / Math.hypot(betaRadial, betaTangential) : 1;
  const radial = normalize3({ x: Math.cos(phi), y: Math.sin(phi), z: 0 });
  const tangent = { x: -Math.sin(phi), y: Math.cos(phi), z: 0 };
  const referenceVelocity = referenceObserverFourVelocity(position);
  const referenceFrame = buildObserverTetrad(position, FALL_PARAMS, referenceVelocity, {
    forward: { t: 0, ...radial },
    right: { t: 0, ...tangent },
    up: { t: 0, x: 0, y: 0, z: 1 },
  });
  const gamma = 1 / Math.sqrt(1 - beta2);
  const fourVelocity = scaleVec4(
    addVec4(
      referenceFrame.eTime,
      addVec4(
        scaleVec4(referenceFrame.eForward, betaRadial * betaScale),
        scaleVec4(referenceFrame.eRight, betaTangential * betaScale),
      ),
    ),
    gamma,
  );
  const momentum = lowerVector(position, FALL_PARAMS, fourVelocity);

  return stateFromGeodesic({ position, momentum }, 0, Math.abs(betaRadial) < 1e-6 ? -1 : Math.sign(betaRadial));
}

export function stepFall(state: FallState, dTau: number): FallState {
  let next = { ...state, position: { ...state.position }, momentum: { ...state.momentum } };
  const steps = Math.max(1, Math.ceil(Math.abs(dTau) / 0.004));
  const h = dTau / steps;
  for (let i = 0; i < steps; i++) {
    if (kerrSchildRadius(spatialPosition(next.position), FALL_PARAMS) <= SINGULARITY_CUTOFF) {
      return clampToSingularity(next);
    }
    const geodesic = stepNullGeodesic(next, FALL_PARAMS, h);
    if (kerrSchildRadius(spatialPosition(geodesic.position), FALL_PARAMS) <= SINGULARITY_CUTOFF) {
      return clampToSingularity(next);
    }
    next = stateFromGeodesic(geodesic, next.tau + h, radialSignFromState(geodesic));
  }
  return next;
}

export function previewFall(state: FallState, maxTau: number): PreviewPoint[] {
  const pts: PreviewPoint[] = [positionFromState(state)];
  let s = { ...state, position: { ...state.position }, momentum: { ...state.momentum } };
  const step = 0.035;
  const samples = Math.ceil(maxTau / step);
  for (let i = 0; i < samples; i++) {
    s = stepFall(s, step);
    if (i % 2 === 0) pts.push(positionFromState(s));
    if (isSingularityReached(s) || s.r > 18.5) break;
  }
  pts.push(positionFromState(s));
  return pts;
}

export function localVelocity(state: FallState): LocalVelocity {
  const position = spatialPosition(state.position);
  const u = fourVelocityFromState(state);
  let speed = 0.995;
  if (state.r > HORIZON_RADIUS + 0.04) {
    const referenceVelocity = referenceObserverFourVelocity(position);
    const gamma = Math.max(-metricDot(position, FALL_PARAMS, u, referenceVelocity), 1);
    speed = Math.sqrt(Math.max(0, 1 - 1 / (gamma * gamma)));
  }

  const v = coordinateVelocity(state.position, state.momentum, FALL_PARAMS);
  const spatial = normalize3({ x: v.x, y: v.y, z: 0 });
  const radial = normalize3({ x: state.position.x, y: state.position.y, z: 0 });
  const tangent = { x: -radial.y, y: radial.x, z: 0 };
  const betaRadial = speed * dot3(spatial, radial);
  const betaTangential = speed * dot3(spatial, tangent);
  return {
    betaRadial,
    betaTangential,
    x: betaRadial * radial.x + betaTangential * tangent.x,
    z: betaRadial * radial.y + betaTangential * tangent.y,
    speed,
  };
}

export function isHorizonCrossed(state: FallState): boolean {
  return state.r <= HORIZON_RADIUS + 0.01;
}

export function isSingularityReached(state: FallState): boolean {
  return state.r <= SINGULARITY_CUTOFF + 0.002;
}

export function timelikeResidual(state: FallState): number {
  return Math.abs(hamiltonian(state, FALL_PARAMS) + 0.5);
}

function stateFromGeodesic(geodesic: GeodesicState, tau: number, radialSign: number): FallState {
  const position = geodesic.position;
  const momentum = geodesic.momentum;
  const r = kerrSchildRadius(spatialPosition(position), FALL_PARAMS);
  const phi = Math.atan2(position.y, position.x);
  return {
    r,
    phi,
    tau,
    t: position.t,
    energy: -momentum.t,
    angularMomentum: position.x * momentum.y - position.y * momentum.x,
    radialSign,
    position,
    momentum,
  };
}

function radialSignFromState(state: GeodesicState): number {
  const v = coordinateVelocity(state.position, state.momentum, FALL_PARAMS);
  const p = spatialPosition(state.position);
  const radial = dot3(p, { x: v.x, y: v.y, z: v.z });
  return radial >= 0 ? 1 : -1;
}

function clampToSingularity(state: FallState): FallState {
  const p = spatialPosition(state.position);
  const len = Math.hypot(p.x, p.y, p.z) || 1;
  return stateFromGeodesic(
    {
      position: {
        ...state.position,
        x: (p.x / len) * SINGULARITY_CUTOFF,
        y: (p.y / len) * SINGULARITY_CUTOFF,
        z: (p.z / len) * SINGULARITY_CUTOFF,
      },
      momentum: state.momentum,
    },
    state.tau,
    state.radialSign,
  );
}

function spatialPosition(position: Vec4): Vec3 {
  return { x: position.x, y: position.y, z: position.z };
}

function referenceObserverFourVelocity(position: Vec3): Vec4 {
  try {
    return staticObserverFourVelocity(position, FALL_PARAMS);
  } catch {
    return infallingKerrSchildObserver(position);
  }
}

function infallingKerrSchildObserver(position: Vec3): Vec4 {
  const l = kerrSchildNullSpatial(position, FALL_PARAMS);
  for (const epsilon of [0.25, 0.1, 0.04, 0.015, 0.006, 0.002]) {
    const inward = 1 - epsilon;
    const candidate = { t: 1, x: -inward * l.x, y: -inward * l.y, z: -inward * l.z };
    const norm = metricDot(position, FALL_PARAMS, candidate, candidate);
    if (norm < -1e-10) {
      return scaleVec4(candidate, 1 / Math.sqrt(-norm));
    }
  }
  throw new Error('Could not construct a timelike Kerr-Schild reference observer');
}

function normalize3(v: Vec3): Vec3 {
  const len = Math.hypot(v.x, v.y, v.z) || 1;
  return { x: v.x / len, y: v.y / len, z: v.z / len };
}

function dot3(a: Vec3, b: Vec3): number {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

function addVec4(a: Vec4, b: Vec4): Vec4 {
  return { t: a.t + b.t, x: a.x + b.x, y: a.y + b.y, z: a.z + b.z };
}

function scaleVec4(v: Vec4, scale: number): Vec4 {
  return { t: v.t * scale, x: v.x * scale, y: v.y * scale, z: v.z * scale };
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}
