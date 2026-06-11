export interface FallState {
  r: number;
  phi: number;
  tau: number;
  t: number;
  energy: number;
  angularMomentum: number;
  radialSign: number;
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

const HORIZON = 1.0;
const SINGULARITY_CUTOFF = 0.08;

export function clampRadius(r: number): number {
  return Math.max(1.08, Math.min(18, r));
}

export function positionFromState(state: FallState): PreviewPoint {
  return {
    x: state.r * Math.cos(state.phi),
    z: state.r * Math.sin(state.phi),
    r: state.r,
  };
}

export function launchFromLocal(local: LocalLaunch): FallState {
  const r = clampRadius(local.r);
  const f = schwarzschildF(r);
  const betaRadial = Math.max(-0.85, Math.min(0.85, local.betaRadial));
  const betaTangential = Math.max(-0.85, Math.min(0.85, local.betaTangential));
  const beta2 = Math.min(betaRadial * betaRadial + betaTangential * betaTangential, 0.85 * 0.85);
  const gamma = 1 / Math.sqrt(1 - beta2);
  return {
    r,
    phi: local.phi,
    tau: 0,
    t: 0,
    energy: gamma * Math.sqrt(f),
    angularMomentum: r * gamma * betaTangential,
    radialSign: Math.abs(betaRadial) < 1e-6 ? -1 : Math.sign(betaRadial),
  };
}

export function stepFall(state: FallState, dTau: number): FallState {
  let next = { ...state };
  const steps = Math.max(1, Math.ceil(Math.abs(dTau) / 0.006));
  const h = dTau / steps;
  for (let i = 0; i < steps; i++) {
    if (next.r <= SINGULARITY_CUTOFF) return { ...next, r: SINGULARITY_CUTOFF };
    next = rk4Step(next, h);
  }
  return next;
}

export function previewFall(state: FallState, maxTau: number): PreviewPoint[] {
  const pts: PreviewPoint[] = [positionFromState(state)];
  let s = { ...state };
  const step = 0.04;
  const samples = Math.ceil(maxTau / step);
  for (let i = 0; i < samples; i++) {
    s = stepFall(s, step);
    if (i % 2 === 0) pts.push(positionFromState(s));
    if (s.r <= SINGULARITY_CUTOFF || s.r > 18.5) break;
  }
  pts.push(positionFromState(s));
  return pts;
}

export function localVelocity(state: FallState): LocalVelocity {
  if (state.r < HORIZON) return interiorVelocity(state);
  const f = Math.max(schwarzschildF(state.r), 1e-5);
  const radialSq = radialPotential(state.r, state.energy, state.angularMomentum);
  const dr = state.radialSign * Math.sqrt(Math.max(radialSq, 0));
  const betaRadial = dr / Math.max(state.energy, 1e-5);
  const betaTangential = (state.angularMomentum / state.r) * Math.sqrt(f) / Math.max(state.energy, 1e-5);
  const erx = Math.cos(state.phi), erz = Math.sin(state.phi);
  const epx = -Math.sin(state.phi), epz = Math.cos(state.phi);
  const x = betaRadial * erx + betaTangential * epx;
  const z = betaRadial * erz + betaTangential * epz;
  return {
    betaRadial,
    betaTangential,
    x,
    z,
    speed: Math.hypot(x, z),
  };
}

export function isHorizonCrossed(state: FallState): boolean {
  return state.r <= 1.01;
}

export function isSingularityReached(state: FallState): boolean {
  return state.r <= SINGULARITY_CUTOFF + 0.002;
}

function schwarzschildF(r: number): number {
  return 1 - HORIZON / Math.max(r, SINGULARITY_CUTOFF);
}

function radialPotential(r: number, energy: number, angularMomentum: number): number {
  const f = schwarzschildF(Math.max(r, SINGULARITY_CUTOFF));
  return energy * energy - f * (1 + angularMomentum * angularMomentum / (r * r));
}

function deriv(state: FallState): Pick<FallState, 'r' | 'phi' | 't'> {
  const f = schwarzschildF(Math.max(state.r, SINGULARITY_CUTOFF));
  let radialSq = radialPotential(state.r, state.energy, state.angularMomentum);
  let sign = state.radialSign;
  if (radialSq < 1e-8 && sign > 0) {
    sign = -1;
    radialSq = 0;
  } else if (radialSq < 1e-10 && sign < 0) {
    radialSq = 1e-10;
  }
  return {
    r: sign * Math.sqrt(Math.max(radialSq, 0)),
    phi: state.angularMomentum / (state.r * state.r),
    t: state.r > HORIZON ? state.energy / Math.max(f, 1e-8) : 0,
  };
}

function rk4Step(state: FallState, h: number): FallState {
  const k1 = deriv(state);
  const s2 = offsetState(state, k1, h * 0.5);
  const k2 = deriv(s2);
  const s3 = offsetState(state, k2, h * 0.5);
  const k3 = deriv(s3);
  const s4 = offsetState(state, k3, h);
  const k4 = deriv(s4);
  const r = state.r + h / 6 * (k1.r + 2 * k2.r + 2 * k3.r + k4.r);
  const phi = state.phi + h / 6 * (k1.phi + 2 * k2.phi + 2 * k3.phi + k4.phi);
  const t = state.t + h / 6 * (k1.t + 2 * k2.t + 2 * k3.t + k4.t);
  const radialSq = radialPotential(Math.max(r, SINGULARITY_CUTOFF), state.energy, state.angularMomentum);
  const radialSign = radialSq < 1e-8 && state.radialSign > 0 ? -1 : state.radialSign;
  return {
    ...state,
    r: Math.max(r, SINGULARITY_CUTOFF),
    phi,
    t,
    tau: state.tau + h,
    radialSign,
  };
}

function offsetState(
  state: FallState,
  k: Pick<FallState, 'r' | 'phi' | 't'>,
  h: number,
): FallState {
  return {
    ...state,
    r: Math.max(state.r + k.r * h, SINGULARITY_CUTOFF),
    phi: state.phi + k.phi * h,
    t: state.t + k.t * h,
    tau: state.tau + h,
  };
}

function interiorVelocity(state: FallState): LocalVelocity {
  const erx = Math.cos(state.phi), erz = Math.sin(state.phi);
  const epx = -Math.sin(state.phi), epz = Math.cos(state.phi);
  const inward = Math.min(0.995, 0.82 + (1 - state.r) * 0.22);
  const swirl = Math.max(-0.35, Math.min(0.35, state.angularMomentum / Math.max(state.r * 8, 0.5)));
  const scale = Math.min(0.995 / Math.hypot(inward, swirl), 1);
  const betaRadial = -inward * scale;
  const betaTangential = swirl * scale;
  const x = betaRadial * erx + betaTangential * epx;
  const z = betaRadial * erz + betaTangential * epz;
  return {
    betaRadial,
    betaTangential,
    x,
    z,
    speed: Math.hypot(x, z),
  };
}
