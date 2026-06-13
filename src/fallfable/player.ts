// The player worldline: a timelike geodesic of the Kerr metric integrated on
// the CPU in double precision, plus the launch helpers the trajectory planner
// uses to drop the player at points of interest with an exact velocity.

import {
  boostedFourVelocity,
  buildTetrad,
  circularOrbitFourVelocity,
  equatorialRho,
  eulerianObserver,
  hamiltonian,
  horizonRadius,
  iscoRadius,
  kerrParams,
  ksRadius,
  lowerVector,
  metricDot,
  normalize3,
  phaseDerivative,
  photonOrbitRadius,
  rk4Step,
  spatial,
  type KerrParams,
  type PhaseState,
  type Vec3,
  type Vec4,
} from './kerr';

export const PARAMS: KerrParams = kerrParams(0.425, 0.5); // chi = 0.85
export const HORIZON = horizonRadius(PARAMS);
export const INNER_HORIZON = PARAMS.mass - Math.sqrt(Math.max(PARAMS.mass ** 2 - PARAMS.spin ** 2, 0));
export const ERGOSPHERE = 2 * PARAMS.mass;
export const ISCO = iscoRadius(PARAMS, true);
export const PHOTON_PROGRADE = photonOrbitRadius(PARAMS, true);
export const PHOTON_RETROGRADE = photonOrbitRadius(PARAMS, false);
export const SINGULARITY_CUTOFF = 0.02;
export const MAP_RADIUS = 16;
export const DISK_INNER = ISCO;
export const DISK_OUTER = 13;
const MANUAL_CARRY_HORIZONTAL_SCALE = 0.992;
const MANUAL_CARRY_VERTICAL_SCALE = 0.96;

export interface PlayerState extends PhaseState {
  r: number;
  tau: number;
  /** Set when the worldline reached the cutoff or lost numerical validity. */
  ended?: boolean;
}

export interface LocalLaunch {
  /** KS radius. */
  r: number;
  phi: number;
  /** Local velocity relative to the Eulerian observer, as fractions of c. */
  betaRadial: number;
  betaTangential: number;
  /** Optional Cartesian height above the disk plane. */
  height?: number;
}

function makeState(state: PhaseState, tau: number): PlayerState {
  return { ...state, r: ksRadius(spatial(state.position), PARAMS), tau };
}

function equatorialPosition(r: number, phi: number): Vec4 {
  const rho = equatorialRho(r, PARAMS);
  return { t: 0, x: rho * Math.cos(phi), y: rho * Math.sin(phi), z: 0 };
}

function stateFromFourVelocity(position: Vec4, u: Vec4): PlayerState {
  return makeState({ position, momentum: lowerVector(spatial(position), PARAMS, u) }, 0);
}

/**
 * Launch with a local velocity measured by the Eulerian (Kerr-Schild slicing)
 * observer. That frame is timelike everywhere, so the planner may place the
 * player inside the ergosphere - or even inside the horizon.
 */
export function launchLocal(launch: LocalLaunch): PlayerState {
  const position = equatorialPosition(launch.r, launch.phi);
  position.z = launch.height ?? 0;
  const p = spatial(position);
  const radial = normalize3({ x: Math.cos(launch.phi), y: Math.sin(launch.phi), z: 0 });
  const tangent: Vec3 = { x: -radial.y, y: radial.x, z: 0 };
  const frame = buildTetrad(p, PARAMS, eulerianObserver(p, PARAMS), radial, { x: 0, y: 0, z: 1 });
  // eForward is the radial hint; tangential velocity rides along eRight, whose
  // Gram-Schmidt sign we align with the +phi tangent.
  const rightSign = Math.sign(frame.eRight.x * tangent.x + frame.eRight.y * tangent.y) || 1;
  const u = boostedFourVelocity(frame, {
    x: launch.betaRadial,
    y: rightSign * launch.betaTangential,
    z: 0,
  });
  return stateFromFourVelocity(position, u);
}

export function stepPlayer(state: PlayerState, dTau: number): PlayerState {
  if (state.ended || state.r <= SINGULARITY_CUTOFF) return { ...state, ended: true };
  let phase: PhaseState = state;
  let r = state.r;
  let remaining = dTau;
  let advanced = 0;
  while (remaining > 1e-9) {
    // Substep shrinks like r^2 so the integrator survives the violent
    // gradients near the ring region deep inside the inner horizon.
    const h = Math.min(remaining, Math.max(2e-6, 0.0025 * Math.min(r * r * r, 1)));
    const nextPhase = rk4Step(phase, PARAMS, h);
    if (!isFinitePhase(nextPhase)) {
      return recoverInvalidStep(makeState(phase, state.tau + advanced), h);
    }
    phase = nextPhase;
    advanced += h;
    remaining -= h;
    r = ksRadius(spatial(phase.position), PARAMS);
    if (!Number.isFinite(r)) {
      return recoverInvalidStep(makeState(phase, state.tau + advanced), h);
    }
    if (r <= SINGULARITY_CUTOFF) {
      return { ...phase, r, tau: state.tau + advanced, ended: true };
    }
  }
  const next: PlayerState = { ...phase, r, tau: state.tau + advanced };
  const residual = constraintResidual(next);
  if (!Number.isFinite(residual) || residual > 0.05) {
    return recoverInvalidStep(state, dTau);
  }
  return next;
}

function isFinitePhase(phase: PhaseState): boolean {
  return (
    Number.isFinite(phase.position.t) &&
    Number.isFinite(phase.position.x) &&
    Number.isFinite(phase.position.y) &&
    Number.isFinite(phase.position.z) &&
    Number.isFinite(phase.momentum.t) &&
    Number.isFinite(phase.momentum.x) &&
    Number.isFinite(phase.momentum.y) &&
    Number.isFinite(phase.momentum.z)
  );
}

function recoverInvalidStep(state: PlayerState, dTau: number): PlayerState {
  if (state.r > INNER_HORIZON * 2) {
    return { ...state, ended: true };
  }
  return carryObserverInward(state, dTau);
}

function carryObserverInward(state: PlayerState, dTau: number): PlayerState {
  // A generic worldline cannot be smoothly continued through the Cauchy
  // horizon in this chart (its momentum diverges on the sheet the ingoing
  // coordinates do not cover) - and beyond it GR is non-deterministic
  // anyway. Carry the indestructible observer inward by hand, at rest in
  // the always-timelike Eulerian frame.
  // The fallback is tuned to contract x/y gently for visual continuity while
  // damping z faster so off-plane numerical drift settles back toward the disk.
  const p: Vec3 = {
    x: state.position.x * MANUAL_CARRY_HORIZONTAL_SCALE,
    y: state.position.y * MANUAL_CARRY_HORIZONTAL_SCALE,
    z: state.position.z * MANUAL_CARRY_VERTICAL_SCALE,
  };
  const u = eulerianObserver(p, PARAMS);
  return makeState(
    {
      position: { t: state.position.t + dTau, ...p },
      momentum: lowerVector(p, PARAMS, u),
    },
    state.tau + dTau,
  );
}

export interface PreviewPoint {
  x: number;
  y: number;
}

/** Coarse look-ahead of the worldline for the planner overlay. */
export function previewPath(state: PlayerState, maxTau: number, maxPoints = 160): PreviewPoint[] {
  const points: PreviewPoint[] = [{ x: state.position.x, y: state.position.y }];
  let s = state;
  const step = maxTau / maxPoints;
  for (let i = 0; i < maxPoints; i++) {
    s = stepPlayer(s, step);
    points.push({ x: s.position.x, y: s.position.y });
    if (s.ended || s.r <= SINGULARITY_CUTOFF || s.r > MAP_RADIUS * 1.4) break;
  }
  return points;
}

export function constraintResidual(state: PlayerState): number {
  return Math.abs(hamiltonian(spatial(state.position), state.momentum, PARAMS) + 0.5);
}

/** Player speed measured by the local Eulerian observer (always defined). */
export function localSpeed(state: PlayerState): number {
  const p = spatial(state.position);
  const uObs = eulerianObserver(p, PARAMS);
  // gamma = -g(u_player, u_obs); u_player = g^{-1} momentum, so gamma = -p_m u_obs^m.
  const gamma = Math.max(
    -(state.momentum.t * uObs.t + state.momentum.x * uObs.x + state.momentum.y * uObs.y + state.momentum.z * uObs.z),
    1,
  );
  return Math.sqrt(Math.max(0, 1 - 1 / (gamma * gamma)));
}

export interface Preset {
  id: string;
  label: string;
  description: string;
  exposure?: number;
  create(): PlayerState;
}

/** Points of interest the planner can jump to with exact initial conditions. */
export const PRESETS: Preset[] = [
  {
    id: 'plunge',
    label: 'plunge',
    description: 'free fall from above the disk plane',
    create: () => launchLocal({ r: 10, phi: Math.PI * 0.25, betaRadial: 0, betaTangential: 0.05, height: 1.6 }),
  },
  {
    id: 'isco',
    label: 'ISCO orbit',
    description: 'marginally stable circular orbit',
    exposure: 0.1,
    create: () => {
      const position = equatorialPosition(ISCO * 1.002, 0);
      return stateFromFourVelocity(position, circularOrbitFourVelocity(spatial(position), PARAMS, true));
    },
  },
  {
    id: 'whirl',
    label: 'photon skim',
    description: 'zoom-whirl past the prograde photon orbit',
    exposure: 0.1,
    create: () => launchLocal({ r: PHOTON_PROGRADE * 2.4, phi: Math.PI, betaRadial: -0.34, betaTangential: 0.62 }),
  },
  {
    id: 'ergo',
    label: 'ergosphere surf',
    description: 'launched retrograde, dragged prograde',
    create: () => launchLocal({ r: ERGOSPHERE * 1.45, phi: -Math.PI / 2, betaRadial: 0, betaTangential: -0.55 }),
  },
  {
    id: 'polar',
    label: 'polar plunge',
    description: 'fall along the spin axis, disk as a halo',
    create: () => launchLocal({ r: 0.2, phi: 0, betaRadial: 0, betaTangential: 0, height: 9 }),
  },
  {
    id: 'inclined',
    label: 'inclined orbit',
    description: 'tilted rosette threading the disk every half lap',
    create: () => launchLocal({ r: 5, phi: 0, betaRadial: 0, betaTangential: 0.36, height: 1.4 }),
  },
  {
    id: 'dive',
    label: 'disk dive',
    description: 'sub-orbital spiral sinking through the disk gas',
    create: () => launchLocal({ r: 7, phi: Math.PI / 2, betaRadial: -0.05, betaTangential: 0.27, height: 0.45 }),
  },
  {
    id: 'retro',
    label: 'retro orbit',
    description: 'retrograde circular orbit, fighting the frame drag',
    create: () => {
      const position = equatorialPosition(4.6, Math.PI);
      return stateFromFourVelocity(position, circularOrbitFourVelocity(spatial(position), PARAMS, false));
    },
  },
];

/**
 * u^m = g^mn p_n, renormalized so the camera sits exactly on the mass shell.
 * If integration drift near the Cauchy horizon has pushed the raised momentum
 * off the timelike cone, fall back to the local Eulerian observer so the
 * camera frame never degenerates.
 */
export function fourVelocity(state: PlayerState): Vec4 {
  const p = spatial(state.position);
  const u = phaseDerivative(p, state.momentum, PARAMS).velocity;
  const norm = metricDot(p, PARAMS, u, u);
  if (!(norm < -1e-9)) return eulerianObserver(p, PARAMS);
  const inv = 1 / Math.sqrt(-norm);
  return { t: u.t * inv, x: u.x * inv, y: u.y * inv, z: u.z * inv };
}
