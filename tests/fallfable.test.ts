import { describe, expect, it } from 'vitest';
import {
  buildTetrad,
  circularOrbitFourVelocity,
  eulerianObserver,
  hamiltonian,
  horizonRadius,
  iscoRadius,
  kerrParams,
  ksRadius,
  lowerVector,
  metricDot,
  phaseDerivative,
  photonOrbitRadius,
  rk4Step,
  spatial,
  tetradResidual,
  type PhaseState,
  type Vec3,
  type Vec4,
} from '../src/fallfable/kerr';
import {
  HORIZON,
  PARAMS,
  PRESETS,
  constraintResidual,
  launchLocal,
  localSpeed,
  stepPlayer,
} from '../src/fallfable/player';

const TEST_POINTS: Vec3[] = [
  { x: 6, y: 2, z: 1.5 },          // generic exterior
  { x: 0.95, y: 0.2, z: 0.1 },     // inside the ergosphere
  { x: 0.6, y: -0.3, z: 0.2 },     // inside the horizon
  { x: 1.2, y: 0.4, z: -2.8 },     // off-equator
];

const TEST_MOMENTA: Vec4[] = [
  { t: -1, x: 0.4, y: -0.8, z: 0.45 },
  { t: -1.3, x: -1.1, y: 0.2, z: 0.7 },
];

describe('fallfable kerr core', () => {
  it('matches the analytic Hamiltonian gradient against f64 finite differences', () => {
    for (const p of TEST_POINTS) {
      for (const m of TEST_MOMENTA) {
        const d = phaseDerivative(p, m, PARAMS);
        const eps = 1e-7 * Math.max(ksRadius(p, PARAMS), 1);
        for (const axis of ['x', 'y', 'z'] as const) {
          const plus = { ...p, [axis]: p[axis] + eps };
          const minus = { ...p, [axis]: p[axis] - eps };
          const fd = (hamiltonian(plus, m, PARAMS) - hamiltonian(minus, m, PARAMS)) / (2 * eps);
          // force = -dH/dx
          expect(-d.force[axis]).toBeCloseTo(fd, 5);
        }
      }
    }
  });

  it('matches dx/dlambda = dH/dp against finite differences', () => {
    const p = TEST_POINTS[0];
    const m = TEST_MOMENTA[0];
    const d = phaseDerivative(p, m, PARAMS);
    const eps = 1e-7;
    const components: (keyof Vec4)[] = ['t', 'x', 'y', 'z'];
    for (const c of components) {
      const plus = { ...m, [c]: m[c] + eps };
      const minus = { ...m, [c]: m[c] - eps };
      const fd = (hamiltonian(p, plus, PARAMS) - hamiltonian(p, minus, PARAMS)) / (2 * eps);
      expect(d.velocity[c]).toBeCloseTo(fd, 6);
    }
  });

  it('conserves H, energy and angular momentum along a null geodesic', () => {
    const p: Vec3 = { x: 5, y: 0, z: 0.4 };
    const frame = buildTetrad(p, PARAMS, eulerianObserver(p, PARAMS), { x: -1, y: 0.3, z: 0 }, { x: 0, y: 0, z: 1 });
    const direction = { x: 0.5, y: 0.2, z: Math.sqrt(1 - 0.25 - 0.04) };
    const contra: Vec4 = {
      t: -frame.eTime.t + direction.x * frame.eRight.t + direction.y * frame.eUp.t + direction.z * frame.eForward.t,
      x: -frame.eTime.x + direction.x * frame.eRight.x + direction.y * frame.eUp.x + direction.z * frame.eForward.x,
      y: -frame.eTime.y + direction.x * frame.eRight.y + direction.y * frame.eUp.y + direction.z * frame.eForward.y,
      z: -frame.eTime.z + direction.x * frame.eRight.z + direction.y * frame.eUp.z + direction.z * frame.eForward.z,
    };
    let s: PhaseState = { position: { t: 0, ...p }, momentum: lowerVector(p, PARAMS, contra) };
    const lz0 = s.position.x * s.momentum.y - s.position.y * s.momentum.x;
    const e0 = s.momentum.t;

    let maxH = 0;
    for (let i = 0; i < 4000; i++) {
      const r = ksRadius(spatial(s.position), PARAMS);
      s = rk4Step(s, PARAMS, 0.01 * Math.min(Math.max(r * 0.55, 0.16), 6));
      maxH = Math.max(maxH, Math.abs(hamiltonian(spatial(s.position), s.momentum, PARAMS)));
      if (ksRadius(spatial(s.position), PARAMS) > 40) break;
    }
    const lz = s.position.x * s.momentum.y - s.position.y * s.momentum.x;
    expect(s.momentum.t).toBe(e0); // exact: stationarity is built in
    expect(Math.abs(lz - lz0)).toBeLessThan(1e-8);
    expect(maxH).toBeLessThan(1e-8);
  });

  it('escapes the horizon when tracing past-directed rays from inside', () => {
    // Fall a player through the horizon first.
    let player = launchLocal({ r: 4, phi: 0.4, betaRadial: 0, betaTangential: 0.15 });
    let guard = 0;
    while (player.r > HORIZON * 0.85 && guard++ < 30000) player = stepPlayer(player, 0.01);
    expect(player.r).toBeLessThan(HORIZON * 0.86);

    const p = spatial(player.position);
    const frame = buildTetrad(p, PARAMS, eulerianObserver(p, PARAMS), { x: 1, y: 0, z: 0 }, { x: 0, y: 0, z: 1 });
    // A transverse view direction: its backward ray must exit and keep going.
    const contra: Vec4 = addScaled(neg(frame.eTime), frame.eUp, 1);
    let ray: PhaseState = { position: player.position, momentum: lowerVector(p, PARAMS, contra) };
    const m0 = momentumScale(ray.momentum);
    let maxR = 0;
    for (let i = 0; i < 6000; i++) {
      const r = ksRadius(spatial(ray.position), PARAMS);
      maxR = Math.max(maxR, r);
      const scale = momentumScale(ray.momentum);
      if (scale > 2.5e3 * m0) break;
      const h = 0.01 * Math.min(Math.max(r * 0.55, 0.16), 6) * Math.min(1, m0 / scale);
      ray = rk4Step(ray, PARAMS, h);
    }
    expect(maxR).toBeGreaterThan(HORIZON * 3);
    // And it travelled into the past.
    expect(ray.position.t).toBeLessThan(player.position.t);
  });

  it('never lets a past-directed ray enter the horizon from outside', () => {
    const player = launchLocal({ r: 3, phi: 0, betaRadial: -0.5, betaTangential: 0 });
    const p = spatial(player.position);
    const frame = buildTetrad(p, PARAMS, eulerianObserver(p, PARAMS), { x: -1, y: 0, z: 0 }, { x: 0, y: 0, z: 1 });
    // Aim straight at the hole.
    const contra: Vec4 = addScaled(neg(frame.eTime), frame.eForward, 1);
    let ray: PhaseState = { position: player.position, momentum: lowerVector(p, PARAMS, contra) };
    const m0 = momentumScale(ray.momentum);
    let minR = Infinity;
    for (let i = 0; i < 8000; i++) {
      const r = ksRadius(spatial(ray.position), PARAMS);
      minR = Math.min(minR, r);
      const scale = momentumScale(ray.momentum);
      if (scale > 2.5e3 * m0) break;
      const h = 0.01 * Math.min(Math.max(r * 0.55, 0.16), 6) * Math.min(1, m0 / scale);
      ray = rk4Step(ray, PARAMS, h);
    }
    expect(minR).toBeGreaterThan(HORIZON * 0.985);
  });
});

describe('fallfable observers and launches', () => {
  it('keeps the Eulerian frame timelike and orthonormal at all radii', () => {
    for (const r of [6, 1.5, 0.9, 0.5, 0.2]) {
      const p: Vec3 = { x: Math.sqrt(r * r + PARAMS.spin * PARAMS.spin), y: 0, z: 0 };
      const u = eulerianObserver(p, PARAMS);
      expect(metricDot(p, PARAMS, u, u)).toBeCloseTo(-1, 10);
      const frame = buildTetrad(p, PARAMS, u, { x: -1, y: 0.2, z: 0 }, { x: 0, y: 0, z: 1 });
      expect(tetradResidual(p, PARAMS, frame)).toBeLessThan(1e-10);
    }
  });

  it('launches from rest inside the ergosphere (static observers cannot)', () => {
    const inErgo = (HORIZON + 2 * PARAMS.mass) / 2;
    const state = launchLocal({ r: inErgo, phi: 1.0, betaRadial: 0, betaTangential: 0 });
    expect(constraintResidual(state)).toBeLessThan(1e-10);
    expect(localSpeed(state)).toBeLessThan(1e-8);
    // Sanity: a static worldline really is impossible here, so the metric's
    // t-t component must be positive (spacelike) at this radius.
    const p = spatial(state.position);
    const staticCandidate: Vec4 = { t: 1, x: 0, y: 0, z: 0 };
    expect(metricDot(p, PARAMS, staticCandidate, staticCandidate)).toBeGreaterThan(0);
  });

  it('keeps the timelike constraint through the horizon and tracks dragging', () => {
    let state = launchLocal({ r: 5, phi: 0, betaRadial: 0, betaTangential: 0 });
    let guard = 0;
    while (state.r > 0.3 && guard++ < 40000) state = stepPlayer(state, 0.01);
    expect(state.r).toBeLessThanOrEqual(0.3);
    expect(constraintResidual(state)).toBeLessThan(1e-6);
    // Frame dragging: a drop from rest must have been swept in the +phi sense.
    const phi = Math.atan2(state.position.y, state.position.x);
    expect(phi).toBeGreaterThan(0.02);
  });

  it('holds a circular orbit at r = 4 for a full period', () => {
    const r = 4;
    const p: Vec3 = { x: Math.sqrt(r * r + PARAMS.spin * PARAMS.spin), y: 0, z: 0 };
    const u = circularOrbitFourVelocity(p, PARAMS, true);
    let s: PhaseState = { position: { t: 0, ...p }, momentum: lowerVector(p, PARAMS, u) };
    const omega = Math.sqrt(PARAMS.mass) / (Math.pow(r, 1.5) + PARAMS.spin * Math.sqrt(PARAMS.mass));
    const period = (2 * Math.PI) / omega;
    let steps = 0;
    while (s.position.t < period && steps++ < 200000) {
      s = rk4Step(s, PARAMS, 0.01);
    }
    expect(ksRadius(spatial(s.position), PARAMS)).toBeCloseTo(r, 4);
    expect(Math.abs(s.position.z)).toBeLessThan(1e-9);
  });

  it('computes textbook ISCO and photon orbit radii', () => {
    const schwarzschild = kerrParams(0, 1);
    expect(iscoRadius(schwarzschild, true)).toBeCloseTo(6, 10);
    expect(photonOrbitRadius(schwarzschild, true)).toBeCloseTo(3, 10);
    const nearExtremal = kerrParams(0.999999, 1);
    expect(iscoRadius(nearExtremal, true)).toBeLessThan(1.05);
    expect(horizonRadius(nearExtremal)).toBeCloseTo(1, 2);
  });

  it('creates valid states from every preset', () => {
    for (const preset of PRESETS) {
      const state = preset.create();
      expect(Number.isFinite(state.r)).toBe(true);
      expect(constraintResidual(state)).toBeLessThan(1e-9);
    }
  });
});

function momentumScale(m: Vec4): number {
  return Math.abs(m.t) + Math.hypot(m.x, m.y, m.z);
}

function neg(v: Vec4): Vec4 {
  return { t: -v.t, x: -v.x, y: -v.y, z: -v.z };
}

function addScaled(a: Vec4, b: Vec4, s: number): Vec4 {
  return { t: a.t + b.t * s, x: a.x + b.x * s, y: a.y + b.y * s, z: a.z + b.z * s };
}
