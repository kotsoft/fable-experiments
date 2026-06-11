import { describe, expect, it } from 'vitest';
import {
  coordinateVelocity,
  hamiltonian,
  nullCovectorFromDirection,
  stepNullGeodesic,
  traceNullGeodesic,
  type GeodesicState,
} from '../src/gr/geodesic';
import { horizonRadius, kerrSchildParams, kerrSchildRadius, type Vec4 } from '../src/gr/kerrSchild';

describe('CPU null geodesic integrator', () => {
  it('launches null photon covectors in flat, Schwarzschild, and Kerr limits', () => {
    const cases = [
      { params: kerrSchildParams(0, 0), position: { t: 0, x: 3, y: 2, z: -1 } },
      { params: kerrSchildParams(0, 1), position: { t: 0, x: 8, y: 0, z: 1 } },
      { params: kerrSchildParams(0.7, 1), position: { t: 0, x: 5, y: 2, z: 3 } },
    ];

    for (const { params, position } of cases) {
      const momentum = nullCovectorFromDirection(position, { x: -0.4, y: 0.2, z: 1 }, params);
      expect(Math.abs(hamiltonian({ position, momentum }, params))).toBeLessThan(1e-12);
    }
  });

  it('keeps flat-space rays straight with constant momentum', () => {
    const params = kerrSchildParams(0, 0);
    const initial: GeodesicState = {
      position: { t: 0, x: 0, y: 0, z: 0 },
      momentum: nullCovectorFromDirection({ t: 0, x: 0, y: 0, z: 0 }, { x: 1, y: 0.5, z: -0.25 }, params),
    };
    const velocity = coordinateVelocity(initial.position, initial.momentum, params);
    let state = initial;

    for (let i = 0; i < 20; i++) state = stepNullGeodesic(state, params, 0.25);

    expect(state.position.t).toBeCloseTo(velocity.t * 5, 12);
    expect(state.position.x).toBeCloseTo(velocity.x * 5, 12);
    expect(state.position.y).toBeCloseTo(velocity.y * 5, 12);
    expect(state.position.z).toBeCloseTo(velocity.z * 5, 12);
    expectVec4Close(state.momentum, initial.momentum, 12);
  });

  it('escapes for an outward Schwarzschild ray while preserving the null constraint', () => {
    const params = kerrSchildParams(0, 1);
    const initial: GeodesicState = {
      position: { t: 0, x: 12, y: 0, z: 0 },
      momentum: nullCovectorFromDirection({ t: 0, x: 12, y: 0, z: 0 }, { x: 1, y: 0.08, z: 0 }, params),
    };

    const result = traceNullGeodesic(initial, params, {
      stepSize: 0.04,
      maxSteps: 2500,
      escapeRadius: 28,
      singularityRadius: 0.2,
    });

    expect(result.status).toBe('escaped');
    expect(result.maxHamiltonianDrift).toBeLessThan(2e-7);
  });

  it('crosses the Schwarzschild horizon for an inward radial ray', () => {
    const params = kerrSchildParams(0, 1);
    const initial: GeodesicState = {
      position: { t: 0, x: 12, y: 0, z: 0 },
      momentum: nullCovectorFromDirection({ t: 0, x: 12, y: 0, z: 0 }, { x: -1, y: 0, z: 0 }, params),
    };

    const result = traceNullGeodesic(initial, params, {
      stepSize: 0.025,
      maxSteps: 3000,
      escapeRadius: 30,
      singularityRadius: 0.2,
    });

    expect(result.status).toBe('horizon');
    expect(kerrSchildRadius(position3(result.state.position), params)).toBeLessThanOrEqual(horizonRadius(params));
    expect(result.maxHamiltonianDrift).toBeLessThan(2e-7);
  });

  it('tracks a Kerr ray without large Hamiltonian drift', () => {
    const params = kerrSchildParams(0.8, 1);
    const initial: GeodesicState = {
      position: { t: 0, x: 8, y: -2, z: 1.5 },
      momentum: nullCovectorFromDirection({ t: 0, x: 8, y: -2, z: 1.5 }, { x: 1, y: 0.25, z: -0.08 }, params),
    };

    const result = traceNullGeodesic(initial, params, {
      stepSize: 0.025,
      maxSteps: 2200,
      escapeRadius: 32,
      singularityRadius: 0.2,
    });

    expect(['escaped', 'max-steps']).toContain(result.status);
    expect(result.maxHamiltonianDrift).toBeLessThan(1e-6);
  });
});

function position3(v: Vec4): { x: number; y: number; z: number } {
  return { x: v.x, y: v.y, z: v.z };
}

function expectVec4Close(actual: Vec4, expected: Vec4, precision: number): void {
  expect(actual.t).toBeCloseTo(expected.t, precision);
  expect(actual.x).toBeCloseTo(expected.x, precision);
  expect(actual.y).toBeCloseTo(expected.y, precision);
  expect(actual.z).toBeCloseTo(expected.z, precision);
}
