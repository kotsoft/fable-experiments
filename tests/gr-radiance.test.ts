import { describe, expect, it } from 'vitest';
import { nullCovectorFromDirection, type GeodesicState } from '../src/gr/geodesic';
import { kerrSchildParams, metricDot } from '../src/gr/kerrSchild';
import {
  blackbodyRgb,
  diskEmitterFourVelocity,
  emissivityWeightedDiskAngularVelocity,
  kerrCircularOrbitAngularVelocity,
  redshiftFactor,
  sampleDiskRadiance,
  type DiskRadianceModel,
} from '../src/gr/radiance';

const model: DiskRadianceModel = {
  innerRadius: 3,
  outerRadius: 18,
  innerTemperature: 7200,
  emissivityScale: 1.25,
  boostPower: 4,
};

describe('disk radiance reference model', () => {
  it('constructs timelike normalized disk emitter velocities', () => {
    const params = kerrSchildParams(0.6, 1);
    const positions = [
      { x: 6, y: 0, z: 0 },
      { x: 8, y: 3, z: 0 },
      { x: -10, y: 2, z: 0 },
    ];

    for (const position of positions) {
      const velocity = diskEmitterFourVelocity(position, params);
      expect(metricDot(position, params, velocity, velocity)).toBeCloseTo(-1, 12);
      expect(velocity.t).toBeGreaterThan(0);
    }
  });

  it('computes different redshift factors for opposite disk rotation directions', () => {
    const params = kerrSchildParams(0, 1);
    const position = { t: 0, x: 8, y: 0, z: 0 };
    const photon = nullCovectorFromDirection(position, { x: 0, y: -1, z: 0 }, params);

    const prograde = diskEmitterFourVelocity({ x: 8, y: 0, z: 0 }, params, 1);
    const retrograde = diskEmitterFourVelocity({ x: 8, y: 0, z: 0 }, params, -1);

    expect(redshiftFactor(photon, retrograde)).toBeGreaterThan(redshiftFactor(photon, prograde));
  });

  it('samples hotter and brighter emission near the disk inner edge', () => {
    const params = kerrSchildParams(0.5, 1);
    const innerState = rayStateAt({ x: 4, y: 0, z: 0 }, { x: 0, y: -1, z: 0 }, params);
    const outerState = rayStateAt({ x: 12, y: 0, z: 0 }, { x: 0, y: -1, z: 0 }, params);

    const noBoostModel = { ...model, boostPower: 0 };
    const inner = sampleDiskRadiance(innerState, params, noBoostModel);
    const outer = sampleDiskRadiance(outerState, params, noBoostModel);

    expect(inner).not.toBeNull();
    expect(outer).not.toBeNull();
    expect(inner!.temperature).toBeGreaterThan(outer!.temperature);
    expect(inner!.bolometricIntensity).toBeGreaterThan(outer!.bolometricIntensity);
    expect(inner!.observedRgb.every(Number.isFinite)).toBe(true);
  });

  it('rejects samples outside the disk annulus', () => {
    const params = kerrSchildParams(0, 1);
    const tooClose = rayStateAt({ x: 2, y: 0, z: 0 }, { x: 0, y: -1, z: 0 }, params);
    const tooFar = rayStateAt({ x: 24, y: 0, z: 0 }, { x: 0, y: -1, z: 0 }, params);

    expect(sampleDiskRadiance(tooClose, params, model)).toBeNull();
    expect(sampleDiskRadiance(tooFar, params, model)).toBeNull();
  });

  it('modulates disk emission as the texture phase advances', () => {
    const params = kerrSchildParams(0.5, 1);
    const state = rayStateAt({ x: 8, y: 2, z: 0 }, { x: 0, y: -1, z: 0 }, params);
    const phaseZero = sampleDiskRadiance(state, params, { ...model, boostPower: 0, emissionPhase: 0 });
    const phaseShifted = sampleDiskRadiance(state, params, { ...model, boostPower: 0, emissionPhase: Math.PI / 6 });

    expect(phaseZero).not.toBeNull();
    expect(phaseShifted).not.toBeNull();
    expect(phaseZero!.temperature).toBeCloseTo(phaseShifted!.temperature, 12);
    expect(phaseZero!.bolometricIntensity).not.toBeCloseTo(phaseShifted!.bolometricIntensity, 6);
  });

  it('uses Kerr circular-orbit direction for disk pattern speed', () => {
    const params = kerrSchildParams(0.6, 1);
    const prograde = kerrCircularOrbitAngularVelocity(6, params, 1);
    const retrograde = kerrCircularOrbitAngularVelocity(6, params, -1);

    expect(prograde).toBeGreaterThan(0);
    expect(retrograde).toBeLessThan(0);
    expect(prograde).toBeCloseTo(1 / (Math.pow(6, 1.5) + 0.6), 12);
    expect(retrograde).toBeCloseTo(-1 / (Math.pow(6, 1.5) - 0.6), 12);
  });

  it('weights animated disk phase toward the brighter inner disk', () => {
    const params = kerrSchildParams(0.5, 1);
    const weighted = emissivityWeightedDiskAngularVelocity(3, 18, params, 1);
    const inner = kerrCircularOrbitAngularVelocity(3, params, 1);
    const outer = kerrCircularOrbitAngularVelocity(18, params, 1);

    expect(weighted).toBeLessThan(inner);
    expect(weighted).toBeGreaterThan(outer);
    expect(weighted).toBeGreaterThan((inner + outer) * 0.5);
  });

  it('keeps blackbody RGB in displayable normalized bounds before intensity scaling', () => {
    for (const temperature of [1800, 4500, 7200, 18000]) {
      const rgb = blackbodyRgb(temperature);
      expect(rgb.every((channel) => channel >= 0 && channel <= 1)).toBe(true);
    }
  });
});

function rayStateAt(
  position: { x: number; y: number; z: number },
  direction: { x: number; y: number; z: number },
  params = kerrSchildParams(0, 1),
): GeodesicState {
  const spacetimePosition = { t: 0, ...position };
  return {
    position: spacetimePosition,
    momentum: nullCovectorFromDirection(spacetimePosition, direction, params),
  };
}
