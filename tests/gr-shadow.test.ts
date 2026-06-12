import { describe, expect, it } from 'vitest';
import { kerrSchildParams } from '../src/gr/kerrSchild';
import {
  equatorialPhotonOrbitRadii,
  kerrShadowBoundary,
  schwarzschildCriticalImpactParameter,
  shadowBounds,
} from '../src/gr/shadow';

describe('analytic Kerr shadow benchmarks', () => {
  it('recovers the Schwarzschild critical impact circle', () => {
    const params = kerrSchildParams(0, 1);
    const boundary = kerrShadowBoundary(params, Math.PI / 2, 64);
    const bounds = shadowBounds(boundary);

    expect(schwarzschildCriticalImpactParameter(params)).toBeCloseTo(3 * Math.sqrt(3), 12);
    expect(bounds.centerAlpha).toBeCloseTo(0, 12);
    expect(bounds.horizontalDiameter).toBeCloseTo(6 * Math.sqrt(3), 2);
    expect(bounds.verticalDiameter).toBeCloseTo(6 * Math.sqrt(3), 2);
    for (const point of boundary) {
      expect(Math.hypot(point.alpha, point.beta)).toBeCloseTo(3 * Math.sqrt(3), 12);
      expect(point.sphericalOrbitRadius).toBeCloseTo(3, 12);
    }
  });

  it('computes Kerr prograde and retrograde equatorial photon orbit radii', () => {
    const params = kerrSchildParams(0.8, 1);
    const radii = equatorialPhotonOrbitRadii(params);

    expect(radii.prograde).toBeCloseTo(1.8110859802363672, 12);
    expect(radii.retrograde).toBeCloseTo(3.818763716895554, 12);
    expect(radii.prograde).toBeLessThan(3);
    expect(radii.retrograde).toBeGreaterThan(3);
  });

  it('captures the asymmetric equatorial Kerr shadow boundary', () => {
    const params = kerrSchildParams(0.8, 1);
    const boundary = kerrShadowBoundary(params, Math.PI / 2, 129);
    const bounds = shadowBounds(boundary);

    expect(boundary.length).toBeGreaterThan(120);
    expect(bounds.minAlpha).toBeCloseTo(-3.2372978366882093, 5);
    expect(bounds.maxAlpha).toBeCloseTo(6.66249720273366, 5);
    expect(bounds.maxBeta).toBeGreaterThan(5.1);
    expect(bounds.centerAlpha).toBeGreaterThan(1.6);
  });

  it('mirrors the horizontal shadow offset when spin reverses', () => {
    const positive = shadowBounds(kerrShadowBoundary(kerrSchildParams(0.65, 1), Math.PI / 2, 129));
    const negative = shadowBounds(kerrShadowBoundary(kerrSchildParams(-0.65, 1), Math.PI / 2, 129));

    expect(positive.centerAlpha).toBeCloseTo(-negative.centerAlpha, 10);
    expect(positive.horizontalDiameter).toBeCloseTo(negative.horizontalDiameter, 10);
    expect(positive.verticalDiameter).toBeCloseTo(negative.verticalDiameter, 10);
  });
});
