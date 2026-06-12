import { describe, expect, it } from 'vitest';
import { nullCovectorFromDirection, type GeodesicState } from '../src/gr/geodesic';
import { refineDiskCrossing } from '../src/gr/disk';
import { kerrSchildParams } from '../src/gr/kerrSchild';

describe('root-refined thin disk crossings', () => {
  it('finds an exact flat-space crossing between coarse samples', () => {
    const params = kerrSchildParams(0, 0);
    const position = { t: 0, x: 4, y: 0, z: 1 };
    const initial: GeodesicState = {
      position,
      momentum: nullCovectorFromDirection(position, { x: 0.5, y: 0, z: -1 }, params),
    };
    const crossing = refineDiskCrossing(initial, params, 4, { innerRadius: 3, outerRadius: 8 });

    expect(crossing).not.toBeNull();
    expect(Math.abs(crossing?.height ?? 1)).toBeLessThan(1e-8);
    expect(crossing?.radius).toBeGreaterThan(3);
    expect(crossing?.radius).toBeLessThan(8);
  });

  it('rejects crossings outside the disk annulus', () => {
    const params = kerrSchildParams(0, 0);
    const position = { t: 0, x: 1, y: 0, z: 1 };
    const initial: GeodesicState = {
      position,
      momentum: nullCovectorFromDirection(position, { x: 0, y: 0, z: -1 }, params),
    };
    const crossing = refineDiskCrossing(initial, params, 3, { innerRadius: 3, outerRadius: 8 });

    expect(crossing).toBeNull();
  });

  it('refines a Schwarzschild disk-plane crossing without large residual height', () => {
    const params = kerrSchildParams(0, 1);
    const position = { t: 0, x: 8, y: 0, z: 1.2 };
    const initial: GeodesicState = {
      position,
      momentum: nullCovectorFromDirection(position, { x: -0.2, y: 0.1, z: -1 }, params),
    };
    const crossing = refineDiskCrossing(initial, params, 3, { innerRadius: 3, outerRadius: 12 });

    expect(crossing).not.toBeNull();
    expect(Math.abs(crossing?.height ?? 1)).toBeLessThan(1e-8);
    expect(crossing?.affineParameter).toBeGreaterThan(0);
    expect(crossing?.affineParameter).toBeLessThan(3);
  });

  it('supports inclined diagnostic disk planes', () => {
    const params = kerrSchildParams(0, 0);
    const position = { t: 0, x: 5, y: 0, z: 1 };
    const initial: GeodesicState = {
      position,
      momentum: nullCovectorFromDirection(position, { x: 0, y: 0.4, z: -1 }, params),
    };
    const crossing = refineDiskCrossing(
      initial,
      params,
      4,
      { innerRadius: 3, outerRadius: 8, normal: { x: 0, y: 0.2, z: 1 } },
    );

    expect(crossing).not.toBeNull();
    expect(Math.abs(crossing?.height ?? 1)).toBeLessThan(1e-8);
  });
});
