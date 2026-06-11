import { describe, expect, it } from 'vitest';
import { hamiltonian, type GeodesicState } from '../src/gr/geodesic';
import { kerrSchildParams, type Vec4 } from '../src/gr/kerrSchild';
import {
  buildObserverTetrad,
  launchPhotonFromTetrad,
  staticObserverFourVelocity,
  tetradResidual,
} from '../src/gr/tetrad';

describe('curved-metric observer tetrads', () => {
  it('builds an orthonormal Schwarzschild observer tetrad', () => {
    const params = kerrSchildParams(0, 1);
    const position = { x: 12, y: 0, z: 0 };
    const tetrad = buildObserverTetrad(position, params, staticObserverFourVelocity(position, params));

    expect(tetradResidual(position, params, tetrad)).toBeLessThan(1e-12);
  });

  it('builds an orthonormal Kerr observer tetrad outside the ergoregion', () => {
    const params = kerrSchildParams(0.85, 1);
    const position = { x: 8, y: 1.5, z: 2.2 };
    const tetrad = buildObserverTetrad(position, params, staticObserverFourVelocity(position, params));

    expect(tetradResidual(position, params, tetrad)).toBeLessThan(1e-12);
  });

  it('launches null photon covectors from local tetrad directions', () => {
    const params = kerrSchildParams(0.7, 1);
    const position = { x: 7, y: -2, z: 1 };
    const tetrad = buildObserverTetrad(position, params, staticObserverFourVelocity(position, params));
    const directions = [
      { x: 0, y: 0, z: 1 },
      { x: 0.3, y: 0.15, z: 1 },
      { x: -0.45, y: 0.2, z: 1 },
    ];

    for (const direction of directions) {
      const momentum = launchPhotonFromTetrad(position, params, tetrad, direction, 2.5);
      const state: GeodesicState = { position: { t: 0, ...position }, momentum };

      expect(Math.abs(hamiltonian(state, params))).toBeLessThan(2e-12);
      expect(covectorContraction(momentum, tetrad.eTime)).toBeCloseTo(-2.5, 12);
    }
  });

  it('rejects static observers inside the Schwarzschild horizon', () => {
    const params = kerrSchildParams(0, 1);

    expect(() => staticObserverFourVelocity({ x: 1.5, y: 0, z: 0 }, params)).toThrow(
      'Static observer is not timelike',
    );
  });
});

function covectorContraction(covector: Vec4, vector: Vec4): number {
  return covector.t * vector.t + covector.x * vector.x + covector.y * vector.y + covector.z * vector.z;
}
