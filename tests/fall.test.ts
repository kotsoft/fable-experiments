import { describe, expect, it } from 'vitest';
import {
  isHorizonCrossed,
  isSingularityReached,
  launchFromLocal,
  localVelocity,
  stepFall,
  type FallState,
} from '../src/fall/physics';
import { createPlayerCamera, observerFrameFromState, tetradResidual } from '../src/fall/tetrad';

const SINGULARITY_CUTOFF = 0.08;

describe('fall geodesics', () => {
  it('conserves energy and angular momentum during integration', () => {
    const launched = launchFromLocal({ r: 10, phi: 0, betaRadial: -0.2, betaTangential: 0.25 });
    let state = launched;
    for (let i = 0; i < 400; i++) state = stepFall(state, 0.02);

    expect(state.energy).toBe(launched.energy);
    expect(state.angularMomentum).toBe(launched.angularMomentum);
  });

  it('keeps local velocity consistent with the Schwarzschild energy relation', () => {
    let state = launchFromLocal({ r: 10, phi: 0, betaRadial: -0.2, betaTangential: 0.25 });
    for (let i = 0; i < 400; i++) state = stepFall(state, 0.02);

    const velocity = localVelocity(state);
    const residual = Math.abs(
      velocity.betaRadial * velocity.betaRadial +
        velocity.betaTangential * velocity.betaTangential -
        (1 - schwarzschildF(state.r) / (state.energy * state.energy)),
    );

    expect(residual).toBeLessThan(1e-4);
  });

  it('falls inward when released from local rest', () => {
    const rest = launchFromLocal({ r: 8, phi: 0, betaRadial: 0, betaTangential: 0 });
    const next = stepFall(rest, 0.5);

    expect(next.r).toBeLessThan(rest.r);
  });

  it('advances distant time faster than proper time near the horizon', () => {
    let near = launchFromLocal({ r: 1.2, phi: 0, betaRadial: 0, betaTangential: 0 });
    near = stepFall(near, 0.2);

    expect(near.t).toBeGreaterThan(near.tau);
  });

  it('crosses the horizon and reaches the singularity cutoff in finite proper time', () => {
    let plunge = launchFromLocal({ r: 4, phi: 0, betaRadial: 0, betaTangential: 0 });
    for (let i = 0; i < 2000 && !isHorizonCrossed(plunge); i++) {
      plunge = stepFall(plunge, 0.02);
    }

    expect(isHorizonCrossed(plunge)).toBe(true);
    expect(plunge.tau).toBeLessThan(40);

    let interior = plunge;
    for (let i = 0; i < 2000 && !isSingularityReached(interior); i++) {
      interior = stepFall(interior, 0.01);
    }

    expect(interior.r).toBeCloseTo(SINGULARITY_CUTOFF, 2);
    expect(isSingularityReached(interior)).toBe(true);
    expect(interior.tau).toBeLessThan(50);
  });
});

describe('observer tetrad', () => {
  it('builds an orthonormal boosted player camera frame', () => {
    const state = launchFromLocal({ r: 10, phi: 0.4, betaRadial: -0.2, betaTangential: 0.25 });
    const camera = createPlayerCamera(-1, 0.2);
    const frame = observerFrameFromState(state, camera);

    expect(tetradResidual(frame.tetrad)).toBeLessThan(1e-12);
  });

  it('stays orthonormal after the observer has fallen for a while', () => {
    let state: FallState = launchFromLocal({ r: 7, phi: 0.3, betaRadial: -0.1, betaTangential: 0.35 });
    for (let i = 0; i < 500; i++) state = stepFall(state, 0.01);

    const frame = observerFrameFromState(state, createPlayerCamera(-0.4, -1));

    expect(tetradResidual(frame.tetrad)).toBeLessThan(1e-12);
  });
});

function schwarzschildF(r: number): number {
  return 1 - 1 / Math.max(r, SINGULARITY_CUTOFF);
}
