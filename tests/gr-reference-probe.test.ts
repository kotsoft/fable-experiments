import { describe, expect, it } from 'vitest';
import { kerrSchildParams } from '../src/gr/kerrSchild';
import { renderProbeGrid } from '../src/gr/referenceProbe';
import { buildObserverTetrad, staticObserverFourVelocity } from '../src/gr/tetrad';

describe('CPU reference ray probe', () => {
  it('renders deterministic structured diagnostics for a Schwarzschild view', () => {
    const params = kerrSchildParams(0, 1);
    const position = { x: 12, y: 0, z: 0 };
    const tetrad = buildObserverTetrad(position, params, staticObserverFourVelocity(position, params));
    const grid = renderProbeGrid(
      params,
      { position, tetrad, verticalFovRadians: 1.1 },
      9,
      5,
      {
        stepSize: 0.05,
        maxSteps: 2200,
        escapeRadius: 32,
        singularityRadius: 0.2,
      },
    );

    const statuses = new Set(grid.rays.map((ray) => ray.status));

    expect(grid.width).toBe(9);
    expect(grid.height).toBe(5);
    expect(grid.rays).toHaveLength(45);
    expect(statuses.has('horizon')).toBe(true);
    expect(statuses.has('escaped')).toBe(true);
    expect(Math.max(...grid.rays.map((ray) => ray.maxHamiltonianDrift))).toBeLessThan(2e-6);
    expect(grid.rays.every((ray) => ray.steps >= 0 && Number.isFinite(ray.finalRadius))).toBe(true);
  });

  it('renders finite Kerr diagnostics suitable for future GPU readback comparisons', () => {
    const params = kerrSchildParams(0.75, 1);
    const position = { x: 10, y: -2, z: 1.5 };
    const tetrad = buildObserverTetrad(position, params, staticObserverFourVelocity(position, params));
    const grid = renderProbeGrid(
      params,
      { position, tetrad, verticalFovRadians: 0.9 },
      6,
      4,
      {
        stepSize: 0.02,
        maxSteps: 3600,
        escapeRadius: 34,
        singularityRadius: 0.2,
      },
    );

    expect(grid.rays).toHaveLength(24);
    expect(grid.rays.every((ray) => ray.color.every(Number.isFinite))).toBe(true);
    expect(Math.max(...grid.rays.map((ray) => ray.maxHamiltonianDrift))).toBeLessThan(2e-6);
  });
});
