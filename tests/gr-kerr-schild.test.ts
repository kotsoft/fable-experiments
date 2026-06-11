import { describe, expect, it } from 'vitest';
import {
  dot3,
  horizonRadius,
  inverseMetricMatrix,
  kerrSchildNullSpatial,
  kerrSchildParams,
  kerrSchildRadius,
  kerrSchildScalar,
  lowerVector,
  metricDot,
  metricMatrix,
  multiplyMatrices,
  raiseCovector,
  type Matrix4,
  type Vec3,
} from '../src/gr/kerrSchild';

describe('Kerr-Schild metric primitives', () => {
  it('reduces to flat Minkowski space when mass is zero', () => {
    const params = kerrSchildParams(0, 0);
    const position = { x: 3, y: -2, z: 4 };
    const a = { t: 1.5, x: 2, y: -0.25, z: 0.75 };
    const b = { t: -0.5, x: 0.2, y: 1.25, z: -3 };

    expect(kerrSchildScalar(position, params)).toBe(0);
    expect(metricDot(position, params, a, b)).toBeCloseTo(0.75 + 0.4 - 0.3125 - 2.25, 12);
    expectMatrixClose(metricMatrix(position, params), [
      [-1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ]);
  });

  it('matches Schwarzschild Kerr-Schild helpers when spin is zero', () => {
    const params = kerrSchildParams(0, 1);
    const position = { x: 3, y: 4, z: 12 };
    const radius = 13;

    expect(kerrSchildRadius(position, params)).toBeCloseTo(radius, 12);
    expect(horizonRadius(params)).toBeCloseTo(2, 12);
    expect(kerrSchildScalar(position, params)).toBeCloseTo(2 / radius, 12);
  });

  it('builds a Euclidean-unit Kerr-Schild spatial null direction', () => {
    const params = kerrSchildParams(0.8, 1);
    const positions: Vec3[] = [
      { x: 4, y: 1, z: 2 },
      { x: -2, y: 5, z: -1 },
      { x: 0.4, y: -0.8, z: 3 },
    ];

    for (const position of positions) {
      const l = kerrSchildNullSpatial(position, params);
      expect(dot3(l, l)).toBeCloseTo(1, 12);
    }
  });

  it('keeps the metric and inverse metric mutually consistent', () => {
    const params = kerrSchildParams(0.74, 1);
    const positions: Vec3[] = [
      { x: 5, y: 2, z: 1 },
      { x: -3, y: 4, z: 0.6 },
      { x: 1.8, y: -2.2, z: 3.5 },
    ];

    for (const position of positions) {
      const product = multiplyMatrices(metricMatrix(position, params), inverseMetricMatrix(position, params));
      expectMatrixClose(product, identity(), 10);
    }
  });

  it('round-trips vectors through lowering and raising', () => {
    const params = kerrSchildParams(0.5, 1);
    const position = { x: 4, y: -1, z: 2 };
    const vector = { t: 1.2, x: -0.4, y: 0.9, z: 0.25 };
    const raised = raiseCovector(position, params, lowerVector(position, params, vector));

    expect(raised.t).toBeCloseTo(vector.t, 10);
    expect(raised.x).toBeCloseTo(vector.x, 10);
    expect(raised.y).toBeCloseTo(vector.y, 10);
    expect(raised.z).toBeCloseTo(vector.z, 10);
  });
});

function identity(): Matrix4 {
  return [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ];
}

function expectMatrixClose(actual: Matrix4, expected: Matrix4, precision = 12): void {
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 4; col++) {
      expect(actual[row][col]).toBeCloseTo(expected[row][col], precision);
    }
  }
}
