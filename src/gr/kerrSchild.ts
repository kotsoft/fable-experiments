export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface Vec4 {
  t: number;
  x: number;
  y: number;
  z: number;
}

export interface KerrSchildParams {
  mass: number;
  spin: number;
}

export type Matrix4 = [
  [number, number, number, number],
  [number, number, number, number],
  [number, number, number, number],
  [number, number, number, number],
];

const EPS = 1e-12;

export function kerrSchildParams(spin = 0, mass = 1): KerrSchildParams {
  const clampedMass = Math.max(mass, 0);
  return {
    mass: clampedMass,
    spin: Math.max(-clampedMass, Math.min(clampedMass, spin)),
  };
}

export function horizonRadius(params: KerrSchildParams): number {
  if (params.mass <= 0) return 0;
  return params.mass + Math.sqrt(Math.max(params.mass * params.mass - params.spin * params.spin, 0));
}

export function kerrSchildRadius(position: Vec3, params: KerrSchildParams): number {
  const a = params.spin;
  const radiusSquared = dot3(position, position);
  if (Math.abs(a) < EPS) return Math.sqrt(radiusSquared);

  const b = radiusSquared - a * a;
  const r2 = 0.5 * (b + Math.sqrt(b * b + 4 * a * a * position.z * position.z));
  return Math.sqrt(Math.max(r2, 0));
}

export function kerrSchildNullSpatial(position: Vec3, params: KerrSchildParams): Vec3 {
  const a = params.spin;
  const r = Math.max(kerrSchildRadius(position, params), EPS);
  const r2 = r * r;
  const denominator = Math.max(r2 + a * a, EPS);
  return {
    x: (r * position.x + a * position.y) / denominator,
    y: (r * position.y - a * position.x) / denominator,
    z: position.z / r,
  };
}

export function kerrSchildScalar(position: Vec3, params: KerrSchildParams): number {
  if (params.mass <= 0) return 0;
  const a = params.spin;
  const r = Math.max(kerrSchildRadius(position, params), EPS);
  const r2 = r * r;
  return (2 * params.mass * r2 * r) / Math.max(r2 * r2 + a * a * position.z * position.z, EPS);
}

export function metricMatrix(position: Vec3, params: KerrSchildParams): Matrix4 {
  const h = kerrSchildScalar(position, params);
  const l = kerrSchildNullSpatial(position, params);
  const cov = [1, l.x, l.y, l.z];
  const eta = [-1, 1, 1, 1];
  return matrixFromKerrSchild(eta, h, cov, 1);
}

export function inverseMetricMatrix(position: Vec3, params: KerrSchildParams): Matrix4 {
  const h = kerrSchildScalar(position, params);
  const l = kerrSchildNullSpatial(position, params);
  const contra = [-1, l.x, l.y, l.z];
  const etaInverse = [-1, 1, 1, 1];
  return matrixFromKerrSchild(etaInverse, h, contra, -1);
}

export function metricDot(position: Vec3, params: KerrSchildParams, a: Vec4, b: Vec4): number {
  const h = kerrSchildScalar(position, params);
  const l = kerrSchildNullSpatial(position, params);
  const la = a.t + l.x * a.x + l.y * a.y + l.z * a.z;
  const lb = b.t + l.x * b.x + l.y * b.y + l.z * b.z;
  return minkowskiDot(a, b) + h * la * lb;
}

export function inverseMetricDot(position: Vec3, params: KerrSchildParams, a: Vec4, b: Vec4): number {
  const h = kerrSchildScalar(position, params);
  const l = kerrSchildNullSpatial(position, params);
  const la = -a.t + l.x * a.x + l.y * a.y + l.z * a.z;
  const lb = -b.t + l.x * b.x + l.y * b.y + l.z * b.z;
  return minkowskiDot(a, b) - h * la * lb;
}

export function lowerVector(position: Vec3, params: KerrSchildParams, vector: Vec4): Vec4 {
  return multiplyMatrixVector(metricMatrix(position, params), vector);
}

export function raiseCovector(position: Vec3, params: KerrSchildParams, covector: Vec4): Vec4 {
  return multiplyMatrixVector(inverseMetricMatrix(position, params), covector);
}

export function minkowskiDot(a: Vec4, b: Vec4): number {
  return -a.t * b.t + a.x * b.x + a.y * b.y + a.z * b.z;
}

export function dot3(a: Vec3, b: Vec3): number {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

export function multiplyMatrices(a: Matrix4, b: Matrix4): Matrix4 {
  return [0, 1, 2, 3].map((row) =>
    [0, 1, 2, 3].map((col) =>
      a[row][0] * b[0][col] +
      a[row][1] * b[1][col] +
      a[row][2] * b[2][col] +
      a[row][3] * b[3][col],
    ),
  ) as Matrix4;
}

function multiplyMatrixVector(matrix: Matrix4, vector: Vec4): Vec4 {
  const v = [vector.t, vector.x, vector.y, vector.z];
  const out = matrix.map((row) => row[0] * v[0] + row[1] * v[1] + row[2] * v[2] + row[3] * v[3]);
  return { t: out[0], x: out[1], y: out[2], z: out[3] };
}

function matrixFromKerrSchild(
  diagonal: number[],
  scalar: number,
  nullVector: number[],
  sign: 1 | -1,
): Matrix4 {
  return [0, 1, 2, 3].map((row) =>
    [0, 1, 2, 3].map((col) => {
      const base = row === col ? diagonal[row] : 0;
      return base + sign * scalar * nullVector[row] * nullVector[col];
    }),
  ) as Matrix4;
}
