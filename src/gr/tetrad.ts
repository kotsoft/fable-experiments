import {
  dot3,
  lowerVector,
  metricDot,
  metricMatrix,
  type KerrSchildParams,
  type Vec3,
  type Vec4,
} from './kerrSchild';

export interface GrTetrad {
  eTime: Vec4;
  eRight: Vec4;
  eUp: Vec4;
  eForward: Vec4;
}

export interface SpatialAxisHints {
  right?: Vec4;
  up?: Vec4;
  forward?: Vec4;
}

export function staticObserverFourVelocity(position: Vec3, params: KerrSchildParams): Vec4 {
  const g = metricMatrix(position, params);
  const gtt = g[0][0];
  if (gtt >= 0) {
    throw new Error('Static observer is not timelike at this position');
  }
  return { t: 1 / Math.sqrt(-gtt), x: 0, y: 0, z: 0 };
}

export function buildObserverTetrad(
  position: Vec3,
  params: KerrSchildParams,
  fourVelocity: Vec4,
  hints: SpatialAxisHints = {},
): GrTetrad {
  const eTime = normalizeTimelike(position, params, fourVelocity);
  const fallbackForward = radialInwardHint(position);
  const fallbackRight = normalizeSpatialHint({ t: 0, x: fallbackForward.z, y: 0, z: -fallbackForward.x });
  const axes: Vec4[] = [eTime];

  const eForward = orthonormalizeSpacelike(
    position,
    params,
    hints.forward ?? fallbackForward,
    axes,
  );
  axes.push(eForward);

  const eRight = orthonormalizeSpacelike(
    position,
    params,
    hints.right ?? fallbackRight,
    axes,
  );
  axes.push(eRight);

  const eUp = orthonormalizeSpacelike(position, params, hints.up ?? { t: 0, x: 0, y: 1, z: 0 }, axes);

  return { eTime, eRight, eUp, eForward };
}

export function launchPhotonFromTetrad(
  position: Vec3,
  params: KerrSchildParams,
  tetrad: GrTetrad,
  localDirection: Vec3,
  energy = 1,
): Vec4 {
  const n = normalize3(localDirection);
  const contravariant = scaleVec4(
    addVec4(
      tetrad.eTime,
      addVec4(
        scaleVec4(tetrad.eRight, n.x),
        addVec4(scaleVec4(tetrad.eUp, n.y), scaleVec4(tetrad.eForward, n.z)),
      ),
    ),
    Math.abs(energy),
  );
  return lowerVector(position, params, contravariant);
}

export function tetradResidual(position: Vec3, params: KerrSchildParams, tetrad: GrTetrad): number {
  const axes = [tetrad.eTime, tetrad.eRight, tetrad.eUp, tetrad.eForward];
  const expected = [-1, 1, 1, 1];
  let residual = 0;
  for (let i = 0; i < axes.length; i++) {
    residual = Math.max(residual, Math.abs(metricDot(position, params, axes[i], axes[i]) - expected[i]));
    for (let j = i + 1; j < axes.length; j++) {
      residual = Math.max(residual, Math.abs(metricDot(position, params, axes[i], axes[j])));
    }
  }
  return residual;
}

function normalizeTimelike(position: Vec3, params: KerrSchildParams, vector: Vec4): Vec4 {
  const norm = metricDot(position, params, vector, vector);
  if (norm >= 0) throw new Error('Observer four-velocity must be timelike');
  const normalized = scaleVec4(vector, 1 / Math.sqrt(-norm));
  return normalized.t < 0 ? scaleVec4(normalized, -1) : normalized;
}

function orthonormalizeSpacelike(
  position: Vec3,
  params: KerrSchildParams,
  candidate: Vec4,
  basis: Vec4[],
): Vec4 {
  let v = candidate;
  for (const axis of basis) {
    const axisNorm = metricDot(position, params, axis, axis);
    v = addVec4(v, scaleVec4(axis, -metricDot(position, params, v, axis) / axisNorm));
  }

  const norm = metricDot(position, params, v, v);
  if (norm <= 1e-10) {
    throw new Error('Could not build spacelike tetrad axis from nearly degenerate hint');
  }
  return scaleVec4(v, 1 / Math.sqrt(norm));
}

function radialInwardHint(position: Vec3): Vec4 {
  const spatial = normalize3({ x: -position.x, y: -position.y, z: -position.z });
  return { t: 0, x: spatial.x, y: spatial.y, z: spatial.z };
}

function normalizeSpatialHint(vector: Vec4): Vec4 {
  const length = Math.hypot(vector.x, vector.y, vector.z) || 1;
  return { t: vector.t / length, x: vector.x / length, y: vector.y / length, z: vector.z / length };
}

function normalize3(v: Vec3): Vec3 {
  const length = Math.sqrt(Math.max(dot3(v, v), 1e-20));
  return { x: v.x / length, y: v.y / length, z: v.z / length };
}

function addVec4(a: Vec4, b: Vec4): Vec4 {
  return { t: a.t + b.t, x: a.x + b.x, y: a.y + b.y, z: a.z + b.z };
}

function scaleVec4(v: Vec4, scale: number): Vec4 {
  return { t: v.t * scale, x: v.x * scale, y: v.y * scale, z: v.z * scale };
}
