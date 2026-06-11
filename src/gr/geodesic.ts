import {
  horizonRadius,
  inverseMetricDot,
  inverseMetricMatrix,
  kerrSchildRadius,
  multiplyMatrices,
  type KerrSchildParams,
  type Matrix4,
  type Vec3,
  type Vec4,
} from './kerrSchild';

export interface GeodesicState {
  position: Vec4;
  momentum: Vec4;
}

export interface TraceOptions {
  stepSize: number;
  maxSteps: number;
  escapeRadius: number;
  singularityRadius: number;
}

export interface TraceResult {
  state: GeodesicState;
  steps: number;
  status: 'escaped' | 'horizon' | 'singularity' | 'max-steps';
  maxHamiltonianDrift: number;
}

export function hamiltonian(state: GeodesicState, params: KerrSchildParams): number {
  return 0.5 * inverseMetricDot(spatialPosition(state.position), params, state.momentum, state.momentum);
}

export function coordinateVelocity(position: Vec4, momentum: Vec4, params: KerrSchildParams): Vec4 {
  const inverse = inverseMetricMatrix(spatialPosition(position), params);
  return multiplyMatrixVector(inverse, momentum);
}

export function nullCovectorFromDirection(
  position: Vec4,
  direction: Vec3,
  params: KerrSchildParams,
  energy = 1,
): Vec4 {
  const n = normalize3(direction);
  const inverse = inverseMetricMatrix(spatialPosition(position), params);
  const pt = -Math.abs(energy);
  const a =
    inverse[1][1] * n.x * n.x +
    inverse[2][2] * n.y * n.y +
    inverse[3][3] * n.z * n.z +
    2 * inverse[1][2] * n.x * n.y +
    2 * inverse[1][3] * n.x * n.z +
    2 * inverse[2][3] * n.y * n.z;
  const b = 2 * pt * (inverse[0][1] * n.x + inverse[0][2] * n.y + inverse[0][3] * n.z);
  const c = inverse[0][0] * pt * pt;
  const disc = Math.max(b * b - 4 * a * c, 0);
  const rootA = (-b + Math.sqrt(disc)) / (2 * a);
  const rootB = (-b - Math.sqrt(disc)) / (2 * a);
  const scale = rootA > 0 ? rootA : rootB;
  return { t: pt, x: scale * n.x, y: scale * n.y, z: scale * n.z };
}

export function stepNullGeodesic(state: GeodesicState, params: KerrSchildParams, step: number): GeodesicState {
  const k1 = derivative(state, params);
  const k2 = derivative(offsetState(state, k1, step * 0.5), params);
  const k3 = derivative(offsetState(state, k2, step * 0.5), params);
  const k4 = derivative(offsetState(state, k3, step), params);
  return combineState(state, [k1, k2, k3, k4], step);
}

export function traceNullGeodesic(
  initial: GeodesicState,
  params: KerrSchildParams,
  options: TraceOptions,
): TraceResult {
  let state = initial;
  const h0 = hamiltonian(initial, params);
  let maxHamiltonianDrift = 0;
  const horizon = horizonRadius(params);

  for (let steps = 0; steps < options.maxSteps; steps++) {
    const radius = kerrSchildRadius(spatialPosition(state.position), params);
    if (radius <= options.singularityRadius) {
      return { state, steps, status: 'singularity', maxHamiltonianDrift };
    }
    if (horizon > 0 && radius <= horizon) {
      return { state, steps, status: 'horizon', maxHamiltonianDrift };
    }
    if (radius >= options.escapeRadius && radialCoordinateSpeed(state, params) > 0) {
      return { state, steps, status: 'escaped', maxHamiltonianDrift };
    }

    state = stepNullGeodesic(state, params, options.stepSize);
    maxHamiltonianDrift = Math.max(maxHamiltonianDrift, Math.abs(hamiltonian(state, params) - h0));
  }

  return { state, steps: options.maxSteps, status: 'max-steps', maxHamiltonianDrift };
}

function derivative(state: GeodesicState, params: KerrSchildParams): GeodesicState {
  const velocity = coordinateVelocity(state.position, state.momentum, params);
  const gradient = hamiltonianGradient(state, params);
  return {
    position: velocity,
    momentum: { t: 0, x: -gradient.x, y: -gradient.y, z: -gradient.z },
  };
}

function hamiltonianGradient(state: GeodesicState, params: KerrSchildParams): Vec3 {
  const p = spatialPosition(state.position);
  const radius = Math.max(kerrSchildRadius(p, params), 1);
  const eps = 1e-5 * radius;
  return {
    x: centralDifference(state, params, { x: eps, y: 0, z: 0 }),
    y: centralDifference(state, params, { x: 0, y: eps, z: 0 }),
    z: centralDifference(state, params, { x: 0, y: 0, z: eps }),
  };
}

function centralDifference(state: GeodesicState, params: KerrSchildParams, delta: Vec3): number {
  const plus = {
    ...state,
    position: {
      ...state.position,
      x: state.position.x + delta.x,
      y: state.position.y + delta.y,
      z: state.position.z + delta.z,
    },
  };
  const minus = {
    ...state,
    position: {
      ...state.position,
      x: state.position.x - delta.x,
      y: state.position.y - delta.y,
      z: state.position.z - delta.z,
    },
  };
  const width = Math.hypot(delta.x, delta.y, delta.z) * 2;
  return (hamiltonian(plus, params) - hamiltonian(minus, params)) / width;
}

function radialCoordinateSpeed(state: GeodesicState, params: KerrSchildParams): number {
  const p = spatialPosition(state.position);
  const velocity = coordinateVelocity(state.position, state.momentum, params);
  const radius = Math.max(kerrSchildRadius(p, params), 1e-8);
  return (p.x * velocity.x + p.y * velocity.y + p.z * velocity.z) / radius;
}

function offsetState(state: GeodesicState, derivativeState: GeodesicState, step: number): GeodesicState {
  return {
    position: addScaledVec4(state.position, derivativeState.position, step),
    momentum: addScaledVec4(state.momentum, derivativeState.momentum, step),
  };
}

function combineState(
  state: GeodesicState,
  derivatives: [GeodesicState, GeodesicState, GeodesicState, GeodesicState],
  step: number,
): GeodesicState {
  const [k1, k2, k3, k4] = derivatives;
  return {
    position: combineVec4(state.position, k1.position, k2.position, k3.position, k4.position, step),
    momentum: combineVec4(state.momentum, k1.momentum, k2.momentum, k3.momentum, k4.momentum, step),
  };
}

function combineVec4(base: Vec4, k1: Vec4, k2: Vec4, k3: Vec4, k4: Vec4, step: number): Vec4 {
  return {
    t: base.t + step / 6 * (k1.t + 2 * k2.t + 2 * k3.t + k4.t),
    x: base.x + step / 6 * (k1.x + 2 * k2.x + 2 * k3.x + k4.x),
    y: base.y + step / 6 * (k1.y + 2 * k2.y + 2 * k3.y + k4.y),
    z: base.z + step / 6 * (k1.z + 2 * k2.z + 2 * k3.z + k4.z),
  };
}

function addScaledVec4(a: Vec4, b: Vec4, scale: number): Vec4 {
  return {
    t: a.t + b.t * scale,
    x: a.x + b.x * scale,
    y: a.y + b.y * scale,
    z: a.z + b.z * scale,
  };
}

function multiplyMatrixVector(matrix: Matrix4, vector: Vec4): Vec4 {
  const v = [vector.t, vector.x, vector.y, vector.z];
  const out = multiplyMatrices(matrix, [
    [v[0], 0, 0, 0],
    [v[1], 0, 0, 0],
    [v[2], 0, 0, 0],
    [v[3], 0, 0, 0],
  ]);
  return { t: out[0][0], x: out[1][0], y: out[2][0], z: out[3][0] };
}

function spatialPosition(position: Vec4): Vec3 {
  return { x: position.x, y: position.y, z: position.z };
}

function normalize3(v: Vec3): Vec3 {
  const length = Math.hypot(v.x, v.y, v.z) || 1;
  return { x: v.x / length, y: v.y / length, z: v.z / length };
}
