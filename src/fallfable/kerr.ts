// Kerr metric in ingoing Kerr-Schild Cartesian coordinates with an analytic
// Hamiltonian gradient. These coordinates are horizon-penetrating, so both the
// camera worldline and the past-directed rendering rays stay regular through
// the event horizon.
//
//   g_mn = eta_mn + f l_m l_n,  f = 2 M r^3 / (r^4 + a^2 z^2)
//   l_m = (1, lvec),  l^m = (-1, lvec),  lvec = ((rx+ay)/S, (ry-ax)/S, z/r)
//   S = r^2 + a^2, and r solves r^4 - r^2 (rho^2 - a^2) - a^2 z^2 = 0.
//
// Geodesics follow the super-Hamiltonian H = (1/2) g^mn p_m p_n with
//   H = (1/2)(-p_t^2 + |p|^2) - (1/2) f Q^2,  Q = l^m p_m = -p_t + lvec . pvec
// whose spatial gradient is evaluated in closed form (no finite differences).

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

export interface KerrParams {
  mass: number;
  spin: number;
}

export interface PhaseState {
  position: Vec4;
  momentum: Vec4;
}

export interface Tetrad {
  eTime: Vec4;
  eRight: Vec4;
  eUp: Vec4;
  eForward: Vec4;
}

const EPS = 1e-300;

export function kerrParams(spin: number, mass: number): KerrParams {
  const m = Math.max(mass, 0);
  return { mass: m, spin: Math.max(-m, Math.min(m, spin)) };
}

export function horizonRadius(params: KerrParams): number {
  const { mass, spin } = params;
  return mass + Math.sqrt(Math.max(mass * mass - spin * spin, 0));
}

export function ergosphereEquatorialRadius(params: KerrParams): number {
  return 2 * params.mass;
}

/** Bardeen-Press-Teukolsky ISCO radius for an equatorial circular orbit. */
export function iscoRadius(params: KerrParams, prograde = true): number {
  const { mass, spin } = params;
  if (mass <= 0) return 0;
  const chi = spin / mass;
  const z1 = 1 + Math.cbrt(1 - chi * chi) * (Math.cbrt(1 + chi) + Math.cbrt(1 - chi));
  const z2 = Math.sqrt(3 * chi * chi + z1 * z1);
  const sign = prograde ? -1 : 1;
  return mass * (3 + z2 + sign * Math.sqrt((3 - z1) * (3 + z1 + 2 * z2)));
}

/** Equatorial circular photon orbit radius. */
export function photonOrbitRadius(params: KerrParams, prograde = true): number {
  const { mass, spin } = params;
  if (mass <= 0) return 0;
  const sign = prograde ? -1 : 1;
  return 2 * mass * (1 + Math.cos((2 / 3) * Math.acos(sign * spin / mass)));
}

/** Kerr-Schild radial coordinate r from the Cartesian position. */
export function ksRadius(p: Vec3, params: KerrParams): number {
  const a = params.spin;
  const rho2 = p.x * p.x + p.y * p.y + p.z * p.z;
  if (a === 0) return Math.sqrt(rho2);
  const b = rho2 - a * a;
  const r2 = 0.5 * (b + Math.sqrt(b * b + 4 * a * a * p.z * p.z));
  return Math.sqrt(Math.max(r2, 0));
}

/** Euclidean cylinder radius corresponding to KS radius r in the z=0 plane. */
export function equatorialRho(r: number, params: KerrParams): number {
  return Math.sqrt(r * r + params.spin * params.spin);
}

export function equatorialKsRadius(rho: number, params: KerrParams): number {
  return Math.sqrt(Math.max(rho * rho - params.spin * params.spin, 0));
}

interface Geometry {
  r: number;
  f: number;
  l: Vec3;
  gradR: Vec3;
  gradF: Vec3;
  /** Jacobian rows: gradient of each component of l. */
  jl: [Vec3, Vec3, Vec3];
}

function geometryAt(p: Vec3, params: KerrParams): Geometry {
  const a = params.spin;
  const m = params.mass;
  const rho2 = p.x * p.x + p.y * p.y + p.z * p.z;
  const b = rho2 - a * a;
  const disc = Math.sqrt(b * b + 4 * a * a * p.z * p.z);
  const r2 = Math.max(0.5 * (b + disc), EPS);
  const r = Math.sqrt(r2);
  const s = r2 + a * a;

  // grad r = (r^2 xvec + a^2 z zhat) / (r (2 r^2 - b)); 2r^2 - b = disc.
  const gradDen = Math.max(r * disc, EPS);
  const gradR: Vec3 = {
    x: (r2 * p.x) / gradDen,
    y: (r2 * p.y) / gradDen,
    z: (r2 * p.z + a * a * p.z) / gradDen,
  };

  const d = Math.max(r2 * r2 + a * a * p.z * p.z, EPS);
  const f = (2 * m * r2 * r) / d;
  // grad f = (2M / D^2) [ r^2 (3 a^2 z^2 - r^4) grad r - 2 a^2 z r^3 zhat ].
  const fScale = (2 * m) / (d * d);
  const fr = r2 * (3 * a * a * p.z * p.z - r2 * r2);
  const gradF: Vec3 = {
    x: fScale * fr * gradR.x,
    y: fScale * fr * gradR.y,
    z: fScale * (fr * gradR.z - 2 * a * a * p.z * r2 * r),
  };

  const l: Vec3 = {
    x: (r * p.x + a * p.y) / s,
    y: (r * p.y - a * p.x) / s,
    z: p.z / r,
  };
  // dl_x = (r dx + a dy)/S + dr (x(a^2-r^2) - 2ary)/S^2, and similarly for y, z.
  const s2 = s * s;
  const cx = (p.x * (a * a - r2) - 2 * a * r * p.y) / s2;
  const cy = (p.y * (a * a - r2) + 2 * a * r * p.x) / s2;
  const jl: [Vec3, Vec3, Vec3] = [
    { x: r / s + gradR.x * cx, y: gradR.y * cx + a / s, z: gradR.z * cx },
    { x: gradR.x * cy - a / s, y: r / s + gradR.y * cy, z: gradR.z * cy },
    { x: -(p.z / r2) * gradR.x, y: -(p.z / r2) * gradR.y, z: 1 / r - (p.z / r2) * gradR.z },
  ];

  return { r, f, l, gradR, gradF, jl };
}

export function metricDot(p: Vec3, params: KerrParams, u: Vec4, v: Vec4): number {
  const g = geometryAt(p, params);
  const lu = u.t + g.l.x * u.x + g.l.y * u.y + g.l.z * u.z;
  const lv = v.t + g.l.x * v.x + g.l.y * v.y + g.l.z * v.z;
  return -u.t * v.t + u.x * v.x + u.y * v.y + u.z * v.z + g.f * lu * lv;
}

export function lowerVector(p: Vec3, params: KerrParams, u: Vec4): Vec4 {
  const g = geometryAt(p, params);
  const lu = u.t + g.l.x * u.x + g.l.y * u.y + g.l.z * u.z;
  return {
    t: -u.t + g.f * lu,
    x: u.x + g.f * lu * g.l.x,
    y: u.y + g.f * lu * g.l.y,
    z: u.z + g.f * lu * g.l.z,
  };
}

/** H = (1/2) g^mn p_m p_n for a covariant momentum p. */
export function hamiltonian(p: Vec3, momentum: Vec4, params: KerrParams): number {
  const g = geometryAt(p, params);
  const q = -momentum.t + g.l.x * momentum.x + g.l.y * momentum.y + g.l.z * momentum.z;
  return (
    0.5 * (-momentum.t * momentum.t + momentum.x * momentum.x + momentum.y * momentum.y + momentum.z * momentum.z) -
    0.5 * g.f * q * q
  );
}

export interface PhaseDerivative {
  velocity: Vec4;
  force: Vec3;
}

/** dx/dlambda = dH/dp and dp/dlambda = -dH/dx, both in closed form. */
export function phaseDerivative(p: Vec3, momentum: Vec4, params: KerrParams): PhaseDerivative {
  const g = geometryAt(p, params);
  const q = -momentum.t + g.l.x * momentum.x + g.l.y * momentum.y + g.l.z * momentum.z;

  const velocity: Vec4 = {
    t: -momentum.t + g.f * q,
    x: momentum.x - g.f * q * g.l.x,
    y: momentum.y - g.f * q * g.l.y,
    z: momentum.z - g.f * q * g.l.z,
  };

  // dQ_i = sum_j p_j dl_j/dx_i.
  const dq: Vec3 = {
    x: momentum.x * g.jl[0].x + momentum.y * g.jl[1].x + momentum.z * g.jl[2].x,
    y: momentum.x * g.jl[0].y + momentum.y * g.jl[1].y + momentum.z * g.jl[2].y,
    z: momentum.x * g.jl[0].z + momentum.y * g.jl[1].z + momentum.z * g.jl[2].z,
  };
  const force: Vec3 = {
    x: 0.5 * q * q * g.gradF.x + g.f * q * dq.x,
    y: 0.5 * q * q * g.gradF.y + g.f * q * dq.y,
    z: 0.5 * q * q * g.gradF.z + g.f * q * dq.z,
  };
  return { velocity, force };
}

export function rk4Step(state: PhaseState, params: KerrParams, h: number): PhaseState {
  const d1 = phaseDerivative(spatial(state.position), state.momentum, params);
  const s2 = advance(state, d1, h * 0.5);
  const d2 = phaseDerivative(spatial(s2.position), s2.momentum, params);
  const s3 = advance(state, d2, h * 0.5);
  const d3 = phaseDerivative(spatial(s3.position), s3.momentum, params);
  const s4 = advance(state, d3, h);
  const d4 = phaseDerivative(spatial(s4.position), s4.momentum, params);
  const w = h / 6;
  return {
    position: {
      t: state.position.t + w * (d1.velocity.t + 2 * d2.velocity.t + 2 * d3.velocity.t + d4.velocity.t),
      x: state.position.x + w * (d1.velocity.x + 2 * d2.velocity.x + 2 * d3.velocity.x + d4.velocity.x),
      y: state.position.y + w * (d1.velocity.y + 2 * d2.velocity.y + 2 * d3.velocity.y + d4.velocity.y),
      z: state.position.z + w * (d1.velocity.z + 2 * d2.velocity.z + 2 * d3.velocity.z + d4.velocity.z),
    },
    momentum: {
      t: state.momentum.t,
      x: state.momentum.x + w * (d1.force.x + 2 * d2.force.x + 2 * d3.force.x + d4.force.x),
      y: state.momentum.y + w * (d1.force.y + 2 * d2.force.y + 2 * d3.force.y + d4.force.y),
      z: state.momentum.z + w * (d1.force.z + 2 * d2.force.z + 2 * d3.force.z + d4.force.z),
    },
  };
}

function advance(state: PhaseState, d: PhaseDerivative, h: number): PhaseState {
  return {
    position: {
      t: state.position.t + d.velocity.t * h,
      x: state.position.x + d.velocity.x * h,
      y: state.position.y + d.velocity.y * h,
      z: state.position.z + d.velocity.z * h,
    },
    momentum: {
      t: state.momentum.t,
      x: state.momentum.x + d.force.x * h,
      y: state.momentum.y + d.force.y * h,
      z: state.momentum.z + d.force.z * h,
    },
  };
}

/**
 * The Eulerian observer of the Kerr-Schild time slicing,
 * u = (sqrt(1+f), -f lvec / sqrt(1+f)). Timelike everywhere (1 + f > 0),
 * including inside the ergosphere and the horizon, so it is a valid launch
 * frame at any radius - unlike a static observer.
 */
export function eulerianObserver(p: Vec3, params: KerrParams): Vec4 {
  const g = geometryAt(p, params);
  const root = Math.sqrt(1 + g.f);
  return { t: root, x: (-g.f * g.l.x) / root, y: (-g.f * g.l.y) / root, z: (-g.f * g.l.z) / root };
}

/** Gram-Schmidt an orthonormal frame around a timelike eTime. */
export function buildTetrad(p: Vec3, params: KerrParams, eTime: Vec4, forwardHint: Vec3, upHint: Vec3): Tetrad {
  const time = normalizeTimelike(p, params, eTime);
  const basis: Vec4[] = [time];
  const forward = orthonormalize(p, params, { t: 0, ...forwardHint }, basis);
  basis.push(forward);
  const upCandidate = orthonormalize(p, params, { t: 0, ...upHint }, basis, true);
  const up = upCandidate ?? orthonormalize(p, params, pickFallback(forwardHint), basis);
  basis.push(up);
  // right = forward x up keeps the screen basis right-handed (no mirroring).
  const right = orthonormalize(p, params, { t: 0, ...cross(forwardHint, upHint) }, basis, true) ??
    orthonormalize(p, params, pickFallback(upHint), basis);
  return { eTime: time, eRight: right, eUp: up, eForward: forward };
}

function pickFallback(v: Vec3): Vec4 {
  const candidates: Vec4[] = [
    { t: 0, x: 1, y: 0, z: 0 },
    { t: 0, x: 0, y: 1, z: 0 },
    { t: 0, x: 0, y: 0, z: 1 },
  ];
  let best = candidates[0];
  let bestScore = Infinity;
  for (const c of candidates) {
    const score = Math.abs(c.x * v.x + c.y * v.y + c.z * v.z);
    if (score < bestScore) {
      bestScore = score;
      best = c;
    }
  }
  return best;
}

function normalizeTimelike(p: Vec3, params: KerrParams, u: Vec4): Vec4 {
  const norm = metricDot(p, params, u, u);
  if (!(norm < 0)) throw new Error('eTime must be timelike');
  const inv = 1 / Math.sqrt(-norm);
  const v = { t: u.t * inv, x: u.x * inv, y: u.y * inv, z: u.z * inv };
  return v.t < 0 ? { t: -v.t, x: -v.x, y: -v.y, z: -v.z } : v;
}

function orthonormalize(p: Vec3, params: KerrParams, candidate: Vec4, basis: Vec4[]): Vec4;
function orthonormalize(p: Vec3, params: KerrParams, candidate: Vec4, basis: Vec4[], lenient: boolean): Vec4 | null;
function orthonormalize(p: Vec3, params: KerrParams, candidate: Vec4, basis: Vec4[], lenient = false): Vec4 | null {
  let v = candidate;
  for (const axis of basis) {
    const an = metricDot(p, params, axis, axis);
    const c = metricDot(p, params, v, axis) / an;
    v = { t: v.t - c * axis.t, x: v.x - c * axis.x, y: v.y - c * axis.y, z: v.z - c * axis.z };
  }
  const norm = metricDot(p, params, v, v);
  if (norm <= 1e-12) {
    if (lenient) return null;
    throw new Error('degenerate tetrad axis');
  }
  const inv = 1 / Math.sqrt(norm);
  return { t: v.t * inv, x: v.x * inv, y: v.y * inv, z: v.z * inv };
}

export function tetradResidual(p: Vec3, params: KerrParams, tetrad: Tetrad): number {
  const axes = [tetrad.eTime, tetrad.eRight, tetrad.eUp, tetrad.eForward];
  const expected = [-1, 1, 1, 1];
  let worst = 0;
  for (let i = 0; i < 4; i++) {
    for (let j = i; j < 4; j++) {
      const want = i === j ? expected[i] : 0;
      worst = Math.max(worst, Math.abs(metricDot(p, params, axes[i], axes[j]) - want));
    }
  }
  return worst;
}

/** Boost the local frame's time leg by (betaForward, betaRight, betaUp). */
export function boostedFourVelocity(tetrad: Tetrad, beta: Vec3): Vec4 {
  const b2 = beta.x * beta.x + beta.y * beta.y + beta.z * beta.z;
  const clamped = Math.min(b2, 0.9801);
  const scale = b2 > 0 ? Math.sqrt(clamped / b2) : 0;
  const gamma = 1 / Math.sqrt(1 - clamped);
  return {
    t: gamma * (tetrad.eTime.t + scale * (beta.x * tetrad.eForward.t + beta.y * tetrad.eRight.t + beta.z * tetrad.eUp.t)),
    x: gamma * (tetrad.eTime.x + scale * (beta.x * tetrad.eForward.x + beta.y * tetrad.eRight.x + beta.z * tetrad.eUp.x)),
    y: gamma * (tetrad.eTime.y + scale * (beta.x * tetrad.eForward.y + beta.y * tetrad.eRight.y + beta.z * tetrad.eUp.y)),
    z: gamma * (tetrad.eTime.z + scale * (beta.x * tetrad.eForward.z + beta.y * tetrad.eRight.z + beta.z * tetrad.eUp.z)),
  };
}

/**
 * Four-velocity of an equatorial circular orbit at KS radius r with
 * Omega = +/- sqrt(M) / (r^(3/2) +/- a sqrt(M)).
 */
export function circularOrbitFourVelocity(p: Vec3, params: KerrParams, prograde = true): Vec4 {
  const r = ksRadius(p, params);
  const sm = Math.sqrt(params.mass);
  const sign = prograde ? 1 : -1;
  const omega = (sign * sm) / (Math.pow(r, 1.5) + sign * params.spin * sm);
  const candidate: Vec4 = { t: 1, x: -omega * p.y, y: omega * p.x, z: 0 };
  const norm = metricDot(p, params, candidate, candidate);
  if (!(norm < 0)) throw new Error('no timelike circular orbit at this radius');
  const inv = 1 / Math.sqrt(-norm);
  return { t: candidate.t * inv, x: candidate.x * inv, y: candidate.y * inv, z: candidate.z * inv };
}

export function spatial(position: Vec4): Vec3 {
  return { x: position.x, y: position.y, z: position.z };
}

export function cross(a: Vec3, b: Vec3): Vec3 {
  return { x: a.y * b.z - a.z * b.y, y: a.z * b.x - a.x * b.z, z: a.x * b.y - a.y * b.x };
}

export function normalize3(v: Vec3): Vec3 {
  const len = Math.hypot(v.x, v.y, v.z) || 1;
  return { x: v.x / len, y: v.y / len, z: v.z / len };
}
