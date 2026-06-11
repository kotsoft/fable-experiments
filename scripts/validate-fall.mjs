import assert from 'node:assert/strict';

const SINGULARITY_CUTOFF = 0.08;

function f(r) {
  return 1 - 1 / Math.max(r, SINGULARITY_CUTOFF);
}

function launchFromLocal({ r, phi, betaRadial, betaTangential }) {
  const beta2 = Math.min(betaRadial * betaRadial + betaTangential * betaTangential, 0.85 * 0.85);
  const gamma = 1 / Math.sqrt(1 - beta2);
  return {
    r,
    phi,
    tau: 0,
    t: 0,
    energy: gamma * Math.sqrt(f(r)),
    angularMomentum: r * gamma * betaTangential,
    radialSign: Math.abs(betaRadial) < 1e-6 ? -1 : Math.sign(betaRadial),
  };
}

function radialPotential(s) {
  return s.energy * s.energy - f(s.r) * (1 + s.angularMomentum * s.angularMomentum / (s.r * s.r));
}

function deriv(s) {
  let radialSq = radialPotential(s);
  let sign = s.radialSign;
  if (radialSq < 1e-8 && sign > 0) {
    sign = -1;
    radialSq = 0;
  } else if (radialSq < 1e-10 && sign < 0) {
    radialSq = 1e-10;
  }
  return {
    r: sign * Math.sqrt(Math.max(radialSq, 0)),
    phi: s.angularMomentum / (s.r * s.r),
    t: s.r > 1 ? s.energy / Math.max(f(s.r), 1e-8) : 0,
  };
}

function offset(s, k, h) {
  return { ...s, r: Math.max(s.r + k.r * h, SINGULARITY_CUTOFF), phi: s.phi + k.phi * h, t: s.t + k.t * h, tau: s.tau + h };
}

function rk4Step(s, h) {
  const k1 = deriv(s);
  const k2 = deriv(offset(s, k1, h * 0.5));
  const k3 = deriv(offset(s, k2, h * 0.5));
  const k4 = deriv(offset(s, k3, h));
  const r = s.r + h / 6 * (k1.r + 2 * k2.r + 2 * k3.r + k4.r);
  const radialSign = radialPotential({ ...s, r: Math.max(r, SINGULARITY_CUTOFF) }) < 1e-8 && s.radialSign > 0 ? -1 : s.radialSign;
  return {
    ...s,
    r: Math.max(r, SINGULARITY_CUTOFF),
    phi: s.phi + h / 6 * (k1.phi + 2 * k2.phi + 2 * k3.phi + k4.phi),
    t: s.t + h / 6 * (k1.t + 2 * k2.t + 2 * k3.t + k4.t),
    tau: s.tau + h,
    radialSign,
  };
}

function stepFall(s, dTau) {
  let next = { ...s };
  const steps = Math.max(1, Math.ceil(Math.abs(dTau) / 0.006));
  const h = dTau / steps;
  for (let i = 0; i < steps; i++) {
    if (next.r <= SINGULARITY_CUTOFF) return { ...next, r: SINGULARITY_CUTOFF };
    next = rk4Step(next, h);
  }
  return next;
}

function localVelocity(s) {
  const radialSq = radialPotential(s);
  const dr = s.radialSign * Math.sqrt(Math.max(radialSq, 0));
  return {
    betaRadial: dr / s.energy,
    betaTangential: (s.angularMomentum / s.r) * Math.sqrt(f(s.r)) / s.energy,
  };
}

function boostAxis(axis, beta, gamma, b2) {
  if (b2 < 1e-10) return { t: 0, ...axis };
  const bd = beta.x * axis.x + beta.y * axis.y + beta.z * axis.z;
  const scale = ((gamma - 1) * bd) / b2;
  return {
    t: gamma * bd,
    x: axis.x + scale * beta.x,
    y: axis.y + scale * beta.y,
    z: axis.z + scale * beta.z,
  };
}

function tetradFromBeta(beta) {
  const b2 = beta.x * beta.x + beta.y * beta.y + beta.z * beta.z;
  const gamma = 1 / Math.sqrt(Math.max(1 - b2, 1e-5));
  return [
    { t: gamma, x: gamma * beta.x, y: gamma * beta.y, z: gamma * beta.z },
    boostAxis({ x: 0, y: 0, z: 1 }, beta, gamma, b2),
    boostAxis({ x: 0, y: 1, z: 0 }, beta, gamma, b2),
    boostAxis({ x: -1, y: 0, z: 0 }, beta, gamma, b2),
  ];
}

function minkowskiDot(a, b) {
  return -a.t * b.t + a.x * b.x + a.y * b.y + a.z * b.z;
}

function tetradResidual(axes) {
  const expected = [-1, 1, 1, 1];
  let residual = 0;
  for (let i = 0; i < axes.length; i++) {
    residual = Math.max(residual, Math.abs(minkowskiDot(axes[i], axes[i]) - expected[i]));
    for (let j = i + 1; j < axes.length; j++) {
      residual = Math.max(residual, Math.abs(minkowskiDot(axes[i], axes[j])));
    }
  }
  return residual;
}

const launched = launchFromLocal({ r: 10, phi: 0, betaRadial: -0.2, betaTangential: 0.25 });
let s = launched;
for (let i = 0; i < 400; i++) s = stepFall(s, 0.02);
assert.equal(s.energy, launched.energy);
assert.equal(s.angularMomentum, launched.angularMomentum);
const v = localVelocity(s);
const residual = Math.abs(v.betaRadial * v.betaRadial + v.betaTangential * v.betaTangential - (1 - f(s.r) / (s.energy * s.energy)));
assert.ok(residual < 1e-4, `local velocity invariant residual ${residual}`);
const tetrad = tetradFromBeta({ x: v.betaRadial, y: 0, z: v.betaTangential });
assert.ok(tetradResidual(tetrad) < 1e-12, `observer tetrad should be orthonormal`);

const rest = launchFromLocal({ r: 8, phi: 0, betaRadial: 0, betaTangential: 0 });
const restNext = stepFall(rest, 0.5);
assert.ok(restNext.r < rest.r, 'local-rest release should fall inward');

let near = launchFromLocal({ r: 1.2, phi: 0, betaRadial: 0, betaTangential: 0 });
near = stepFall(near, 0.2);
assert.ok(near.t > near.tau, 'distant time should advance faster than proper time near the horizon');

let plunge = launchFromLocal({ r: 4, phi: 0, betaRadial: 0, betaTangential: 0 });
for (let i = 0; i < 2000 && plunge.r > 1.01; i++) plunge = stepFall(plunge, 0.02);
assert.ok(plunge.r <= 1.01, 'radial plunge should cross the horizon');
assert.ok(plunge.tau < 40, 'horizon crossing should happen in finite proper time');

let interior = plunge;
for (let i = 0; i < 2000 && interior.r > SINGULARITY_CUTOFF + 0.002; i++) interior = stepFall(interior, 0.01);
assert.ok(interior.r <= SINGULARITY_CUTOFF + 0.002, 'interior plunge should reach the singularity cutoff');
assert.ok(interior.tau < 50, 'singularity cutoff should be reached in finite proper time');

console.log('fall validation passed');
