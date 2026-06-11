// Validates the Kerr geodesic formulas used by the renderer and tutorial
// against known exact results (M = 1, spin a):
//   - prograde/retrograde photon orbit radii  r_ph = 2(1 + cos(2/3 acos(∓a)))
//   - critical impact parameters from Bardeen's xi(r_ph)
//   - frame-dragging sense (a > 0 must drag counterclockwise, +phi)
//   - ISCO radius formula
// Integrator: Hamiltonian in Cartesian Kerr-Schild coordinates,
//   H = 1/2 [ -pt^2 + |p|^2 - f (l^mu p_mu)^2 ],  f = 2Mr^3/(r^4 + a^2 z^2)
// with numerical dH/dx (central differences) and RK4 — the same scheme the
// fragment shader uses.

const a = 0.95;

function hamiltonian(x, y, z, px, py, pz, pt) {
  const R2 = x * x + y * y + z * z;
  const b = R2 - a * a;
  const r2 = 0.5 * (b + Math.sqrt(b * b + 4 * a * a * z * z));
  const r = Math.sqrt(r2);
  const f = (2 * r2 * r) / (r2 * r2 + a * a * z * z);
  const den = r2 + a * a;
  const lx = (r * x + a * y) / den;
  const ly = (r * y - a * x) / den;
  const lz = z / r;
  const L = -pt + lx * px + ly * py + lz * pz; // l^mu p_mu, l^t = -1
  return 0.5 * (-pt * pt + px * px + py * py + pz * pz - f * L * L);
}

// state s = [x,y,z,px,py,pz]; pt is a conserved parameter
function deriv(s, pt) {
  const [x, y, z, px, py, pz] = s;
  const R2 = x * x + y * y + z * z;
  const b = R2 - a * a;
  const r2 = 0.5 * (b + Math.sqrt(b * b + 4 * a * a * z * z));
  const r = Math.sqrt(r2);
  const f = (2 * r2 * r) / (r2 * r2 + a * a * z * z);
  const den = r2 + a * a;
  const lx = (r * x + a * y) / den;
  const ly = (r * y - a * x) / den;
  const lz = z / r;
  const L = -pt + lx * px + ly * py + lz * pz;
  // dx/dlambda = dH/dp (analytic)
  const dx = px - f * L * lx;
  const dy = py - f * L * ly;
  const dz = pz - f * L * lz;
  // dp/dlambda = -dH/dx (numerical)
  const eps = 1e-6 * Math.max(1, Math.sqrt(R2));
  const g = (i) => {
    const q = [x, y, z];
    q[i] += eps;
    const hp = hamiltonian(q[0], q[1], q[2], px, py, pz, pt);
    q[i] -= 2 * eps;
    const hm = hamiltonian(q[0], q[1], q[2], px, py, pz, pt);
    return -(hp - hm) / (2 * eps);
  };
  return [dx, dy, dz, g(0), g(1), g(2)];
}

function rk4(s, pt, dl) {
  const k1 = deriv(s, pt);
  const s2 = s.map((v, i) => v + 0.5 * dl * k1[i]);
  const k2 = deriv(s2, pt);
  const s3 = s.map((v, i) => v + 0.5 * dl * k2[i]);
  const k3 = deriv(s3, pt);
  const s4 = s.map((v, i) => v + dl * k3[i]);
  const k4 = deriv(s4, pt);
  return s.map((v, i) => v + (dl / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
}

const rplus = 1 + Math.sqrt(1 - a * a);

// forward photon (pt = -1) fired from (x0, b, 0) in direction (-1, 0, 0)
function trace(b) {
  const x0 = 60;
  // solve null condition for momentum scale s: p = s * dir
  const R2 = x0 * x0 + b * b;
  const bb = R2 - a * a;
  const r2 = 0.5 * (bb + Math.sqrt(bb * bb));
  const r = Math.sqrt(r2);
  const f = (2 * r2 * r) / (r2 * r2);
  const den = r2 + a * a;
  const c = -(r * x0 + a * b) / den; // l . dir with dir = (-1,0,0)
  const A = 1 - f * c * c, B = -2 * f * c, C = -(1 + f);
  const s0 = (-B + Math.sqrt(B * B - 4 * A * C)) / (2 * A);
  let s = [x0, b, 0, -s0, 0, 0];
  let phiWind = 0;
  let prevx = x0, prevy = b;
  for (let i = 0; i < 200000; i++) {
    const R = Math.hypot(s[0], s[1], s[2]);
    const rr = Math.sqrt(Math.max(R * R - a * a, 1e-9)); // equatorial BL r
    if (rr < rplus * 1.001) return { captured: true, phiWind };
    if (R > 70 && s[0] * s[3] + s[1] * s[4] + s[2] * s[5] > 0)
      return { captured: false, phiWind };
    const dl = Math.min(Math.max(0.015 * rr, 0.002), 0.5);
    s = rk4(s, -1, dl);
    phiWind += Math.atan2(prevx * s[1] - prevy * s[0], prevx * s[0] + prevy * s[1]);
    prevx = s[0]; prevy = s[1];
  }
  return { captured: true, phiWind };
}

function bisectCritical(lo, hi) {
  // lo captured, hi escaped (by |b|)
  for (let i = 0; i < 40; i++) {
    const mid = (lo + hi) / 2;
    if (trace(mid).captured === trace(lo).captured) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

// --- theory ---
const rphPro = 2 * (1 + Math.cos((2 / 3) * Math.acos(-a)));
const rphRet = 2 * (1 + Math.cos((2 / 3) * Math.acos(a)));
const xi = (r) => (r * r * (3 - r) - a * a * (1 + r)) / (a * (r - 1));
const Z1 = 1 + Math.cbrt(1 - a * a) * (Math.cbrt(1 + a) + Math.cbrt(1 - a));
const Z2 = Math.sqrt(3 * a * a + Z1 * Z1);
const iscoPro = 3 + Z2 - Math.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2));

console.log(`a = ${a},  r+ = ${rplus.toFixed(4)}`);
console.log(`theory: r_ph prograde ${rphPro.toFixed(4)}, retrograde ${rphRet.toFixed(4)}`);
console.log(`theory: b_crit prograde ${xi(rphPro).toFixed(4)}, retrograde ${xi(rphRet).toFixed(4)}`);
console.log(`theory: ISCO prograde ${iscoPro.toFixed(4)}`);

// frame-dragging sense: b = 0 photon (zero angular momentum) must wind in +phi
const drag = trace(1e-9);
console.log(`\nzero-L photon: captured=${drag.captured}, phi wind=${drag.phiWind.toFixed(4)} (must be > 0 for CCW drag)`);

// critical b on the +y side (prograde, since L_z = +b here)
const bPro = bisectCritical(2.0, 3.5);
console.log(`measured b_crit, +b side: ${bPro.toFixed(4)} (expect ~${xi(rphPro).toFixed(4)})`);

// critical b on the -y side (retrograde)
const traceNeg = (b) => trace(-b);
let lo = 6.0, hi = 8.0; // lo captured, hi escaped
for (let i = 0; i < 40; i++) {
  const mid = (lo + hi) / 2;
  if (traceNeg(mid).captured) lo = mid;
  else hi = mid;
}
console.log(`measured b_crit, -b side: ${((lo + hi) / 2).toFixed(4)} (expect ~${(-xi(rphRet)).toFixed(4)})`);
