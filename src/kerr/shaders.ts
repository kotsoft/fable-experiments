// GLSL sources for the Kerr (rotating) black hole renderer.
//
// Physics notes:
// - Units: G = c = M = 1. Spin parameter a = J/M ∈ [0, 1).
//   Horizon r+ = 1 + sqrt(1 - a²); prograde ISCO from Bardeen's formula
//   (computed in TS, passed as a uniform).
// - Unlike Schwarzschild, Kerr photons admit no equivalent flat-space force
//   law, so we integrate the actual geodesics. We use Cartesian Kerr-Schild
//   coordinates (horizon-penetrating, no polar axis trouble) and Hamiltonian
//   form:  H = 1/2 [ -pt² + |p|² - f (l·p)² ],  f = 2r³/(r⁴ + a²z²),
//   with dx/dλ = ∂H/∂p (analytic) and dp/dλ = -∂H/∂x (central differences),
//   advanced with RK4. Backwards camera rays use pt = +1 (a reversed null
//   geodesic is the same curve with p → -p; this matters in Kerr because
//   frame dragging is not time-symmetric).
// - Disk redshift is exact for circular equatorial orbits:
//   g = 1 / (u^t (1 - Ω λ)), Ω = 1/(r^{3/2} + a),
//   u^t = 1/sqrt(1 - 3/r + 2a r^{-3/2}), λ = L_z/E of the photon.
// - Formulas validated against Bardeen's exact critical impact parameters in
//   scripts/validate-kerr.mjs before being ported here.

export const VERT_SRC = `#version 300 es
void main() {
  vec2 v = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(v * 2.0 - 1.0, 0.0, 1.0);
}
`;

export const FRAG_SRC = `#version 300 es
precision highp float;

out vec4 fragColor;

uniform vec2  uRes;
uniform float uTime;
uniform vec3  uCamPos;
uniform vec3  uCamFwd;
uniform vec3  uCamRight;
uniform vec3  uCamUp;
uniform float uSpin;     // a
uniform float uHorizon;  // r+ = 1 + sqrt(1 - a^2)
uniform float uIsco;     // prograde ISCO radius

const float PI = 3.14159265359;

const float DISK_OUT  = 17.0;
const float T_INNER   = 6500.0;
const float ESCAPE_R  = 60.0;
const int   MAX_STEPS = 300;

// ---------- hashing / noise (same as the Schwarzschild renderer) ----------
float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
vec3 hash33(vec3 p3) {
  p3 = fract(p3 * vec3(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yxz + 33.33);
  return fract((p3.xxy + p3.yxx) * p3.zyx);
}
float noise2(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash12(i),              hash12(i + vec2(1, 0)), f.x),
             mix(hash12(i + vec2(0, 1)), hash12(i + vec2(1, 1)), f.x), f.y);
}
float fbm(vec2 p) {
  float a = 0.5, s = 0.0;
  for (int i = 0; i < 5; i++) {
    s += a * noise2(p);
    p = p * 2.07 + vec2(13.7, 7.1);
    a *= 0.5;
  }
  return s;
}

vec3 blackbody(float t) {
  t = max(t, 400.0);
  vec3 c;
  c.r = 56100000.0 * pow(t, -1.5) + 148.0;
  c.g = t > 6500.0 ? 35200000.0 * pow(t, -1.5) + 184.0 : 100.04 * log(t) - 623.6;
  c.b = 194.18 * log(t) - 1448.6;
  c = clamp(c, 0.0, 255.0) / 255.0;
  if (t < 1000.0) c *= t / 1000.0;
  return c;
}

vec3 stars(vec3 d) {
  vec3 col = vec3(0.0);
  for (int layer = 0; layer < 2; layer++) {
    float scale = layer == 0 ? 22.0 : 47.0;
    vec3 q = d * scale;
    vec3 id = floor(q);
    vec3 h = hash33(id);
    vec3 sp = id + 0.2 + 0.6 * h;
    float dist = length(q - sp);
    float lit = step(0.82, hash33(id + 17.0).x);
    float core = exp(-dist * dist * 220.0);
    float temp = mix(2800.0, 14000.0, h.y * h.y);
    float mag = 0.3 + 2.2 * h.z * h.z;
    col += lit * core * mag * blackbody(temp);
  }
  vec3 gn = normalize(vec3(0.35, 0.2, 1.0));
  float band = exp(-pow(dot(d, gn) * 3.2, 2.0));
  float neb = fbm(vec2(atan(d.y, d.x) * 3.0, d.z * 6.0) + 3.7);
  col += band * (0.012 + 0.05 * neb * neb) * vec3(0.55, 0.62, 0.85);
  return col;
}

// ---------- Kerr-Schild machinery (spin axis = +z) ----------
// Boyer-Lindquist r from Cartesian Kerr-Schild position
float ksRadius(vec3 q) {
  float R2 = dot(q, q);
  float b = R2 - uSpin * uSpin;
  float r2 = 0.5 * (b + sqrt(b * b + 4.0 * uSpin * uSpin * q.z * q.z));
  return sqrt(max(r2, 1e-8));
}

// H = 1/2 [ -pt^2 + |p|^2 - f (l.p)^2 ] with pt = +1 (backwards ray)
float hamiltonian(vec3 q, vec3 p) {
  float a = uSpin;
  float r = ksRadius(q);
  float r2 = r * r;
  float f = 2.0 * r2 * r / (r2 * r2 + a * a * q.z * q.z);
  float den = r2 + a * a;
  vec3 l = vec3((r * q.x + a * q.y) / den, (r * q.y - a * q.x) / den, q.z / r);
  float L = -1.0 + dot(l, p);
  return 0.5 * (-1.0 + dot(p, p) - f * L * L);
}

// dx/dlambda = dH/dp (analytic)
vec3 dHdp(vec3 q, vec3 p) {
  float a = uSpin;
  float r = ksRadius(q);
  float r2 = r * r;
  float f = 2.0 * r2 * r / (r2 * r2 + a * a * q.z * q.z);
  float den = r2 + a * a;
  vec3 l = vec3((r * q.x + a * q.y) / den, (r * q.y - a * q.x) / den, q.z / r);
  float L = -1.0 + dot(l, p);
  return p - f * L * l;
}

// dp/dlambda = -dH/dx (central differences)
vec3 dHdxNeg(vec3 q, vec3 p) {
  float eps = 7.0e-4 * max(ksRadius(q), 0.5);
  vec2 e = vec2(eps, 0.0);
  return -vec3(
    hamiltonian(q + e.xyy, p) - hamiltonian(q - e.xyy, p),
    hamiltonian(q + e.yxy, p) - hamiltonian(q - e.yxy, p),
    hamiltonian(q + e.yyx, p) - hamiltonian(q - e.yyx, p)
  ) / (2.0 * eps);
}

// ---------- accretion disk (equatorial plane z = 0) ----------
vec3 diskEmission(vec3 q, vec3 p, out float absorb) {
  absorb = 0.0;
  float a = uSpin;
  float r = ksRadius(q); // ~ BL radius near the equator
  if (r < uIsco - 0.3 || r > DISK_OUT) return vec3(0.0);

  float h = 0.14 + 0.03 * (r - uIsco);
  float vert = exp(-0.5 * q.z * q.z / (h * h));
  if (vert < 1e-4) return vec3(0.0);

  // prograde Keplerian rotation (same sense as the frame dragging)
  float omega = 1.0 / (pow(r, 1.5) + a);
  float ang = omega * (uTime + 150.0) * 0.55;
  float ca = cos(ang), sa = sin(ang);
  vec2 qq = mat2(ca, -sa, sa, ca) * q.xy; // rotate back by the orbit angle

  float n = fbm(qq * 0.7) * 0.7 + fbm(qq * 2.1) * 0.3;
  float phi = atan(q.y, q.x);
  float spiral = 0.5 + 0.5 * sin(3.0 * phi - 7.0 * log(r) + ang * 3.0);
  float rings = fbm(vec2(r * 1.55, 0.7));
  float dens = vert * (0.1 + 0.9 * pow(n, 2.4)) * (0.5 + 0.5 * spiral) * (0.35 + 0.65 * rings);
  dens *= smoothstep(uIsco - 0.3, uIsco + 0.6, r);
  dens *= 1.0 - smoothstep(DISK_OUT - 6.0, DISK_OUT, r);
  if (dens < 1e-4) return vec3(0.0);

  // exact redshift for a circular equatorial emitter, observer at infinity:
  // g = 1 / (u^t (1 - Omega * lambda)), lambda = L_z/E of the photon.
  // Our backwards ray has p_phys = -p, so lambda = -(x p_y - y p_x).
  float ut2 = 1.0 - 3.0 / r + 2.0 * a * pow(r, -1.5);
  float ut = inversesqrt(max(ut2, 0.02));
  float lam = -(q.x * p.y - q.y * p.x);
  float g = 1.0 / max(ut * (1.0 - omega * lam), 0.2);
  g = min(g, 3.5);

  float T = T_INNER * pow(r / uIsco, -0.75);
  vec3 cobs = blackbody(T * g);
  float lum = pow(T / T_INNER, 1.8) * pow(g, 3.0);

  absorb = dens * 1.3;
  return cobs * dens * lum * 4.0;
}

// ---------- geodesic integration ----------
vec3 trace(vec3 ro, vec3 rd) {
  float a = uSpin;
  vec3 q = ro;

  // momentum scale s so that H(q, s*rd) = 0 with pt = +1
  {
    float r = ksRadius(q);
    float r2 = r * r;
    float f = 2.0 * r2 * r / (r2 * r2 + a * a * q.z * q.z);
    float den = r2 + a * a;
    vec3 l = vec3((r * q.x + a * q.y) / den, (r * q.y - a * q.x) / den, q.z / r);
    float c = dot(l, rd);
    float A = 1.0 - f * c * c, B = 2.0 * f * c, C = -(1.0 + f);
    rd *= (-B + sqrt(B * B - 4.0 * A * C)) / (2.0 * A);
  }
  vec3 p = rd;

  vec3 col = vec3(0.0);
  float trans = 1.0;

  for (int i = 0; i < MAX_STEPS; i++) {
    float r = ksRadius(q);

    if (r < uHorizon * 1.02) { trans = 0.0; break; }
    if (dot(q, q) > ESCAPE_R * ESCAPE_R && dot(q, dHdp(q, p)) > 0.0) {
      col += trans * stars(normalize(dHdp(q, p)));
      return col;
    }

    // fine steps near the hole: frame dragging makes phi wind fast there
    float dl = clamp(0.085 * (r - 0.9 * uHorizon), 0.006, 2.0);
    if (abs(q.z) < 1.6 && r < DISK_OUT + 2.0 && r > uIsco - 2.0)
      dl = min(dl, 0.17);

    // RK4
    vec3 k1q = dHdp(q, p),               k1p = dHdxNeg(q, p);
    vec3 k2q = dHdp(q + 0.5 * dl * k1q, p + 0.5 * dl * k1p);
    vec3 k2p = dHdxNeg(q + 0.5 * dl * k1q, p + 0.5 * dl * k1p);
    vec3 k3q = dHdp(q + 0.5 * dl * k2q, p + 0.5 * dl * k2p);
    vec3 k3p = dHdxNeg(q + 0.5 * dl * k2q, p + 0.5 * dl * k2p);
    vec3 k4q = dHdp(q + dl * k3q,        p + dl * k3p);
    vec3 k4p = dHdxNeg(q + dl * k3q,     p + dl * k3p);
    q += dl / 6.0 * (k1q + 2.0 * k2q + 2.0 * k3q + k4q);
    p += dl / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p);

    float ab;
    vec3 e = diskEmission(q, p, ab);
    if (ab > 0.0) {
      col += trans * e * dl;
      trans *= exp(-ab * dl);
      if (trans < 0.005) break;
    }
  }
  return col;
}

vec3 aces(vec3 x) {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

void main() {
  vec2 ndc = (2.0 * gl_FragCoord.xy - uRes) / uRes.y;
  float focal = 1.6;
  vec3 rd = normalize(uCamFwd * focal + uCamRight * ndc.x + uCamUp * ndc.y);

  vec3 col = trace(uCamPos, rd);

  col *= 1.35;
  col = aces(col);
  col = pow(col, vec3(1.0 / 2.2));
  col += (hash12(gl_FragCoord.xy + fract(uTime)) - 0.5) / 255.0;
  fragColor = vec4(col, 1.0);
}
`;
