// GLSL sources for the black hole renderer.
//
// Physics notes:
// - Units: rs (Schwarzschild radius) = 1, c = 1, so M = 1/2.
// - Null geodesics in Schwarzschild obey the Binet equation u'' + u = (3/2) rs u^2.
//   In 3D Cartesian form this is an inverse-quartic "force":
//       a = -(3/2) rs * h^2 * x / |x|^5
//   where h = |x × v| is the (conserved) specific angular momentum of the photon.
//   This is exact for Schwarzschild photons, so the lensing (Einstein ring,
//   photon ring at r = 1.5 rs, shadow at ~2.6 rs apparent radius) is physical.
// - The accretion disk is a thin volumetric slab in the equatorial plane from the
//   ISCO (r = 3 rs) outward. Gas follows circular Keplerian orbits; emission is
//   blackbody with T ∝ r^-3/4 (Shakura–Sunyaev), shifted and beamed by the
//   combined gravitational + Doppler factor g, with intensity scaled by g^3
//   (relativistic invariance of I_ν/ν^3). This is the asymmetry Interstellar
//   famously toned down — here it stays on.

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

const float PI = 3.14159265359;

// --- scene parameters (rs = 1) ---
const float DISK_IN   = 3.0;   // ISCO for Schwarzschild
const float DISK_OUT  = 14.0;
const float T_INNER   = 6500.0; // K at the inner edge
const float ESCAPE_R  = 40.0;
const int   MAX_STEPS = 380;

// ---------- hashing / noise ----------
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
  return mix(mix(hash12(i),                 hash12(i + vec2(1, 0)), f.x),
             mix(hash12(i + vec2(0, 1)),    hash12(i + vec2(1, 1)), f.x), f.y);
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

// ---------- blackbody color (approx Planckian locus, t in Kelvin) ----------
vec3 blackbody(float t) {
  t = max(t, 400.0);
  vec3 c;
  c.r = 56100000.0 * pow(t, -1.5) + 148.0;
  c.g = t > 6500.0 ? 35200000.0 * pow(t, -1.5) + 184.0 : 100.04 * log(t) - 623.6;
  c.b = 194.18 * log(t) - 1448.6;
  c = clamp(c, 0.0, 255.0) / 255.0;
  if (t < 1000.0) c *= t / 1000.0; // fade to black below dull red
  return c;
}

// ---------- starfield + faint galaxy band (sampled with the *bent* ray) ----------
vec3 stars(vec3 d) {
  vec3 col = vec3(0.0);
  for (int layer = 0; layer < 2; layer++) {
    float scale = layer == 0 ? 22.0 : 47.0;
    vec3 q = d * scale;
    vec3 id = floor(q);
    vec3 h = hash33(id);
    // one candidate star per cell, only a fraction of cells lit
    vec3 sp = id + 0.2 + 0.6 * h;
    float dist = length(q - sp);
    float lit = step(0.82, hash33(id + 17.0).x);
    float core = exp(-dist * dist * 220.0);
    float temp = mix(2800.0, 14000.0, h.y * h.y);
    float mag = 0.3 + 2.2 * h.z * h.z;
    col += lit * core * mag * blackbody(temp);
  }
  // tilted dim galactic band
  vec3 gn = normalize(vec3(0.35, 1.0, 0.2));
  float band = exp(-pow(dot(d, gn) * 3.2, 2.0));
  float neb = fbm(vec2(atan(d.z, d.x) * 3.0, d.y * 6.0) + 3.7);
  col += band * (0.012 + 0.05 * neb * neb) * vec3(0.55, 0.62, 0.85);
  return col;
}

// ---------- accretion disk ----------
// Returns rgb emission rate and writes absorption coefficient.
vec3 diskEmission(vec3 p, vec3 rayDir, out float absorb) {
  absorb = 0.0;
  float r = length(p.xz);
  if (r < DISK_IN - 0.4 || r > DISK_OUT) return vec3(0.0);

  // slab thickness, slightly flaring outward
  float h = 0.07 + 0.015 * (r - DISK_IN);
  float vert = exp(-0.5 * p.y * p.y / (h * h));
  if (vert < 1e-4) return vec3(0.0);

  // differential (Keplerian) rotation: Omega = sqrt(M/r^3), M = 1/2
  float omega = 0.7071 / (r * sqrt(r));
  float ang = omega * (uTime + 90.0) * 0.6; // offset pre-shears the turbulence into streaks
  float ca = cos(ang), sa = sin(ang);
  vec2 q = mat2(ca, -sa, sa, ca) * p.xz;

  // turbulent streaks: fbm advected by the shear + integer-harmonic spirals (seam-free)
  float n = fbm(q * 1.4) * 0.7 + fbm(q * 4.2) * 0.3;
  float phi = atan(p.z, p.x);
  float spiral = 0.5 + 0.5 * sin(3.0 * phi - 7.0 * log(r) + ang * 3.0);
  float rings = fbm(vec2(r * 3.1, 0.7));
  float dens = vert * (0.1 + 0.9 * pow(n, 2.4)) * (0.5 + 0.5 * spiral) * (0.35 + 0.65 * rings);
  dens *= smoothstep(DISK_IN - 0.4, DISK_IN + 0.8, r);     // soft inner edge
  dens *= 1.0 - smoothstep(DISK_OUT - 5.0, DISK_OUT, r);    // outer fade

  if (dens < 1e-4) return vec3(0.0);

  // local orbital velocity measured by a static observer: v = sqrt(M/(r - 2M)), M = 1/2
  float beta = sqrt(0.5 / max(r - 1.0, 0.55));
  beta = min(beta, 0.85);
  vec3 betaVec = beta * normalize(vec3(p.z, 0.0, -p.x)); // prograde tangent

  // photon propagation direction (disk -> camera) is opposite to marching dir
  vec3 photonDir = -normalize(rayDir);
  float gamma = inversesqrt(1.0 - beta * beta);
  float doppler = 1.0 / (gamma * (1.0 - dot(betaVec, photonDir)));
  float gGrav = sqrt(max(1.0 - 1.0 / r, 0.0));
  float g = doppler * gGrav;

  // Shakura–Sunyaev temperature profile, shifted by g for the observer
  float T = T_INNER * pow(r / DISK_IN, -0.75);
  vec3 cobs = blackbody(T * g);

  // I_obs = g^3 I_em ; gentle radial emissivity so the outer disk stays visible
  float lum = pow(T / T_INNER, 1.8) * pow(g, 3.0);

  absorb = dens * 2.6;
  return cobs * dens * lum * 8.0;
}

// ---------- geodesic integration ----------
vec3 trace(vec3 ro, vec3 rd) {
  vec3 p = ro;
  vec3 v = rd;
  vec3 hv = cross(p, v);
  float h2 = dot(hv, hv); // conserved angular momentum^2

  vec3 col = vec3(0.0);
  float trans = 1.0;

  for (int i = 0; i < MAX_STEPS; i++) {
    float r2 = dot(p, p);
    float r = sqrt(r2);

    if (r < 1.0) { trans = 0.0; break; }           // crossed the horizon: black
    if (r2 > ESCAPE_R * ESCAPE_R && dot(p, v) > 0.0) {
      col += trans * stars(normalize(v));           // escaped: lensed sky
      return col;
    }

    // adaptive step: fine near the hole and inside the disk slab
    float dt = clamp(0.12 * r, 0.02, 0.7);
    if (abs(p.y) < 0.9 && r < DISK_OUT + 1.0 && r > DISK_IN - 1.5)
      dt = min(dt, 0.085);

    // velocity Verlet on a = -(3/2) h^2 p / r^5   (rs = 1)
    vec3 a1 = -1.5 * h2 * p / (r2 * r2 * r);
    vec3 pn = p + v * dt + 0.5 * a1 * dt * dt;
    float rn2 = dot(pn, pn);
    vec3 a2 = -1.5 * h2 * pn / (rn2 * rn2 * sqrt(rn2));
    v += 0.5 * (a1 + a2) * dt;
    p = pn;

    float ab;
    vec3 e = diskEmission(p, v, ab);
    if (ab > 0.0) {
      col += trans * e * dt;
      trans *= exp(-ab * dt);
      if (trans < 0.005) break;
    }
  }
  return col; // captured or absorbed
}

// ---------- tonemap ----------
vec3 aces(vec3 x) {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

void main() {
  vec2 ndc = (2.0 * gl_FragCoord.xy - uRes) / uRes.y;
  float focal = 1.6; // ~64 deg vertical fov
  vec3 rd = normalize(uCamFwd * focal + uCamRight * ndc.x + uCamUp * ndc.y);

  vec3 col = trace(uCamPos, rd);

  col *= 1.35;           // exposure
  col = aces(col);
  col = pow(col, vec3(1.0 / 2.2));

  // mild dithering to kill banding in the dark sky
  col += (hash12(gl_FragCoord.xy + fract(uTime)) - 0.5) / 255.0;
  fragColor = vec4(col, 1.0);
}
`;
