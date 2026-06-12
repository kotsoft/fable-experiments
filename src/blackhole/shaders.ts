// GLSL sources for the black hole renderer.
//
// Physics notes:
// - Units: rs (Schwarzschild radius) = 1, c = 1, so M = 1/2.
// - Null geodesics in Schwarzschild obey the Binet equation u'' + u = (3/2) rs u^2.
//   In 3D Cartesian form this is an inverse-quartic "force":
//       a = -(3/2) rs * h^2 * x / |x|^5
//   where h = |x × v| is the conserved specific angular momentum of the photon.
// - The accretion disk is a thin volumetric slab in the equatorial plane from the
//   ISCO outward, with blackbody emission shifted and beamed by relativistic g^3.
import {
  ACES_GLSL,
  BLACKBODY_GLSL,
  createStarsGlsl,
  HASH_NOISE_GLSL,
  JITTER_GLSL,
  VERT_SRC,
} from '../common/glsl';

export { VERT_SRC };

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

const float DISK_IN   = 3.0;
const float DISK_OUT  = 14.0;
const float T_INNER   = 6500.0;
const float ESCAPE_R  = 40.0;
const int   MAX_STEPS = 380;

${HASH_NOISE_GLSL}
${BLACKBODY_GLSL}
${createStarsGlsl({ bandNormal: [0.35, 1.0, 0.2] })}
${JITTER_GLSL}

vec3 diskEmission(vec3 p, vec3 rayDir, out float absorb) {
  absorb = 0.0;
  float r = length(p.xz);
  if (r < DISK_IN - 0.4 || r > DISK_OUT) return vec3(0.0);

  float h = 0.07 + 0.015 * (r - DISK_IN);
  float vert = exp(-0.5 * p.y * p.y / (h * h));
  if (vert < 1e-4) return vec3(0.0);

  float omega = 0.7071 / (r * sqrt(r));
  float ang = omega * (uTime + 90.0) * 0.6;
  float ca = cos(ang), sa = sin(ang);
  vec2 q = mat2(ca, -sa, sa, ca) * p.xz;

  float n = fbm(q * 1.4) * 0.7 + fbm(q * 4.2) * 0.3;
  float phi = atan(p.z, p.x);
  float spiral = 0.5 + 0.5 * sin(3.0 * phi - 7.0 * log(r) + ang * 3.0);
  float rings = fbm(vec2(r * 3.1, 0.7));
  float dens = vert * (0.1 + 0.9 * pow(n, 2.4)) * (0.5 + 0.5 * spiral) * (0.35 + 0.65 * rings);
  dens *= smoothstep(DISK_IN - 0.4, DISK_IN + 0.8, r);
  dens *= 1.0 - smoothstep(DISK_OUT - 5.0, DISK_OUT, r);

  if (dens < 1e-4) return vec3(0.0);

  float beta = sqrt(0.5 / max(r - 1.0, 0.55));
  beta = min(beta, 0.85);
  vec3 betaVec = beta * normalize(vec3(p.z, 0.0, -p.x));
  vec3 photonDir = -normalize(rayDir);
  float gamma = inversesqrt(1.0 - beta * beta);
  float doppler = 1.0 / (gamma * (1.0 - dot(betaVec, photonDir)));
  float gGrav = sqrt(max(1.0 - 1.0 / r, 0.0));
  float g = doppler * gGrav;

  float T = T_INNER * pow(r / DISK_IN, -0.75);
  vec3 cobs = blackbody(T * g);
  float lum = pow(T / T_INNER, 1.8) * pow(g, 3.0);

  absorb = dens * 2.6;
  return cobs * dens * lum * 8.0;
}

vec3 trace(vec3 ro, vec3 rd, float samplePhase) {
  vec3 p = ro;
  vec3 v = rd;
  vec3 hv = cross(p, v);
  float h2 = dot(hv, hv);

  vec3 col = vec3(0.0);
  float trans = 1.0;

  for (int i = 0; i < MAX_STEPS; i++) {
    float r2 = dot(p, p);
    float r = sqrt(r2);

    if (r < 1.0) { trans = 0.0; break; }
    if (r2 > ESCAPE_R * ESCAPE_R && dot(p, v) > 0.0) {
      col += trans * stars(normalize(v));
      return col;
    }

    float dt = clamp(0.12 * r, 0.02, 0.7);
    if (abs(p.y) < 0.9 && r < DISK_OUT + 1.0 && r > DISK_IN - 1.5)
      dt = min(dt, 0.085);

    vec3 p0 = p;
    vec3 a1 = -1.5 * h2 * p / (r2 * r2 * r);
    vec3 pn = p + v * dt + 0.5 * a1 * dt * dt;
    float rn2 = dot(pn, pn);
    vec3 a2 = -1.5 * h2 * pn / (rn2 * rn2 * sqrt(rn2));
    v += 0.5 * (a1 + a2) * dt;
    p = pn;

    float ab;
    vec3 e = diskEmission(mix(p0, p, samplePhase), v, ab);
    if (ab > 0.0) {
      col += trans * e * dt;
      trans *= exp(-ab * dt);
      if (trans < 0.005) break;
    }
  }
  return col;
}

${ACES_GLSL}

void main() {
  vec2 pixel = gl_FragCoord.xy + pixelJitter() - 0.5;
  float samplePhase = samplePhaseJitter();

  vec2 ndc = (2.0 * pixel - uRes) / uRes.y;
  float focal = 1.6;
  vec3 rd = normalize(uCamFwd * focal + uCamRight * ndc.x + uCamUp * ndc.y);

  vec3 col = trace(uCamPos, rd, samplePhase);

  col *= 1.35;
  col = aces(col);
  col = pow(col, vec3(1.0 / 2.2));
  col += (hash12(gl_FragCoord.xy + fract(uTime)) - 0.5) / 255.0;
  fragColor = vec4(col, 1.0);
}
`;
