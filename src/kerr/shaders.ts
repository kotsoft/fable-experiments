// GLSL sources for the Kerr (rotating) black hole renderer.
//
// Physics notes:
// - Units: G = c = M = 1. Spin parameter a = J/M.
// - The shader integrates null geodesics in Cartesian Kerr-Schild coordinates:
//   H = 1/2 [ -pt² + |p|² - f (l·p)² ], with RK4 and numerical dH/dx.
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
uniform float uSpin;
uniform float uHorizon;
uniform float uIsco;

const float PI = 3.14159265359;

const float DISK_OUT  = 17.0;
const float T_INNER   = 6500.0;
const float ESCAPE_R  = 60.0;
const int   MAX_STEPS = 300;

${HASH_NOISE_GLSL}
${BLACKBODY_GLSL}
${createStarsGlsl({ bandNormal: [0.35, 0.2, 1.0] })}
${JITTER_GLSL}

float ksRadius(vec3 q) {
  float R2 = dot(q, q);
  float b = R2 - uSpin * uSpin;
  float r2 = 0.5 * (b + sqrt(b * b + 4.0 * uSpin * uSpin * q.z * q.z));
  return sqrt(max(r2, 1e-8));
}

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

vec3 dHdxNeg(vec3 q, vec3 p) {
  float eps = 7.0e-4 * max(ksRadius(q), 0.5);
  vec2 e = vec2(eps, 0.0);
  return -vec3(
    hamiltonian(q + e.xyy, p) - hamiltonian(q - e.xyy, p),
    hamiltonian(q + e.yxy, p) - hamiltonian(q - e.yxy, p),
    hamiltonian(q + e.yyx, p) - hamiltonian(q - e.yyx, p)
  ) / (2.0 * eps);
}

vec3 diskEmission(vec3 q, vec3 p, out float absorb) {
  absorb = 0.0;
  float a = uSpin;
  float r = ksRadius(q);
  if (r < uIsco - 0.3 || r > DISK_OUT) return vec3(0.0);

  float h = 0.14 + 0.03 * (r - uIsco);
  float vert = exp(-0.5 * q.z * q.z / (h * h));
  if (vert < 1e-4) return vec3(0.0);

  float omega = 1.0 / (pow(r, 1.5) + a);
  float ang = omega * (uTime + 150.0);
  float ca = cos(ang), sa = sin(ang);
  vec2 qq = mat2(ca, -sa, sa, ca) * q.xy;

  float n = fbm(qq * 0.7) * 0.7 + fbm(qq * 2.1) * 0.3;
  float phi = atan(q.y, q.x);
  float spiral = 0.5 + 0.5 * sin(3.0 * phi - 7.0 * log(r) + ang * 3.0);
  float rings = fbm(vec2(r * 1.55, 0.7));
  float dens = vert * (0.1 + 0.9 * pow(n, 2.4)) * (0.5 + 0.5 * spiral) * (0.35 + 0.65 * rings);
  dens *= smoothstep(uIsco - 0.3, uIsco + 0.6, r);
  dens *= 1.0 - smoothstep(DISK_OUT - 6.0, DISK_OUT, r);
  if (dens < 1e-4) return vec3(0.0);

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

vec3 trace(vec3 ro, vec3 rd, float samplePhase) {
  float a = uSpin;
  vec3 q = ro;

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

    float dl = clamp(0.085 * (r - 0.9 * uHorizon), 0.006, 2.0);
    if (abs(q.z) < 1.6 && r < DISK_OUT + 2.0 && r > uIsco - 2.0)
      dl = min(dl, 0.17);

    vec3 q0 = q;
    vec3 p0 = p;

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
    vec3 e = diskEmission(mix(q0, q, samplePhase), mix(p0, p, samplePhase), ab);
    if (ab > 0.0) {
      col += trans * e * dl;
      trans *= exp(-ab * dl);
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
