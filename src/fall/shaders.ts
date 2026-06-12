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
uniform vec3  uObserverPos;
uniform vec4  uTetradTime;
uniform vec4  uTetradRight;
uniform vec4  uTetradUp;
uniform vec4  uTetradForward;
uniform vec3  uObserverBeta;
uniform float uInterior;
uniform float uSingularityFade;
uniform float uDiskEnabled;

const float DISK_IN   = 3.0;
const float DISK_OUT  = 14.0;
const float T_INNER   = 6200.0;
const float ESCAPE_R  = 45.0;
const int   MAX_STEPS = 360;
const vec3  DISK_NORMAL = normalize(vec3(0.18, 1.0, 0.0));
const vec3  DISK_AXIS_A = vec3(0.0, 0.0, 1.0);
const vec3  DISK_AXIS_B = normalize(cross(DISK_NORMAL, DISK_AXIS_A));

${HASH_NOISE_GLSL}
${BLACKBODY_GLSL}
${createStarsGlsl({ bandNormal: [0.35, 1.0, 0.2] })}
${JITTER_GLSL}

vec3 aberrateToStaticFrame(vec3 rd, vec3 beta) {
  float b2 = dot(beta, beta);
  if (b2 < 1e-6) return rd;
  float gamma = inversesqrt(max(1.0 - b2, 1e-5));
  float bd = dot(beta, rd);
  vec3 n = rd + (((gamma - 1.0) * bd / b2) + gamma) * beta;
  return normalize(n / (gamma * (1.0 + bd)));
}

vec3 diskVolumeEmission(vec3 p, vec3 rayDir, out float absorption) {
  absorption = 0.0;
  if (uDiskEnabled < 0.5) return vec3(0.0);
  vec2 dp = vec2(dot(p, DISK_AXIS_B), dot(p, DISK_AXIS_A));
  float r = length(dp);
  if (r < DISK_IN - 0.35 || r > DISK_OUT) return vec3(0.0);

  float height = dot(p, DISK_NORMAL);
  float h = 0.12 + 0.026 * (r - DISK_IN);
  float vertical = exp(-0.5 * height * height / (h * h));
  if (vertical < 1e-4) return vec3(0.0);

  float omega = 0.7071 / (r * sqrt(r));
  float ang = omega * (uTime + 80.0);
  float ca = cos(ang), sa = sin(ang);
  vec2 q = mat2(ca, -sa, sa, ca) * dp;

  float n = fbm(q * 0.62) * 0.75 + fbm(q * 1.35) * 0.25;
  float phi = atan(dp.y, dp.x);
  float spiral = 0.5 + 0.5 * sin(2.0 * phi - 4.2 * log(r) + ang * 1.35);
  float density = vertical * (0.58 + 0.42 * n) * (0.82 + 0.18 * spiral);
  density *= smoothstep(DISK_IN - 0.35, DISK_IN + 0.65, r);
  density *= 1.0 - smoothstep(DISK_OUT - 5.0, DISK_OUT, r);
  if (density < 1e-4) return vec3(0.0);

  float beta = sqrt(0.5 / max(r - 1.0, 0.55));
  beta = min(beta, 0.85);
  vec3 radialVec = normalize(DISK_AXIS_B * dp.x + DISK_AXIS_A * dp.y);
  vec3 betaVec = beta * normalize(cross(DISK_NORMAL, radialVec));
  vec3 photonDir = -normalize(rayDir);
  float gamma = inversesqrt(1.0 - beta * beta);
  float doppler = 1.0 / (gamma * (1.0 - dot(betaVec, photonDir)));
  float gGrav = sqrt(max(1.0 - 1.0 / r, 0.0));
  float g = doppler * gGrav;

  float T = T_INNER * pow(r / DISK_IN, -0.75);
  vec3 cobs = blackbody(T * g);
  float lum = pow(T / T_INNER, 1.8) * pow(g, 3.0);

  absorption = density * 0.055;
  return cobs * density * lum * 2.65;
}

float ksRadius(vec3 q) {
  return max(length(q), 0.045);
}

float hamiltonian(vec3 q, vec3 p) {
  float r = ksRadius(q);
  float f = 1.0 / r; // rs = 1, so 2M/r = 1/r
  vec3 l = q / r;
  float L = -1.0 + dot(l, p); // backwards camera ray, pt = +1
  return 0.5 * (-1.0 + dot(p, p) - f * L * L);
}

vec3 dHdp(vec3 q, vec3 p) {
  float r = ksRadius(q);
  float f = 1.0 / r;
  vec3 l = q / r;
  float L = -1.0 + dot(l, p);
  return p - f * L * l;
}

vec3 dHdxNeg(vec3 q, vec3 p) {
  float eps = 5.0e-4 * max(ksRadius(q), 0.25);
  vec2 e = vec2(eps, 0.0);
  return -vec3(
    hamiltonian(q + e.xyy, p) - hamiltonian(q - e.xyy, p),
    hamiltonian(q + e.yxy, p) - hamiltonian(q - e.yxy, p),
    hamiltonian(q + e.yyx, p) - hamiltonian(q - e.yyx, p)
  ) / (2.0 * eps);
}

vec3 traceHamiltonian(vec3 ro, vec3 rd, float samplePhase) {
  vec3 q = ro;
  bool cameraInside = length(ro) < 1.0;

  {
    float r = ksRadius(q);
    float f = 1.0 / r;
    vec3 l = q / r;
    float c = dot(l, rd);
    float A = 1.0 - f * c * c;
    float B = 2.0 * f * c;
    float C = -(1.0 + f);
    float disc = max(B * B - 4.0 * A * C, 1e-6);
    rd *= (-B + sqrt(disc)) / (2.0 * A);
  }
  vec3 p = rd;
  vec3 col = vec3(0.0);
  float trans = 1.0;

  for (int i = 0; i < MAX_STEPS; i++) {
    float r = ksRadius(q);
    vec3 rayVel = dHdp(q, p);

    if (!cameraInside && r < 1.002) break;
    if (r < 0.055) break;
    if (dot(q, q) > ESCAPE_R * ESCAPE_R && dot(q, rayVel) > 0.0) {
      col += trans * stars(normalize(rayVel));
      return col;
    }

    float dl = clamp(0.075 * r, 0.004, 0.75);
    if (uDiskEnabled > 0.5 && abs(dot(q, DISK_NORMAL)) < 0.8 && r < DISK_OUT + 1.0 && r > DISK_IN - 1.5)
      dl = min(dl, 0.032);

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

    if (!cameraInside && ksRadius(q) < 1.002) break;

    for (int j = 0; j < 3; j++) {
      float st = (float(j) + samplePhase) / 3.0;
      float subDl = dl / 3.0;
      float ab;
      vec3 e = diskVolumeEmission(
        mix(q0, q, st),
        normalize(mix(dHdp(q0, p0), dHdp(q, p), st)),
        ab
      );
      if (ab > 0.0) {
        col += trans * e * subDl;
        trans *= exp(-ab * subDl);
      }
      if (trans < 0.004) break;
    }
  }
  return col;
}

float radialPotential(float r, float b) {
  float f = 1.0 - 1.0 / r;
  return 1.0 - f * b * b / (r * r);
}

vec3 traceSchwarzschildExterior(vec3 ro, vec3 rd, float samplePhase) {
  float r = length(ro);
  vec3 er0 = ro / r;
  float nRadial = dot(rd, er0);
  vec3 tangent = rd - nRadial * er0;
  float nTangent = length(tangent);
  vec3 et0 = nTangent > 1e-5
    ? tangent / nTangent
    : normalize(cross(abs(er0.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0), er0));

  float f0 = max(1.0 - 1.0 / r, 1e-4);
  float b = r * nTangent / sqrt(f0);
  float radialSign = nRadial < 0.0 ? -1.0 : 1.0;
  float psi = 0.0;
  vec3 col = vec3(0.0);
  float trans = 1.0;

  for (int i = 0; i < MAX_STEPS; i++) {
    if (r < 1.001) break;

    vec3 er = cos(psi) * er0 + sin(psi) * et0;
    vec3 et = -sin(psi) * er0 + cos(psi) * et0;
    float vRadial2 = radialPotential(r, b);

    if (vRadial2 < 0.0) {
      radialSign = 1.0;
      vRadial2 = max(-vRadial2, 1e-5);
    }

    if (r > ESCAPE_R && radialSign > 0.0) {
      vec3 finalDir = normalize(radialSign * sqrt(max(vRadial2, 0.0)) * er + (b / r) * et);
      col += trans * stars(finalDir);
      return col;
    }

    float ds = clamp(0.045 * r, 0.004, 0.34);
    if (uDiskEnabled > 0.5 && abs(dot(r * er, DISK_NORMAL)) < 0.8 && r < DISK_OUT + 1.0 && r > DISK_IN - 1.5)
      ds = min(ds, 0.026);

    vec3 q0 = r * er;
    float r0 = r;
    float psi0 = psi;
    float vRadial = sqrt(max(vRadial2, 1e-6));
    r += radialSign * vRadial * ds;
    psi += b * ds / max(r0 * r0, 1e-4);

    if (radialSign < 0.0 && radialPotential(max(r, 1.001), b) < 0.0) {
      radialSign = 1.0;
      r = r0;
      psi = psi0;
    }

    vec3 er1 = cos(psi) * er0 + sin(psi) * et0;
    vec3 q1 = r * er1;

    for (int j = 0; j < 3; j++) {
      float st = (float(j) + samplePhase) / 3.0;
      float subDs = ds / 3.0;
      float ab;
      vec3 e = diskVolumeEmission(mix(q0, q1, st), normalize(q1 - q0), ab);
      if (ab > 0.0) {
        col += trans * e * subDs;
        trans *= exp(-ab * subDs);
      }
      if (trans < 0.004) break;
    }
  }
  return col;
}

vec3 trace(vec3 ro, vec3 rd, float samplePhase) {
  if (length(ro) < 1.0) return traceHamiltonian(ro, rd, samplePhase);
  return traceSchwarzschildExterior(ro, rd, samplePhase);
}

${ACES_GLSL}

void main() {
  vec2 pixel = gl_FragCoord.xy + pixelJitter() - 0.5;
  vec2 ndc = (2.0 * pixel - uRes) / uRes.y;
  float focal = 1.35;
  vec3 rdLocal = normalize(vec3(ndc.x, ndc.y, focal));
  vec4 photon = uTetradTime +
    uTetradRight * rdLocal.x +
    uTetradUp * rdLocal.y +
    uTetradForward * rdLocal.z;
  vec3 rd = normalize(photon.yzw);
  float r = length(uObserverPos);

  float samplePhase = samplePhaseJitter();
  vec3 col = trace(uObserverPos, rd, samplePhase);

  float f = max(abs(1.0 - 1.0 / r), 0.025);
  float b2 = min(dot(uObserverBeta, uObserverBeta), 0.95);
  float gamma = inversesqrt(1.0 - b2);
  float shift = clamp(gamma * (1.0 + dot(uObserverBeta, rd)) / sqrt(f), 0.25, 4.0);
  col *= mix(vec3(1.18, 0.78, 0.62), vec3(0.68, 0.82, 1.35), smoothstep(0.75, 1.8, shift));
  col *= 1.0 + 0.25 * log(max(shift, 0.1));

  col *= 1.45;
  col += uInterior * vec3(0.02, 0.025, 0.04) * smoothstep(0.85, 0.2, r);
  col = aces(col);
  col = pow(col, vec3(1.0 / 2.2));
  col += (hash12(gl_FragCoord.xy + fract(uTime)) - 0.5) / 255.0;
  col = mix(col, vec3(0.0), uSingularityFade);
  fragColor = vec4(col, 1.0);
}
`;
