export const VERT_SRC = `#version 300 es
void main() {
  vec2 v = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(v * 2.0 - 1.0, 0.0, 1.0);
}
`;

export const HASH_NOISE_GLSL = `
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
`;

export const BLACKBODY_GLSL = `
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
`;

export const ACES_GLSL = `
vec3 aces(vec3 x) {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}
`;

export const JITTER_GLSL = `
vec2 pixelJitter() {
  return vec2(
    hash12(gl_FragCoord.xy + 13.17),
    hash12(gl_FragCoord.yx + 71.43)
  );
}
float samplePhaseJitter() {
  return hash12(gl_FragCoord.xy + 127.1);
}
`;

export function createStarsGlsl(options: {
  bandNormal: [number, number, number];
}): string {
  const [x, y, z] = options.bandNormal;

  return `
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
  vec3 gn = normalize(vec3(${x}, ${y}, ${z}));
  float band = exp(-pow(dot(d, gn) * 3.2, 2.0));
  vec2 nebUv1 = vec2(d.x + 0.37 * d.z, d.y - 0.21 * d.x) * 5.0;
  vec2 nebUv2 = vec2(d.z - 0.31 * d.x, d.y + 0.27 * d.z) * 5.0;
  float neb = 0.5 * fbm(nebUv1 + 3.7) + 0.5 * fbm(nebUv2 + 11.1);
  col += band * (0.012 + 0.05 * neb * neb) * vec3(0.55, 0.62, 0.85);
  return col;
}
`;
}
