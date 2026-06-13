// WGSL for the fallfable renderer.
//
// One compute kernel traces a past-directed null geodesic per pixel - the
// time-reverse of the photon that arrived at the camera - through the Kerr
// metric in ingoing Kerr-Schild coordinates. Inside the horizon the past light
// cone points strictly outward, so the view keeps working below r+.
//
// The Hamiltonian gradient is analytic (mirrors src/fallfable/kerr.ts), which
// is both several times faster and far more accurate in f32 than finite
// differences of H.

const SKY_COMMON_WGSL = /* wgsl */ `
const PI = 3.141592653589793;
const TAU = 6.283185307179586;

fn blackbody(temp: f32) -> vec3<f32> {
  let t = clamp(temp, 800.0, 40000.0) * 0.01;
  var r = 1.0;
  var g = 0.39 * log(t) - 0.632;
  var b = 0.543 * log(max(t - 10.0, 1.0e-3)) - 1.196;
  if (t > 66.0) {
    r = 1.293 * pow(t - 60.0, -0.1332);
    g = 1.13 * pow(t - 60.0, -0.0755);
    b = 1.0;
  }
  return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn hash31(p: vec3<f32>) -> f32 {
  var q = fract(p * vec3<f32>(443.897, 441.423, 437.195));
  q = q + dot(q, q.yzx + 19.19);
  return fract((q.x + q.y) * q.z);
}

fn hash33(p: vec3<f32>) -> vec3<f32> {
  var q = fract(p * vec3<f32>(443.897, 441.423, 437.195));
  q = q + dot(q, q.yxz + 19.19);
  return fract((q.xxy + q.yzz) * q.zyx);
}

fn vnoise(p: vec3<f32>) -> f32 {
  let i = floor(p);
  let w = fract(p);
  let s = w * w * (3.0 - 2.0 * w);
  let n000 = hash31(i);
  let n100 = hash31(i + vec3<f32>(1.0, 0.0, 0.0));
  let n010 = hash31(i + vec3<f32>(0.0, 1.0, 0.0));
  let n110 = hash31(i + vec3<f32>(1.0, 1.0, 0.0));
  let n001 = hash31(i + vec3<f32>(0.0, 0.0, 1.0));
  let n101 = hash31(i + vec3<f32>(1.0, 0.0, 1.0));
  let n011 = hash31(i + vec3<f32>(0.0, 1.0, 1.0));
  let n111 = hash31(i + vec3<f32>(1.0, 1.0, 1.0));
  return mix(
    mix(mix(n000, n100, s.x), mix(n010, n110, s.x), s.y),
    mix(mix(n001, n101, s.x), mix(n011, n111, s.x), s.y),
    s.z
  );
}

fn fbm(p: vec3<f32>) -> f32 {
  var total = 0.0;
  var amp = 0.5;
  var q = p;
  for (var i = 0; i < 4; i = i + 1) {
    total = total + amp * vnoise(q);
    q = q * 2.13 + vec3<f32>(7.7, 3.1, 1.9);
    amp = amp * 0.5;
  }
  return total;
}

fn fbm3(p: vec3<f32>) -> f32 {
  var total = 0.0;
  var amp = 0.5;
  var q = p;
  for (var i = 0; i < 3; i = i + 1) {
    total = total + amp * vnoise(q);
    q = q * 2.13 + vec3<f32>(7.7, 3.1, 1.9);
    amp = amp * 0.5;
  }
  return total;
}

fn star_falloff(ang2: f32) -> f32 {
  let x = clamp(1.0 - ang2 * 110.0, 0.0, 1.0);
  let s = x * x * (3.0 - 2.0 * x);
  return s * s;
}

fn star_layer(dir: vec3<f32>, scale: f32, density: f32, gain: f32, gshift: f32) -> vec3<f32> {
  let cell = floor(dir * scale);
  let pick = hash31(cell + 71.7);
  if (pick < 1.0 - density) {
    return vec3<f32>(0.0);
  }

  let h = hash33(cell);
  // Jitter stays inside the cell and the compact falloff dies before the cell
  // boundary, so the single containing-cell lookup never shows seams.
  let starDir = normalize(cell + 0.5 + (h - 0.5) * 0.6);
  let c = dot(dir, starDir);
  let ang2 = 2.0 * max(1.0 - c, 0.0) * scale * scale;
  let falloff = star_falloff(ang2);
  if (falloff <= 0.0) {
    return vec3<f32>(0.0);
  }

  let intensity = hash31(cell + 17.3);
  let intensity2 = intensity * intensity;
  let brightness = intensity2 * intensity2 * intensity * gain;
  let temp = (2600.0 + 11000.0 * h.z * h.z) * gshift;
  return blackbody(temp) * brightness * falloff;
}

const GALACTIC_POLE = vec3<f32>(0.184, 0.92, 0.353);
const GALACTIC_CORE = vec3<f32>(0.927, 0.0, 0.375);

fn milky_way(dir: vec3<f32>) -> vec3<f32> {
  let pole = normalize(GALACTIC_POLE);
  let sinLat = dot(dir, pole);
  let band = exp(-sinLat * sinLat * 14.0);
  if (band < 1.0e-3) {
    return vec3<f32>(0.0);
  }
  let clouds = fbm(dir * 4.6 + 3.1);
  let wisps = fbm(dir * 11.0 - 5.7);
  let dust = smoothstep(0.5, 0.78, fbm(dir * 6.5 - 11.3)) * exp(-sinLat * sinLat * 70.0);
  let core = pow(max(dot(dir, normalize(GALACTIC_CORE)), 0.0), 5.0);
  let warm = vec3<f32>(1.0, 0.78, 0.55);
  let cool = vec3<f32>(0.5, 0.62, 1.0);
  let tint = mix(cool, warm, clamp(0.2 + 1.3 * core, 0.0, 1.0));
  let glow = band * (0.3 + 0.55 * clouds + 0.35 * wisps * band) * (1.0 - 0.82 * dust) * (0.35 + 1.8 * core);
  return tint * glow;
}

fn sky_direction_to_uv(dir: vec3<f32>) -> vec2<f32> {
  let d = normalize(dir);
  let lon = atan2(d.z, d.x);
  let uCoord = fract(lon / TAU + 0.5);
  let vCoord = acos(clamp(d.y, -1.0, 1.0)) / PI;
  return vec2<f32>(uCoord, vCoord);
}

fn sky_uv_to_direction(uv: vec2<f32>) -> vec3<f32> {
  let theta = uv.y * PI;
  let lon = (uv.x - 0.5) * TAU;
  let ring = sin(theta);
  return vec3<f32>(cos(lon) * ring, cos(theta), sin(lon) * ring);
}
`;

export const SKY_ATLAS_WGSL = /* wgsl */ `
${SKY_COMMON_WGSL}

@group(0) @binding(0) var milkyOut: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let dims = textureDimensions(milkyOut);
  if (id.x >= dims.x || id.y >= dims.y) {
    return;
  }

  let uv = (vec2<f32>(f32(id.x), f32(id.y)) + vec2<f32>(0.5)) / vec2<f32>(f32(dims.x), f32(dims.y));
  let d = sky_uv_to_direction(uv);
  let milky = milky_way(d) * blackbody(5400.0) * 0.9;

  textureStore(milkyOut, vec2<i32>(i32(id.x), i32(id.y)), vec4<f32>(milky, 1.0));
}
`;

export const DISK_NOISE_WGSL = /* wgsl */ `
const DISK_NOISE_WIDTH = 256u;
const DISK_NOISE_HEIGHT = 256u;
const DISK_NOISE_DEPTH = 64u;
const DISK_NOISE_OCTAVES = 5;

@group(0) @binding(0) var noiseOut: texture_storage_3d<rgba8unorm, write>;

fn wrap_index(value: i32, period: u32) -> u32 {
  let p = i32(period);
  return u32(((value % p) + p) % p);
}

fn hash_index3(x: u32, y: u32, z: u32) -> f32 {
  var h = (x + 0x9e3779b9u) * 0x85ebca6bu;
  h = h ^ ((y + 0xc2b2ae35u) * 0x27d4eb2fu);
  h = h ^ ((z + 0x165667b1u) * 0x9e3779b1u);
  h = (h ^ (h >> 15u)) * 0x2c1b3c6du;
  h = (h ^ (h >> 12u)) * 0x297a2d39u;
  h = h ^ (h >> 15u);
  return f32(h) * 2.3283064365386963e-10;
}

fn periodic_value_noise(p: vec3<f32>, period: vec3<u32>) -> f32 {
  let i = vec3<i32>(floor(p));
  let f = fract(p);
  let s = f * f * (3.0 - 2.0 * f);
  let x0 = wrap_index(i.x, period.x);
  let x1 = wrap_index(i.x + 1, period.x);
  let y0 = wrap_index(i.y, period.y);
  let y1 = wrap_index(i.y + 1, period.y);
  let z0 = wrap_index(i.z, period.z);
  let z1 = wrap_index(i.z + 1, period.z);

  let n000 = hash_index3(x0, y0, z0);
  let n100 = hash_index3(x1, y0, z0);
  let n010 = hash_index3(x0, y1, z0);
  let n110 = hash_index3(x1, y1, z0);
  let n001 = hash_index3(x0, y0, z1);
  let n101 = hash_index3(x1, y0, z1);
  let n011 = hash_index3(x0, y1, z1);
  let n111 = hash_index3(x1, y1, z1);

  return mix(
    mix(mix(n000, n100, s.x), mix(n010, n110, s.x), s.y),
    mix(mix(n001, n101, s.x), mix(n011, n111, s.x), s.y),
    s.z
  );
}

fn periodic_fbm3(p: vec3<f32>) -> f32 {
  var total = 0.0;
  var amp = 0.5;
  var q = p;
  var period = vec3<u32>(DISK_NOISE_WIDTH, DISK_NOISE_HEIGHT, DISK_NOISE_DEPTH);
  for (var i = 0; i < DISK_NOISE_OCTAVES; i = i + 1) {
    total = total + amp * periodic_value_noise(q, period);
    q = q * 2.0 + vec3<f32>(7.7, 3.1, 1.9);
    period = period * vec3<u32>(2u);
    amp = amp * 0.5;
  }
  return total;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= DISK_NOISE_WIDTH || id.y >= DISK_NOISE_HEIGHT || id.z >= DISK_NOISE_DEPTH) {
    return;
  }
  let p = vec3<f32>(f32(id.x), f32(id.y), f32(id.z)) + vec3<f32>(0.5);
  let noise = clamp(periodic_fbm3(p), 0.0, 1.0);
  textureStore(noiseOut, vec3<i32>(id), vec4<f32>(noise, 0.0, 0.0, 1.0));
}
`;

export const TRACE_WGSL = /* wgsl */ `
struct Uniforms {
  camPosition: vec4<f32>, // t, x, y, z
  eTime: vec4<f32>,
  eRight: vec4<f32>,
  eUp: vec4<f32>,
  eForward: vec4<f32>,
  geo: vec4<f32>,    // spin, mass, baseStep, escapeRadius
  march: vec4<f32>,  // maxSteps, singularityCutoff, tanHalfFov, aspect
  disk: vec4<f32>,   // innerRadius, outerRadius, innerTemperature, emissivity
  disk2: vec4<f32>,  // boostPower, spinDirection, scaleHeight, absorption
  sky: vec4<f32>,    // starIntensity, milkyWayIntensity, ambient, diagnosticMode
  anim: vec4<f32>,   // textureTimeScale, hotspotIntensity, unused, unused
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var outImage: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var skySampler: sampler;
@group(0) @binding(3) var skyMilky: texture_2d<f32>;
@group(0) @binding(4) var classifierImage: texture_2d<f32>;
@group(0) @binding(5) var lensOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var lensImage: texture_2d<f32>;
@group(0) @binding(7) var diskNoiseSampler: sampler;
@group(0) @binding(8) var diskNoise: texture_3d<f32>;
const DISK_NOISE_PERIOD = vec3<f32>(256.0, 256.0, 64.0);

// ---------------------------------------------------------------- geometry

struct Geometry {
  r: f32,
  f: f32,
  l: vec3<f32>,
  gradR: vec3<f32>,
  gradF: vec3<f32>,
  jlx: vec3<f32>,
  jly: vec3<f32>,
  jlz: vec3<f32>,
};

fn ks_radius(p: vec3<f32>) -> f32 {
  let a = u.geo.x;
  let b = dot(p, p) - a * a;
  let r2 = 0.5 * (b + sqrt(b * b + 4.0 * a * a * p.z * p.z));
  return sqrt(max(r2, 0.0));
}

fn geometry_at(p: vec3<f32>) -> Geometry {
  let a = u.geo.x;
  let m = u.geo.y;
  let b = dot(p, p) - a * a;
  let disc = sqrt(b * b + 4.0 * a * a * p.z * p.z);
  let r2 = max(0.5 * (b + disc), 1.0e-12);
  let r = sqrt(r2);
  let s = r2 + a * a;

  let gradDen = max(r * disc, 1.0e-12);
  let gradR = vec3<f32>(r2 * p.x, r2 * p.y, (r2 + a * a) * p.z) / gradDen;

  let d = max(r2 * r2 + a * a * p.z * p.z, 1.0e-12);
  let f = 2.0 * m * r2 * r / d;
  let fScale = 2.0 * m / (d * d);
  let fr = r2 * (3.0 * a * a * p.z * p.z - r2 * r2);
  let gradF = fScale * (fr * gradR - vec3<f32>(0.0, 0.0, 2.0 * a * a * p.z * r2 * r));

  let l = vec3<f32>((r * p.x + a * p.y) / s, (r * p.y - a * p.x) / s, p.z / r);
  let s2 = s * s;
  let cx = (p.x * (a * a - r2) - 2.0 * a * r * p.y) / s2;
  let cy = (p.y * (a * a - r2) + 2.0 * a * r * p.x) / s2;
  let jlx = vec3<f32>(r / s, a / s, 0.0) + gradR * cx;
  let jly = vec3<f32>(-a / s, r / s, 0.0) + gradR * cy;
  let jlz = vec3<f32>(0.0, 0.0, 1.0 / r) - gradR * (p.z / r2);

  return Geometry(r, f, l, gradR, gradF, jlx, jly, jlz);
}

fn metric_dot(p: vec3<f32>, av: vec4<f32>, bv: vec4<f32>) -> f32 {
  let g = geometry_at(p);
  let la = av.x + dot(g.l, av.yzw);
  let lb = bv.x + dot(g.l, bv.yzw);
  return -av.x * bv.x + dot(av.yzw, bv.yzw) + g.f * la * lb;
}

fn lower_vec(p: vec3<f32>, v: vec4<f32>) -> vec4<f32> {
  let g = geometry_at(p);
  let lv = v.x + dot(g.l, v.yzw);
  return vec4<f32>(-v.x + g.f * lv, v.yzw + g.f * lv * g.l);
}

struct Derivative {
  velocity: vec4<f32>,
  force: vec3<f32>,
};

fn phase_derivative(p: vec3<f32>, mom: vec4<f32>) -> Derivative {
  let g = geometry_at(p);
  let q = -mom.x + dot(g.l, mom.yzw);
  let velocity = vec4<f32>(-mom.x + g.f * q, mom.yzw - g.f * q * g.l);
  let dq = vec3<f32>(
    mom.y * g.jlx.x + mom.z * g.jly.x + mom.w * g.jlz.x,
    mom.y * g.jlx.y + mom.z * g.jly.y + mom.w * g.jlz.y,
    mom.y * g.jlx.z + mom.z * g.jly.z + mom.w * g.jlz.z
  );
  let force = 0.5 * q * q * g.gradF + g.f * q * dq;
  return Derivative(velocity, force);
}

struct Phase {
  position: vec4<f32>,
  momentum: vec4<f32>,
};

fn rk4(s: Phase, h: f32) -> Phase {
  let d1 = phase_derivative(s.position.yzw, s.momentum);
  return rk4_from_d1(s, h, d1);
}

fn rk4_from_d1(s: Phase, h: f32, d1: Derivative) -> Phase {
  let s2 = Phase(s.position + d1.velocity * (h * 0.5), s.momentum + vec4<f32>(0.0, d1.force) * (h * 0.5));
  let d2 = phase_derivative(s2.position.yzw, s2.momentum);
  let s3 = Phase(s.position + d2.velocity * (h * 0.5), s.momentum + vec4<f32>(0.0, d2.force) * (h * 0.5));
  let d3 = phase_derivative(s3.position.yzw, s3.momentum);
  let s4 = Phase(s.position + d3.velocity * h, s.momentum + vec4<f32>(0.0, d3.force) * h);
  let d4 = phase_derivative(s4.position.yzw, s4.momentum);
  let w = h / 6.0;
  return Phase(
    s.position + w * (d1.velocity + 2.0 * d2.velocity + 2.0 * d3.velocity + d4.velocity),
    s.momentum + w * vec4<f32>(0.0, d1.force + 2.0 * d2.force + 2.0 * d3.force + d4.force)
  );
}

// ---------------------------------------------------------------- shading

${SKY_COMMON_WGSL}

// gshift = observed/emitted frequency ratio for light from infinity.
fn sky_radiance(dir: vec3<f32>, gshift: f32) -> vec3<f32> {
  let d = normalize(dir);
  let uv = sky_direction_to_uv(d);
  let g3 = clamp(gshift * gshift * gshift, 1.0e-4, 28.0);
  let shiftTint = blackbody(5400.0 * gshift) / max(blackbody(5400.0), vec3<f32>(1.0e-3));
  let milky = textureSampleLevel(skyMilky, skySampler, uv, 0.0).rgb * shiftTint;
  let stars =
    star_layer(d, 16.0, 0.22, 1.6, gshift) +
    star_layer(d, 33.0, 0.28, 0.85, gshift) +
    star_layer(d, 64.0, 0.33, 0.45, gshift);
  var col = vec3<f32>(0.012, 0.015, 0.028) * u.sky.z;
  col = col + milky * u.sky.y;
  col = col + stars * u.sky.x;
  return col * g3;
}

fn termination_color(status: f32) -> vec3<f32> {
  if (status == 0.0) { return vec3<f32>(0.0, 0.55, 0.12); } // max steps
  if (status == 1.0) { return vec3<f32>(0.62, 0.54, 0.0); } // singularity
  if (status == 2.0) { return vec3<f32>(0.62, 0.04, 0.02); } // horizon
  if (status == 3.0) { return vec3<f32>(0.55, 0.0, 0.58); } // momentum blow-up
  if (status == 4.0) { return vec3<f32>(0.0, 0.22, 0.72); } // sky escape
  return vec3<f32>(0.9, 0.42, 0.04); // disk opacity
}

fn cost_color(t: f32) -> vec3<f32> {
  let x = clamp(t, 0.0, 1.0);
  let cold = vec3<f32>(0.02, 0.08, 0.45);
  let mid = vec3<f32>(0.0, 0.78, 0.62);
  let warm = vec3<f32>(0.95, 0.85, 0.12);
  let hot = vec3<f32>(1.0, 0.05, 0.02);
  let low = mix(cold, mid, smoothstep(0.0, 0.45, x));
  let high = mix(warm, hot, smoothstep(0.72, 1.0, x));
  return mix(low, high, smoothstep(0.35, 0.82, x));
}

// ---------------------------------------------------------------- disk

fn disk_omega(r: f32) -> f32 {
  let sm = sqrt(u.geo.y);
  let dir = u.disk2.y;
  return dir * sm / (pow(r, 1.5) + dir * u.geo.x * sm);
}

fn disk_emitter(p: vec3<f32>, r: f32) -> vec4<f32> {
  let om = disk_omega(r);
  let candidate = vec4<f32>(1.0, -om * p.y, om * p.x, 0.0);
  var n = metric_dot(p, candidate, candidate);
  if (n < -1.0e-6) {
    return candidate / sqrt(-n);
  }
  // No timelike circular orbit here: fall back to the Eulerian observer.
  let g = geometry_at(p);
  let root = sqrt(1.0 + g.f);
  return vec4<f32>(root, -g.f * g.l / root);
}

struct DiskSample {
  radiance: vec3<f32>,
  opacity: f32,
};

// A few bright blobs orbiting at their local Keplerian rate (the disk-flare
// phenomenology seen around Sgr A*). Their sweep - and the lensed echoes of it
// - is what makes the disk visibly rotate.
fn hotspot_boost(p: vec3<f32>, r: f32, t: f32) -> f32 {
  let az = atan2(p.y, p.x);
  var boost = 0.0;
  // (orbit radius, phase offset, gaussian sigma^2, amplitude)
  var spots = array<vec4<f32>, 3>(
    vec4<f32>(1.7, 0.0, 0.035, 2.6),
    vec4<f32>(2.7, 2.2, 0.07, 1.7),
    vec4<f32>(4.1, 4.4, 0.14, 1.1)
  );
  for (var i = 0; i < 3; i = i + 1) {
    let s = spots[i];
    let phase = disk_omega(s.x) * t + s.y;
    let dphi = az - phase;
    let w = atan2(sin(dphi), cos(dphi));
    let d2 = (r - s.x) * (r - s.x) + (s.x * w) * (s.x * w);
    boost = boost + s.w * exp(-d2 / s.z);
  }
  return boost;
}

fn disk_turbulence(
  p: vec3<f32>,
  angle: f32,
  zOffset: f32,
  inner: f32,
  tangentStretch: f32,
  spiralSign: f32,
) -> f32 {
  let ca = cos(angle);
  let sa = sin(angle);
  let q = vec3<f32>(p.x * ca - p.y * sa, p.x * sa + p.y * ca, p.z * 6.0 + zOffset);
  let rr = max(length(q.xy), 1.0e-3);
  let az = atan2(q.y, q.x);
  let spiralAz = az + spiralSign * 0.42 * log(max(rr / inner, 0.18));
  let arcRadius = rr / tangentStretch;
  let noiseP = vec3<f32>(
    rr * 1.35 + q.z * 0.06,
    cos(spiralAz) * arcRadius,
    sin(spiralAz) * arcRadius + q.z
  );
  return textureSampleLevel(diskNoise, diskNoiseSampler, fract(noiseP * (4.4 / inner) / DISK_NOISE_PERIOD), 0.0).r;
}

fn disk_sample(pos: vec4<f32>, mom: vec4<f32>, dl: f32) -> DiskSample {
  let p = pos.yzw;
  let r = ks_radius(p);
  let inner = u.disk.x;
  let outer = u.disk.y;
  if (r > outer || r < u.march.y * 2.0) {
    return DiskSample(vec3<f32>(0.0), 0.0);
  }
  // Below the ISCO the flow plunges: it stops radiating efficiently but does
  // not go dark, so keep a steeply fading stream all the way down. This is
  // what an observer inside the horizon sees of the matter falling ahead.
  var plungeFade = 1.0;
  var x = inner / r;
  if (r < inner) {
    plungeFade = exp((r - inner) * 4.5 / inner);
    x = min(x, 1.15);
  }
  let scaleHeight = u.disk2.z * r;
  let zr = p.z / scaleHeight;
  if (abs(zr) > 3.4) {
    return DiskSample(vec3<f32>(0.0), 0.0);
  }
  let vertical = exp(-0.5 * zr * zr);

  // Turbulence advected by differential rotation: rotate the sample point back
  // by Omega(r) * t (the ray carries its own coordinate time, so light-travel
  // delay across the disk is automatic) and look up static fbm there. The
  // r-dependent angle shears the pattern into trailing streaks over time.
  // Texture time runs faster than geometry time (u.anim.x): real Keplerian
  // periods at these radii are minutes, which reads as a frozen disk. The
  // redshift/beaming physics below still uses the true orbital velocity.
  let t_anim = pos.x * u.anim.x;
  // Bounded dual-phase advection: shearing one static field forever winds it
  // up without limit (and the wind-up transient reads as counter-rotation).
  // Instead, two staggered copies are each sheared within a bounded window
  // and crossfaded, so the gas continuously renews while drifting at the true
  // local orbital rate.
  let period = 15.0;
  let phase = t_anim / period;
  let t1 = fract(phase) * period;
  let t2 = fract(phase + 0.5) * period;
  let w1 = 1.0 - abs(2.0 * fract(phase) - 1.0);
  let om = disk_omega(r);
  let a1 = -om * t1;
  let a2 = -om * t2;
  // Artistic velocity shear without a blur trail: faster inner orbits stretch
  // the texture into sharper tangent-aligned filaments, while outer gas keeps
  // the softer unsheared turbulence.
  let velocityShear = smoothstep(0.22, 4.6, abs(om) * period) * smoothstep(0.10, 0.95, x);
  let tangentStretch = 1.0 + 4.4 * velocityShear;
  let spinSign = sign(om);
  let turb = mix(
    disk_turbulence(p, a2, 13.7, inner, tangentStretch, spinSign),
    disk_turbulence(p, a1, 0.0, inner, tangentStretch, spinSign),
    w1
  );
  let sm = smoothstep(0.31, 0.77, turb);
  let streaks = 0.12 + (1.38 + 0.22 * velocityShear) * sm * sm;

  // Novikov-Thorne-like profile: T = Tin (rin/r)^(3/4) (1 - sqrt(rin/r))^(1/4),
  // floored near the inner edge so the rim stays hot where the flow plunges.
  let nt = max(1.0 - sqrt(min(x, 1.0) * 0.995), 0.06);
  let temp = u.disk.z * pow(x, 0.75) * pow(nt, 0.25) * 1.55;
  if (temp < 900.0) {
    return DiskSample(vec3<f32>(0.0), 0.0);
  }

  let emitter = disk_emitter(p, r);
  // Past-directed ray: nu_obs / nu_em = 1 / dot(q, u_emitter).
  let freq = dot(mom, emitter);
  if (freq <= 1.0e-5) {
    return DiskSample(vec3<f32>(0.0), 0.0);
  }
  let gshift = 1.0 / freq;

  let density = vertical * streaks;
  let tNorm = temp / u.disk.z;
  let power = tNorm * tNorm * tNorm * tNorm;
  // Steep radial emissivity falloff: the outer disk is a translucent veil, so
  // a camera inside the slab is not swimming in glow.
  var hotspot = 0.0;
  if (r > 1.2 && r < 5.0 && u.anim.y > 0.0) {
    hotspot = hotspot_boost(p, r, t_anim);
  }
  let falloff = pow(x, 2.1) * plungeFade * (1.0 + hotspot * u.anim.y);
  let boosted = min(pow(gshift, u.disk2.x), 12.0);
  let radiance = blackbody(temp * gshift) * (u.disk.w * density * power * falloff * boosted * dl * 34.0);
  let opacity = u.disk2.w * density * pow(x, 1.4) * plungeFade * dl * 2.1;
  return DiskSample(radiance, opacity);
}

// ---------------------------------------------------------------- trace

const CLASSIFIER_MAX_STEP_SCALE = 0.4;
const SAFE_SKY_MAX_DISK_SIGNAL = 0.35;
const SAFE_SKY_NEIGHBOR_MAX_DISK_SIGNAL = 0.75;

struct TraceResult {
  color: vec3<f32>,
  status: f32,
  steps: f32,
  escapeDir: vec3<f32>,
  gshift: f32,
  diskSignal: f32,
};

fn trace_result_mode(px: vec2<f32>, dims: vec2<f32>, shadeSky: bool, maxStepScale: f32) -> TraceResult {
  let spin = u.geo.x;
  let mass = u.geo.y;
  let baseStep = u.geo.z;
  let escapeR = u.geo.w;
  let maxSteps = max(1, i32(floor(u.march.x * maxStepScale)));
  let cutoff = u.march.y;
  let horizon = mass + sqrt(max(mass * mass - spin * spin, 0.0));

  let ndc = vec2<f32>(
    (2.0 * (px.x + 0.5) / dims.x - 1.0) * u.march.w,
    1.0 - 2.0 * (px.y + 0.5) / dims.y
  );
  let n = normalize(vec3<f32>(ndc * u.march.z, 1.0));
  // Per-pixel step jitter decorrelates volumetric sampling phases between
  // neighboring rays, turning coherent slab banding into invisible noise.
  let stepJitter = 0.72 + 0.56 * hash31(vec3<f32>(px.x, px.y, 0.37));

  // Past-directed momentum of the photon arriving from view direction n:
  // q = -e_t + n_x e_right + n_y e_up + n_z e_forward, lowered. Then
  // dot(q, u_camera) = 1, so all redshifts come out per unit observed energy.
  let contra = -u.eTime + u.eRight * n.x + u.eUp * n.y + u.eForward * n.z;
  var state = Phase(u.camPosition, lower_vec(u.camPosition.yzw, contra));

  let m0 = abs(state.momentum.x) + length(state.momentum.yzw);
  let cameraInside = ks_radius(u.camPosition.yzw) <= horizon;
  var exited = !cameraInside;

  var color = vec3<f32>(0.0);
  var trans = 1.0;
  var stepsTaken = 0.0;
  var debugStatus = 0.0; // 0 max-steps, 1 cutoff, 2 horizon, 3 blow-up, 4 sky, 5 disk opacity
  var escapeDir = vec3<f32>(0.0, 0.0, 1.0);
  var escapeGshift = 1.0;
  var diskSignal = 0.0;

  for (var step = 0; step < maxSteps; step = step + 1) {
    stepsTaken = f32(step + 1);
    let p = state.position.yzw;
    let r = ks_radius(p);

    if (r <= cutoff) {
      debugStatus = 1.0;
      break; // singularity: remaining transmittance stays black
    }
    if (!exited && r > horizon * 1.02) {
      exited = true;
    }
    if (exited && r <= horizon) {
      debugStatus = 2.0;
      break; // reached the horizon from outside: infinitely redshifted
    }

    let mscale = abs(state.momentum.x) + length(state.momentum.yzw);
    // Near-critical horizon skimmers can spend hundreds of extra steps with
    // exploding momentum before they visibly become shadow.
    if (mscale > 2.75e2 * m0) {
      debugStatus = 3.0;
      break; // horizon-skimmer: momentum blow-up = shadow
    }

    let d1 = phase_derivative(p, state.momentum);

    let v = d1.velocity.yzw;
    let radialSpeed = dot(p, v);
    var escaped = r >= escapeR && radialSpeed > 0.0;
    // Strongly radial rays can skip the last weak-field shell; grazing rays
    // still use the full escape radius so photon-ring paths keep their shape.
    let fastEscapeR = max(u.disk.y * 1.65, escapeR * 0.8);
    let fastEscapeCos2 = 0.74;
    if (!escaped && r >= fastEscapeR && radialSpeed > 0.0) {
      escaped = radialSpeed * radialSpeed > fastEscapeCos2 * dot(p, p) * dot(v, v);
    }
    if (escaped) {
      // p_t is conserved; for past-directed q the sky shift is 1 / q_t.
      escapeDir = normalize(v);
      escapeGshift = 1.0 / max(state.momentum.x, 1.0e-4);
      if (shadeSky) {
        color = color + trans * sky_radiance(escapeDir, escapeGshift);
      }
      trans = 0.0;
      debugStatus = 4.0;
      break;
    }

    var h = baseStep * stepJitter * clamp(r * 0.55, 0.16, 6.0) * min(1.0, m0 / mscale);
    let scaleHeight = u.disk2.z * r;
    if (r > u.march.y * 2.0 && r < u.disk.y * 1.1 && abs(p.z) < 4.0 * scaleHeight) {
      h = min(h, (0.6 * scaleHeight + 0.015) * stepJitter);
      let sample = disk_sample(state.position, state.momentum, h * length(d1.velocity.yzw));
      if (sample.opacity > 0.0 || sample.radiance.x + sample.radiance.y + sample.radiance.z > 0.0) {
        let diskLum = dot(sample.radiance, vec3<f32>(0.2126, 0.7152, 0.0722));
        diskSignal = min(1.0, diskSignal + sample.opacity * 2.0 + diskLum * 0.02);
        color = color + trans * sample.radiance;
        trans = trans * exp(-sample.opacity);
        if (trans < 0.012) {
          debugStatus = 5.0;
          break; // disk is optically thick here
        }
      }
    }

    state = rk4_from_d1(state, h, d1);
  }

  return TraceResult(color, debugStatus, stepsTaken, escapeDir, escapeGshift, diskSignal);
}

fn trace_result(px: vec2<f32>, dims: vec2<f32>) -> TraceResult {
  return trace_result_mode(px, dims, true, 1.0);
}

fn trace_linear_cost(result: TraceResult) -> f32 {
  return clamp(result.steps / max(u.march.x, 1.0), 0.0, 1.0);
}

fn trace_cost(result: TraceResult) -> f32 {
  return pow(trace_linear_cost(result), 0.55);
}

fn trace_debug_color(result: TraceResult) -> vec3<f32> {
  let cost = trace_cost(result);
  if (u.sky.w > 0.5) {
    if (u.sky.w < 1.5) {
      return termination_color(result.status);
    }
    if (u.sky.w < 2.5) {
      return cost_color(cost);
    }
    return termination_color(result.status) * (0.16 + 1.85 * cost);
  }
  return result.color;
}

fn trace(px: vec2<f32>, dims: vec2<f32>) -> vec3<f32> {
  return trace_debug_color(trace_result(px, dims));
}

fn classifier_feature(result: TraceResult) -> vec4<f32> {
  return vec4<f32>(result.status, trace_linear_cost(result), trace_cost(result), result.diskSignal);
}

fn lens_feature(result: TraceResult) -> vec4<f32> {
  return vec4<f32>(normalize(result.escapeDir), clamp(result.gshift, 1.0e-4, 8.0));
}

fn classifier_status(feature: vec4<f32>) -> f32 {
  return floor(feature.x + 0.5);
}

fn classifier_risk(feature: vec4<f32>) -> f32 {
  let status = classifier_status(feature);
  var risk = feature.y;
  if (status == 5.0 || feature.w > SAFE_SKY_MAX_DISK_SIGNAL) {
    risk = risk + 2.0; // disk samples are never "safe sky/shadow" tiles.
  }
  if (status == 0.0 || status == 1.0) {
    risk = risk + 1.5; // cap/singularity means keep the tile on the slow path.
  }
  return risk;
}

fn worse_feature(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
  if (classifier_risk(b) > classifier_risk(a)) {
    return b;
  }
  return a;
}

fn classifier_color_from_feature(feature: vec4<f32>) -> vec3<f32> {
  let status = classifier_status(feature);
  let cost = feature.z;
  if (feature.w > SAFE_SKY_MAX_DISK_SIGNAL) {
    return vec3<f32>(0.9, 0.42, 0.04) * (0.16 + 1.85 * max(cost, feature.w));
  }
  if (status == 4.0 && cost < 0.38) {
    return mix(vec3<f32>(0.015, 0.07, 0.19), vec3<f32>(0.0, 0.34, 0.88), cost / 0.38);
  }
  if ((status == 2.0 || status == 3.0) && cost < 0.42) {
    return mix(vec3<f32>(0.10, 0.01, 0.015), vec3<f32>(0.62, 0.04, 0.12), cost / 0.42);
  }
  return termination_color(status) * (0.16 + 1.85 * cost);
}

fn load_classifier_feature(coord: vec2<u32>, dims: vec2<u32>) -> vec4<f32> {
  let clamped = min(coord, dims - vec2<u32>(1u, 1u));
  return textureLoad(classifierImage, vec2<i32>(i32(clamped.x), i32(clamped.y)), 0);
}

fn load_lens_feature(coord: vec2<u32>, dims: vec2<u32>) -> vec4<f32> {
  let clamped = min(coord, dims - vec2<u32>(1u, 1u));
  return textureLoad(lensImage, vec2<i32>(i32(clamped.x), i32(clamped.y)), 0);
}

fn lens_dir(feature: vec4<f32>) -> vec3<f32> {
  let dir = feature.xyz;
  if (dot(dir, dir) < 1.0e-8) {
    return vec3<f32>(0.0, 0.0, 1.0);
  }
  return normalize(dir);
}

fn tile_feature(tileCoord: vec2<u32>, classDims: vec2<u32>) -> vec4<f32> {
  let tileBase = tileCoord * 2u;
  var feature = load_classifier_feature(tileBase, classDims);
  feature = worse_feature(feature, load_classifier_feature(tileBase + vec2<u32>(1u, 0u), classDims));
  feature = worse_feature(feature, load_classifier_feature(tileBase + vec2<u32>(0u, 1u), classDims));
  feature = worse_feature(feature, load_classifier_feature(tileBase + vec2<u32>(1u, 1u), classDims));
  return feature;
}

fn tile_is_confirmed_shadow(tileCoord: vec2<u32>) -> bool {
  let feature = tile_feature(tileCoord, textureDimensions(classifierImage));
  let status = classifier_status(feature);
  return status == 2.0 || status == 3.0;
}

const ADAPTIVE_TILE_UNSAFE = 0u;
const ADAPTIVE_TILE_SHADOW = 1u;
const ADAPTIVE_TILE_SKY = 2u;
const SAFE_SKY_MAX_LINEAR_COST = 0.34;
const SAFE_SKY_MIN_DIR_DOT = 0.9975;
const SAFE_SKY_MAX_GSHIFT_SPREAD = 0.35;

var<workgroup> adaptiveTileKind: u32;

fn offset_tile_coord(tileCoord: vec2<u32>, offset: vec2<i32>, tileDims: vec2<u32>) -> vec2<u32> {
  let maxTile = vec2<i32>(i32(tileDims.x) - 1, i32(tileDims.y) - 1);
  return vec2<u32>(clamp(vec2<i32>(tileCoord) + offset, vec2<i32>(0, 0), maxTile));
}

fn tile_core_is_safe_sky(
  tileCoord: vec2<u32>,
  classDims: vec2<u32>,
  lensDims: vec2<u32>,
  maxDiskSignal: f32,
) -> bool {
  let tileBase = tileCoord * 2u;
  let f00 = load_classifier_feature(tileBase, classDims);
  let f10 = load_classifier_feature(tileBase + vec2<u32>(1u, 0u), classDims);
  let f01 = load_classifier_feature(tileBase + vec2<u32>(0u, 1u), classDims);
  let f11 = load_classifier_feature(tileBase + vec2<u32>(1u, 1u), classDims);
  if (
    classifier_status(f00) != 4.0 ||
    classifier_status(f10) != 4.0 ||
    classifier_status(f01) != 4.0 ||
    classifier_status(f11) != 4.0 ||
    f00.w > maxDiskSignal ||
    f10.w > maxDiskSignal ||
    f01.w > maxDiskSignal ||
    f11.w > maxDiskSignal
  ) {
    return false;
  }

  let maxCost = max(max(f00.y, f10.y), max(f01.y, f11.y));
  if (maxCost > SAFE_SKY_MAX_LINEAR_COST) {
    return false;
  }

  let l00 = load_lens_feature(tileBase, lensDims);
  let l10 = load_lens_feature(tileBase + vec2<u32>(1u, 0u), lensDims);
  let l01 = load_lens_feature(tileBase + vec2<u32>(0u, 1u), lensDims);
  let l11 = load_lens_feature(tileBase + vec2<u32>(1u, 1u), lensDims);
  let d00 = lens_dir(l00);
  let d10 = lens_dir(l10);
  let d01 = lens_dir(l01);
  let d11 = lens_dir(l11);
  let dirSum = d00 + d10 + d01 + d11;
  if (dot(dirSum, dirSum) < 1.0e-4) {
    return false;
  }
  let meanDir = normalize(dirSum);
  let minDirDot = min(min(dot(d00, meanDir), dot(d10, meanDir)), min(dot(d01, meanDir), dot(d11, meanDir)));
  if (minDirDot < SAFE_SKY_MIN_DIR_DOT) {
    return false;
  }

  let gMin = min(min(l00.w, l10.w), min(l01.w, l11.w));
  let gMax = max(max(l00.w, l10.w), max(l01.w, l11.w));
  if (gMax - gMin > SAFE_SKY_MAX_GSHIFT_SPREAD * max(gMax, 1.0e-3)) {
    return false;
  }

  return true;
}

fn adaptive_tile_kind(tileCoord: vec2<u32>) -> u32 {
  let classDims = textureDimensions(classifierImage);
  let lensDims = textureDimensions(lensImage);
  if (tile_is_confirmed_shadow(tileCoord)) {
    return ADAPTIVE_TILE_SHADOW;
  }

  let tileDims = (classDims + vec2<u32>(1u, 1u)) / 2u;
  for (var oy = -1; oy <= 1; oy = oy + 1) {
    for (var ox = -1; ox <= 1; ox = ox + 1) {
      let neighbor = offset_tile_coord(tileCoord, vec2<i32>(ox, oy), tileDims);
      var diskLimit = SAFE_SKY_NEIGHBOR_MAX_DISK_SIGNAL;
      if (ox == 0 && oy == 0) {
        diskLimit = SAFE_SKY_MAX_DISK_SIGNAL;
      }
      if (!tile_core_is_safe_sky(neighbor, classDims, lensDims, diskLimit)) {
        return ADAPTIVE_TILE_UNSAFE;
      }
    }
  }
  return ADAPTIVE_TILE_SKY;
}

fn tile_lens_at_pixel(tileCoord: vec2<u32>, id: vec2<u32>) -> vec4<f32> {
  let lensDims = textureDimensions(lensImage);
  let tileBase = tileCoord * 2u;
  let l00 = load_lens_feature(tileBase, lensDims);
  let l10 = load_lens_feature(tileBase + vec2<u32>(1u, 0u), lensDims);
  let l01 = load_lens_feature(tileBase + vec2<u32>(0u, 1u), lensDims);
  let l11 = load_lens_feature(tileBase + vec2<u32>(1u, 1u), lensDims);
  let tileOrigin = tileCoord * 8u;
  let localPx = vec2<f32>(f32(id.x - tileOrigin.x), f32(id.y - tileOrigin.y)) + vec2<f32>(0.5);
  let t = clamp((localPx - vec2<f32>(1.5)) / 4.0, vec2<f32>(0.0), vec2<f32>(1.0));
  let dirX0 = mix(lens_dir(l00), lens_dir(l10), t.x);
  let dirX1 = mix(lens_dir(l01), lens_dir(l11), t.x);
  let dir = normalize(mix(dirX0, dirX1, t.y));
  let gX0 = mix(l00.w, l10.w, t.x);
  let gX1 = mix(l01.w, l11.w, t.x);
  return vec4<f32>(dir, mix(gX0, gX1, t.y));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let dims = textureDimensions(outImage);
  if (id.x >= dims.x || id.y >= dims.y) {
    return;
  }
  let rgb = trace(vec2<f32>(f32(id.x), f32(id.y)), vec2<f32>(f32(dims.x), f32(dims.y)));
  textureStore(outImage, vec2<i32>(i32(id.x), i32(id.y)), vec4<f32>(rgb, 1.0));
}

@compute @workgroup_size(8, 8)
fn classify_features_main(@builtin(global_invocation_id) id: vec3<u32>) {
  let dims = textureDimensions(outImage);
  if (id.x >= dims.x || id.y >= dims.y) {
    return;
  }
  let origin = id.xy * 4u;
  let fullDims = vec2<f32>(f32(dims.x * 4u), f32(dims.y * 4u));
  let samplePx = vec2<f32>(f32(origin.x) + 1.5, f32(origin.y) + 1.5);
  let result = trace_result_mode(samplePx, fullDims, false, CLASSIFIER_MAX_STEP_SCALE);
  textureStore(outImage, vec2<i32>(i32(id.x), i32(id.y)), classifier_feature(result));
  textureStore(lensOut, vec2<i32>(i32(id.x), i32(id.y)), lens_feature(result));
}

@compute @workgroup_size(8, 8)
fn visualize_classifier_main(@builtin(global_invocation_id) id: vec3<u32>) {
  let dims = textureDimensions(outImage);
  if (id.x >= dims.x || id.y >= dims.y) {
    return;
  }
  let classDims = textureDimensions(classifierImage);
  var feature = load_classifier_feature(id.xy / 4u, classDims);
  if (u.sky.w > 4.5) {
    feature = tile_feature(id.xy / 8u, classDims);
  }
  let rgb = classifier_color_from_feature(feature);
  textureStore(outImage, vec2<i32>(i32(id.x), i32(id.y)), vec4<f32>(rgb, 1.0));
}

@compute @workgroup_size(8, 8)
fn shadow_fill_main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(workgroup_id) workgroupId: vec3<u32>) {
  let dims = textureDimensions(outImage);
  if (id.x >= dims.x || id.y >= dims.y) {
    return;
  }
  if (!tile_is_confirmed_shadow(workgroupId.xy)) {
    return;
  }
  var rgb = vec3<f32>(0.0);
  if (u.sky.w > 6.5) {
    rgb = vec3<f32>(0.0, 0.09, 0.12);
  }
  textureStore(outImage, vec2<i32>(i32(id.x), i32(id.y)), vec4<f32>(rgb, 1.0));
}

@compute @workgroup_size(8, 8)
fn shadow_trace_main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(workgroup_id) workgroupId: vec3<u32>) {
  let dims = textureDimensions(outImage);
  if (id.x >= dims.x || id.y >= dims.y) {
    return;
  }
  if (tile_is_confirmed_shadow(workgroupId.xy)) {
    return;
  }
  let rgb = trace_result(vec2<f32>(f32(id.x), f32(id.y)), vec2<f32>(f32(dims.x), f32(dims.y))).color;
  textureStore(outImage, vec2<i32>(i32(id.x), i32(id.y)), vec4<f32>(rgb, 1.0));
}

@compute @workgroup_size(8, 8)
fn adaptive_fill_main(
  @builtin(global_invocation_id) id: vec3<u32>,
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_id) localId: vec3<u32>
) {
  if (localId.x == 0u && localId.y == 0u) {
    adaptiveTileKind = adaptive_tile_kind(workgroupId.xy);
  }
  workgroupBarrier();

  let dims = textureDimensions(outImage);
  if (id.x >= dims.x || id.y >= dims.y || adaptiveTileKind == ADAPTIVE_TILE_UNSAFE) {
    return;
  }

  var rgb = vec3<f32>(0.0);
  if (adaptiveTileKind == ADAPTIVE_TILE_SKY) {
    let lens = tile_lens_at_pixel(workgroupId.xy, id.xy);
    rgb = sky_radiance(lens.xyz, lens.w);
    if (u.sky.w > 8.5) {
      rgb = rgb * vec3<f32>(0.82, 1.12, 0.9) + vec3<f32>(0.0, 0.035, 0.02);
    }
  } else if (u.sky.w > 8.5) {
    rgb = vec3<f32>(0.0, 0.09, 0.12);
  }
  textureStore(outImage, vec2<i32>(i32(id.x), i32(id.y)), vec4<f32>(rgb, 1.0));
}

@compute @workgroup_size(8, 8)
fn adaptive_trace_main(
  @builtin(global_invocation_id) id: vec3<u32>,
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_id) localId: vec3<u32>
) {
  if (localId.x == 0u && localId.y == 0u) {
    adaptiveTileKind = adaptive_tile_kind(workgroupId.xy);
  }
  workgroupBarrier();

  let dims = textureDimensions(outImage);
  if (id.x >= dims.x || id.y >= dims.y || adaptiveTileKind != ADAPTIVE_TILE_UNSAFE) {
    return;
  }
  let rgb = trace_result(vec2<f32>(f32(id.x), f32(id.y)), vec2<f32>(f32(dims.x), f32(dims.y))).color;
  textureStore(outImage, vec2<i32>(i32(id.x), i32(id.y)), vec4<f32>(rgb, 1.0));
}
`;

export const PRESENT_WGSL = /* wgsl */ `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var srcSampler: sampler;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vertex_main(@builtin(vertex_index) i: u32) -> VertexOutput {
  var corners = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0)
  );
  var out: VertexOutput;
  out.position = vec4<f32>(corners[i], 0.0, 1.0);
  out.uv = corners[i] * vec2<f32>(0.5, -0.5) + 0.5;
  return out;
}

fn aces(x: vec3<f32>) -> vec3<f32> {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fragment_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
  let hdr = textureSampleLevel(src, srcSampler, uv, 0.0).rgb;
  let mapped = aces(hdr);
  return vec4<f32>(pow(mapped, vec3<f32>(1.0 / 2.2)), 1.0);
}
`;
