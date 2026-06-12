import {
  COMPOSITE_INPUT_FLOATS_PER_RAY,
  COMPOSITE_OUTPUT_FLOATS_PER_RAY,
  compareCompositeReadback,
  compositeOutputByteLength,
  compositeRayCount,
} from '../gr/compositeReadback';
import {
  COMPOSITE_CAMERA_UNIFORM_FLOATS,
  createCompositeCameraUniforms,
  type CompositeCameraSampleOptions,
} from '../gr/compositeSamples';
import { CAMERA_SAMPLE_SHADER } from './webgpuCameraSampleProbe';

export interface WebGpuCompositeProbeResult {
  supported: boolean;
  message: string;
  output?: Float32Array;
  maxAbsDiff?: number;
  statusMismatches?: number;
  diskMismatches?: number;
}

interface WebGpuConstants {
  GPUBufferUsage?: {
    UNIFORM: number;
    STORAGE: number;
    COPY_DST: number;
    COPY_SRC: number;
    MAP_READ: number;
  };
  GPUTextureUsage?: {
    RENDER_ATTACHMENT: number;
  };
  GPUMapMode?: {
    READ: number;
  };
}

export interface WebGpuCompositeRunResult {
  supported: boolean;
  message: string;
  output?: Float32Array;
}

export interface WebGpuCompositeCanvasOptions {
  readback?: boolean;
}

const COMPOSITE_SHADER = `
struct RaySample {
  position: vec4<f32>,
  momentum: vec4<f32>,
  observerVelocity: vec4<f32>,
  paramsA: vec4<f32>,
  paramsB: vec4<f32>,
  paramsC: vec4<f32>,
  paramsD: vec4<f32>,
};

struct StepOutput {
  position: vec4<f32>,
  momentum: vec4<f32>,
};

struct CompositeOutput {
  summary: vec4<f32>,
  color: vec4<f32>,
  sky: vec4<f32>,
  disk: vec4<f32>,
};

struct VolumeEmission {
  color: vec3<f32>,
  weight: f32,
  radius: f32,
};

@group(0) @binding(0) var<storage, read> samples: array<RaySample>;
@group(0) @binding(1) var<storage, read_write> outputData: array<CompositeOutput>;

fn ks_radius(p: vec3<f32>, a: f32) -> f32 {
  let r2_euclid = dot(p, p);
  if (abs(a) < 1.0e-6) {
    return sqrt(r2_euclid);
  }
  let b = r2_euclid - a * a;
  let r2 = 0.5 * (b + sqrt(b * b + 4.0 * a * a * p.z * p.z));
  return sqrt(max(r2, 0.0));
}

fn horizon_radius(mass: f32, spin: f32) -> f32 {
  if (mass <= 0.0) {
    return 0.0;
  }
  return mass + sqrt(max(mass * mass - spin * spin, 0.0));
}

fn ks_l(p: vec3<f32>, a: f32, r: f32) -> vec3<f32> {
  let den = max(r * r + a * a, 1.0e-8);
  return vec3<f32>(
    (r * p.x + a * p.y) / den,
    (r * p.y - a * p.x) / den,
    p.z / max(r, 1.0e-8)
  );
}

fn ks_scalar(position: vec3<f32>, spin: f32, mass: f32) -> f32 {
  if (mass <= 0.0) {
    return 0.0;
  }
  let r = ks_radius(position, spin);
  let r2 = r * r;
  return 2.0 * mass * r2 * r / max(r2 * r2 + spin * spin * position.z * position.z, 1.0e-8);
}

fn ks_radius_gradient(position: vec3<f32>, spin: f32, radius: f32) -> vec3<f32> {
  let safe_radius = max(radius, 1.0e-8);
  if (abs(spin) < 1.0e-6) {
    return position / safe_radius;
  }

  let radius2 = radius * radius;
  let b = dot(position, position) - spin * spin;
  let denominator = max(2.0 * radius2 - b, 1.0e-8);
  return vec3<f32>(
    radius * position.x / denominator,
    radius * position.y / denominator,
    position.z * (radius2 + spin * spin) / (safe_radius * denominator)
  );
}

fn ks_scalar_gradient(
  position: vec3<f32>,
  spin: f32,
  mass: f32,
  radius: f32,
  radius_gradient: vec3<f32>
) -> vec3<f32> {
  if (mass <= 0.0) {
    return vec3<f32>(0.0);
  }

  let radius2 = radius * radius;
  let radius3 = radius2 * radius;
  let denominator = max(radius2 * radius2 + spin * spin * position.z * position.z, 1.0e-8);
  let numerator = 2.0 * mass * radius3;
  let numerator_gradient = 6.0 * mass * radius2 * radius_gradient;
  let denominator_gradient =
    4.0 * radius3 * radius_gradient +
    vec3<f32>(0.0, 0.0, 2.0 * spin * spin * position.z);
  return (numerator_gradient * denominator - numerator * denominator_gradient) / (denominator * denominator);
}

fn ks_l_derivative_dot(
  position: vec3<f32>,
  spin: f32,
  radius: f32,
  radius_derivative: f32,
  basis: vec3<f32>,
  spatial_momentum: vec3<f32>
) -> f32 {
  let radius2 = radius * radius;
  let denominator = max(radius2 + spin * spin, 1.0e-8);
  let denominator_derivative = 2.0 * radius * radius_derivative;
  let denominator2 = denominator * denominator;

  let nx = radius * position.x + spin * position.y;
  let ny = radius * position.y - spin * position.x;
  let dnx = radius_derivative * position.x + radius * basis.x + spin * basis.y;
  let dny = radius_derivative * position.y + radius * basis.y - spin * basis.x;

  let dlx = (dnx * denominator - nx * denominator_derivative) / denominator2;
  let dly = (dny * denominator - ny * denominator_derivative) / denominator2;
  let dlz = (basis.z * radius - position.z * radius_derivative) / max(radius2, 1.0e-8);
  return dot(vec3<f32>(dlx, dly, dlz), spatial_momentum);
}

fn hamiltonian(position: vec3<f32>, momentum: vec4<f32>, spin: f32, mass: f32) -> f32 {
  let r = ks_radius(position, spin);
  let scalar = ks_scalar(position, spin, mass);
  let l = ks_l(position, spin, r);
  let pt = momentum.x;
  let ps = momentum.yzw;
  let minkowski = -pt * pt + dot(ps, ps);
  let projected = -pt + dot(l, ps);
  return 0.5 * (minkowski - scalar * projected * projected);
}

fn metric_dot(position: vec3<f32>, spin: f32, mass: f32, a: vec4<f32>, b: vec4<f32>) -> f32 {
  let r = ks_radius(position, spin);
  let scalar = ks_scalar(position, spin, mass);
  let l = ks_l(position, spin, r);
  let la = a.x + dot(l, a.yzw);
  let lb = b.x + dot(l, b.yzw);
  return -a.x * b.x + dot(a.yzw, b.yzw) + scalar * la * lb;
}

fn coordinate_velocity(position: vec3<f32>, momentum: vec4<f32>, spin: f32, mass: f32) -> vec4<f32> {
  let r = ks_radius(position, spin);
  let scalar = ks_scalar(position, spin, mass);
  let l = ks_l(position, spin, r);
  let projected = -momentum.x + dot(l, momentum.yzw);
  return vec4<f32>(
    -momentum.x + scalar * projected,
    momentum.y - scalar * projected * l.x,
    momentum.z - scalar * projected * l.y,
    momentum.w - scalar * projected * l.z
  );
}

fn radial_coordinate_speed(position: vec3<f32>, momentum: vec4<f32>, spin: f32, mass: f32) -> f32 {
  let velocity = coordinate_velocity(position, momentum, spin, mass);
  let radius = max(ks_radius(position, spin), 1.0e-8);
  return dot(position, velocity.yzw) / radius;
}

fn gradient(position: vec3<f32>, momentum: vec4<f32>, spin: f32, mass: f32) -> vec3<f32> {
  let radius = max(ks_radius(position, spin), 1.0e-8);
  let scalar = ks_scalar(position, spin, mass);
  let radius_grad = ks_radius_gradient(position, spin, radius);
  let scalar_grad = ks_scalar_gradient(position, spin, mass, radius, radius_grad);
  let spatial_momentum = momentum.yzw;
  let l = ks_l(position, spin, radius);
  let projected = -momentum.x + dot(l, spatial_momentum);
  let projected_grad = vec3<f32>(
    ks_l_derivative_dot(position, spin, radius, radius_grad.x, vec3<f32>(1.0, 0.0, 0.0), spatial_momentum),
    ks_l_derivative_dot(position, spin, radius, radius_grad.y, vec3<f32>(0.0, 1.0, 0.0), spatial_momentum),
    ks_l_derivative_dot(position, spin, radius, radius_grad.z, vec3<f32>(0.0, 0.0, 1.0), spatial_momentum)
  );
  return -0.5 * (
    scalar_grad * (projected * projected) +
    projected_grad * (2.0 * scalar * projected)
  );
}

fn derivative(position: vec4<f32>, momentum: vec4<f32>, spin: f32, mass: f32) -> StepOutput {
  let v = coordinate_velocity(position.yzw, momentum, spin, mass);
  let g = gradient(position.yzw, momentum, spin, mass);
  return StepOutput(v, vec4<f32>(0.0, -g.x, -g.y, -g.z));
}

fn add_scaled(a: StepOutput, b: StepOutput, scale: f32) -> StepOutput {
  return StepOutput(a.position + b.position * scale, a.momentum + b.momentum * scale);
}

fn rk4(position: vec4<f32>, momentum: vec4<f32>, spin: f32, mass: f32, step: f32) -> StepOutput {
  let s0 = StepOutput(position, momentum);
  let k1 = derivative(s0.position, s0.momentum, spin, mass);
  let s1 = add_scaled(s0, k1, step * 0.5);
  let k2 = derivative(s1.position, s1.momentum, spin, mass);
  let s2 = add_scaled(s0, k2, step * 0.5);
  let k3 = derivative(s2.position, s2.momentum, spin, mass);
  let s3 = add_scaled(s0, k3, step);
  let k4 = derivative(s3.position, s3.momentum, spin, mass);
  return StepOutput(
    position + step / 6.0 * (k1.position + 2.0 * k2.position + 2.0 * k3.position + k4.position),
    momentum + step / 6.0 * (k1.momentum + 2.0 * k2.momentum + 2.0 * k3.momentum + k4.momentum)
  );
}

fn normalize_timelike(position: vec3<f32>, spin: f32, mass: f32, vector: vec4<f32>) -> vec4<f32> {
  let norm = metric_dot(position, spin, mass, vector, vector);
  var out = vector / sqrt(max(-norm, 1.0e-8));
  if (out.x < 0.0) {
    out = -out;
  }
  return out;
}

fn disk_emitter_four_velocity(position: vec3<f32>, spin: f32, mass: f32, spin_direction: f32) -> vec4<f32> {
  let radius = max(ks_radius(position, spin), 1.0e-6);
  let radial_len = max(length(vec2<f32>(position.x, position.y)), 1.0e-8);
  let radial = vec3<f32>(position.x / radial_len, position.y / radial_len, 0.0);
  let tangent = vec3<f32>(-spin_direction * radial.y, spin_direction * radial.x, 0.0);
  var beta_magnitude = min(sqrt(mass / max(radius - 2.0 * mass, 1.5)), 0.75);

  for (var i = 0u; i < 12u; i = i + 1u) {
    let beta = tangent * beta_magnitude;
    let gamma = 1.0 / sqrt(max(1.0 - dot(beta, beta), 1.0e-6));
    let flat_guess = vec4<f32>(gamma, gamma * beta.x, gamma * beta.y, gamma * beta.z);
    if (metric_dot(position, spin, mass, flat_guess, flat_guess) < -1.0e-8) {
      return normalize_timelike(position, spin, mass, flat_guess);
    }
    beta_magnitude = beta_magnitude * 0.75;
  }

  return normalize_timelike(position, spin, mass, vec4<f32>(1.0, 0.0, 0.0, 0.0));
}

fn redshift_factor(momentum: vec4<f32>, emitter_velocity: vec4<f32>, observer_velocity: vec4<f32>) -> f32 {
  let emitter_frequency = -dot(momentum, emitter_velocity);
  let observer_frequency = select(-momentum.x, -dot(momentum, observer_velocity), observer_velocity.x > 0.0);
  if (emitter_frequency <= 0.0 || observer_frequency <= 0.0) {
    return 0.0;
  }
  return observer_frequency / emitter_frequency;
}

fn clamp01(value: f32) -> f32 {
  return clamp(value, 0.0, 1.0);
}

fn blackbody_rgb(temperature: f32) -> vec3<f32> {
  let t = clamp(temperature, 1000.0, 40000.0) / 100.0;
  let red = select(clamp01(1.292936186 * pow(t - 60.0, -0.1332047592)), 1.0, t <= 66.0);
  let green = select(
    clamp01(1.129890861 * pow(t - 60.0, -0.0755148492)),
    clamp01(0.3900815788 * log(t) - 0.6318414438),
    t <= 66.0
  );
  let blue = select(select(clamp01(0.5432067891 * log(t - 10.0) - 1.1962540891), 0.0, t <= 19.0), 1.0, t >= 66.0);
  return vec3<f32>(red, green, blue);
}

fn add_star(color: vec3<f32>, direction: vec3<f32>, star_direction: vec3<f32>, sharpness: f32, tint: vec3<f32>) -> vec3<f32> {
  let strength = pow(max(dot(direction, normalize(star_direction)), 0.0), sharpness);
  return color + tint * strength;
}

fn hash2(p: vec2<f32>) -> f32 {
  return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453123);
}

fn value_noise(p: vec2<f32>) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let a = hash2(i);
  let b = hash2(i + vec2<f32>(1.0, 0.0));
  let c = hash2(i + vec2<f32>(0.0, 1.0));
  let d = hash2(i + vec2<f32>(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn procedural_stars(direction: vec3<f32>) -> vec3<f32> {
  let dir = normalize(direction);
  let uv = vec2<f32>(
    atan2(dir.y, dir.x) / 6.28318530718 + 0.5,
    asin(clamp(dir.z, -1.0, 1.0)) / 3.14159265359 + 0.5
  );
  let grid = vec2<f32>(420.0, 210.0);
  let p = uv * grid;
  let cell = floor(p);
  let local = fract(p) - vec2<f32>(0.5);
  let rnd = hash2(cell);
  let small_star = smoothstep(0.065, 0.0, length(local)) * step(0.992, rnd);
  let bright_star = smoothstep(0.12, 0.0, length(local)) * step(0.9985, rnd);
  let warm = 0.65 + 0.35 * hash2(cell + vec2<f32>(17.0, 43.0));
  return small_star * vec3<f32>(0.65, 0.78, 1.0) * 0.65 +
    bright_star * vec3<f32>(1.25, warm, 0.72) * 2.4;
}

fn legacy_background_color(direction: vec3<f32>) -> vec3<f32> {
  let dir = normalize(direction);
  let band_coordinate = abs(0.58 * dir.z + 0.34 * dir.y - 0.18 * dir.x);
  let band_noise = 0.45 + 0.55 * value_noise(vec2<f32>(atan2(dir.y, dir.x) * 2.4, dir.z * 5.5));
  let band = pow(max(1.0 - band_coordinate * 5.6, 0.0), 2.15) * band_noise;
  var color = vec3<f32>(
    0.0025 + 0.006 * max(dir.z, 0.0),
    0.004 + 0.006 * max(dir.y, 0.0),
    0.011 + 0.018 * max(-dir.z, 0.0)
  ) + vec3<f32>(0.055, 0.047, 0.035) * band;
  color = color + procedural_stars(dir);
  color = add_star(color, dir, vec3<f32>(0.42, 0.16, 0.89), 220.0, vec3<f32>(1.9, 1.65, 1.2));
  color = add_star(color, dir, vec3<f32>(-0.68, -0.08, 0.73), 180.0, vec3<f32>(1.15, 1.35, 1.9));
  color = add_star(color, dir, vec3<f32>(0.09, 0.82, -0.56), 260.0, vec3<f32>(1.7, 1.45, 1.05));
  color = add_star(color, dir, vec3<f32>(-0.24, -0.74, -0.63), 200.0, vec3<f32>(1.25, 1.55, 1.8));
  return color;
}

fn observed_disk_rgb(position: vec3<f32>, momentum: vec4<f32>, observer_velocity: vec4<f32>, input: RaySample) -> vec3<f32> {
  let spin = input.paramsA.x;
  let mass = input.paramsB.y;
  let inner_radius = input.paramsC.x;
  let inner_temperature = input.paramsC.z;
  let emissivity_scale = input.paramsC.w;
  let boost_power = input.paramsD.x;
  let spin_direction = input.paramsD.y;
  let emission_phase = input.paramsD.z;
  let radius = ks_radius(position, spin);
  let emitter_velocity = disk_emitter_four_velocity(position, spin, mass, spin_direction);
  let redshift = redshift_factor(momentum, emitter_velocity, observer_velocity);
  let temperature = inner_temperature * pow(radius / inner_radius, -0.75);
  let observed_temperature = max(redshift, 0.0) * temperature;
  let observed_rgb = blackbody_rgb(observed_temperature);
  let radial_falloff = pow(radius / inner_radius, -2.4);
  let azimuth = atan2(position.y, position.x);
  let advected_azimuth = azimuth - spin_direction * emission_phase;
  let spiral = cos(6.0 * advected_azimuth + 1.35 * log(max(radius, 1.0e-4)));
  let texture = 0.65 + 0.35 * (0.5 + 0.5 * spiral);
  let redshift_weight = select(0.0, pow(redshift, boost_power), redshift > 0.0);
  let bolometric = emissivity_scale * texture * radial_falloff * redshift_weight;
  return observed_rgb * bolometric;
}

fn disk_scale_height(radius: f32) -> f32 {
  return max(0.025, 0.015 * max(radius, 0.0));
}

fn adaptive_trace_step(
  position: vec3<f32>,
  momentum: vec4<f32>,
  input: RaySample,
  radius: f32,
  horizon: f32
) -> f32 {
  let base_step = input.paramsA.y;
  let spin = input.paramsA.x;
  let mass = input.paramsB.y;
  let singularity_radius = input.paramsA.w;
  var step = base_step * clamp(radius, 0.8, 18.0);
  step = min(step, 0.55);

  if (horizon > 0.0) {
    let horizon_distance = abs(radius - horizon);
    let horizon_step = base_step + 0.32 * max(horizon_distance, 0.015);
    step = min(step, horizon_step);
  }

  let singularity_distance = max(radius - singularity_radius, 0.01);
  step = min(step, max(base_step * 0.5, 0.22 * singularity_distance));

  let inner_radius = input.paramsC.x;
  let outer_radius = input.paramsC.y;
  if (radius >= inner_radius && radius <= outer_radius) {
    let velocity = coordinate_velocity(position, momentum, spin, mass).yzw;
    let vertical_speed = max(abs(velocity.z), 0.08);
    let scale_height = disk_scale_height(radius);
    let disk_distance = abs(position.z);
    let disk_step = 0.42 * max(scale_height + disk_distance, scale_height) / vertical_speed;
    step = min(step, clamp(disk_step, base_step * 0.75, 0.18));
  }

  return clamp(step, base_step * 0.5, 0.55);
}

fn disk_volume_emission(
  position: vec3<f32>,
  momentum: vec4<f32>,
  observer_velocity: vec4<f32>,
  input: RaySample,
  path_step_size: f32
) -> VolumeEmission {
  let spin = input.paramsA.x;
  let inner_radius = input.paramsC.x;
  let outer_radius = input.paramsC.y;
  let radius = ks_radius(position, spin);
  if (radius < inner_radius || radius > outer_radius) {
    return VolumeEmission(vec3<f32>(0.0), 0.0, -1.0);
  }

  let scale_height = disk_scale_height(radius);
  let vertical = position.z / scale_height;
  let density = exp(-0.5 * vertical * vertical);
  if (density < 1.0e-5) {
    return VolumeEmission(vec3<f32>(0.0), 0.0, -1.0);
  }

  let path_weight = min(density * max(path_step_size, 0.0) / (2.50662827463 * scale_height), 0.018);
  let color = observed_disk_rgb(position, momentum, observer_velocity, input) * path_weight;
  return VolumeEmission(color, color.r + color.g + color.b, radius);
}

fn diagnostic_color(status: f32, position: vec3<f32>, momentum: vec4<f32>, spin: f32, mass: f32) -> vec3<f32> {
  if (status == 1.0) {
    return vec3<f32>(0.0, 0.0, 0.0);
  }
  if (status == 2.0) {
    return vec3<f32>(0.0, 0.0, 0.0);
  }
  let velocity = coordinate_velocity(position, momentum, spin, mass).yzw;
  let dir = normalize(velocity);
  return legacy_background_color(dir);
}

fn escaped_sky_direction(position: vec3<f32>, momentum: vec4<f32>, spin: f32, mass: f32) -> vec3<f32> {
  return normalize(coordinate_velocity(position, momentum, spin, mass).yzw);
}

fn finish_composite(
  status: f32,
  steps: u32,
  radius: f32,
  disk_radius: f32,
  readback_color: vec3<f32>,
  display_disk_color: vec3<f32>,
  max_drift: f32,
  sky_direction: vec3<f32>,
  sky_mix: f32
) -> CompositeOutput {
  return CompositeOutput(
    vec4<f32>(status, f32(steps), radius, disk_radius),
    vec4<f32>(readback_color, max_drift),
    vec4<f32>(sky_direction, sky_mix),
    vec4<f32>(display_disk_color, 0.0)
  );
}

fn trace_composite(input: RaySample) -> CompositeOutput {
  let spin = input.paramsA.x;
  let step_size = input.paramsA.y;
  let escape_radius = input.paramsA.z;
  let singularity_radius = input.paramsA.w;
  let max_steps = u32(input.paramsB.x);
  let mass = input.paramsB.y;
  let horizon = horizon_radius(mass, spin);
  let h0 = hamiltonian(input.position.yzw, input.momentum, spin, mass);
  let camera_inside_horizon = horizon > 0.0 && ks_radius(input.position.yzw, spin) <= horizon;
  var position = input.position;
  var momentum = input.momentum;
  var max_drift = 0.0;
  var accumulated_disk = vec3<f32>(0.0);
  var brightest_disk_weight = 0.0;
  var brightest_disk_radius = -1.0;

  for (var steps = 0u; steps < max_steps; steps = steps + 1u) {
    let radius = ks_radius(position.yzw, spin);
    if (radius <= singularity_radius) {
      let rgb = diagnostic_color(2.0, position.yzw, momentum, spin, mass);
      if (brightest_disk_weight > 1.0e-8) {
        return finish_composite(4.0, steps, radius, brightest_disk_radius, accumulated_disk + rgb, accumulated_disk + rgb, max_drift, vec3<f32>(0.0, 0.0, 1.0), 0.0);
      }
      return finish_composite(2.0, steps, radius, -1.0, rgb, rgb, max_drift, vec3<f32>(0.0, 0.0, 1.0), 0.0);
    }
    if (!camera_inside_horizon && horizon > 0.0 && radius <= horizon) {
      let rgb = diagnostic_color(1.0, position.yzw, momentum, spin, mass);
      if (brightest_disk_weight > 1.0e-8) {
        return finish_composite(4.0, steps, radius, brightest_disk_radius, accumulated_disk + rgb, accumulated_disk + rgb, max_drift, vec3<f32>(0.0, 0.0, 1.0), 0.0);
      }
      return finish_composite(1.0, steps, radius, -1.0, rgb, rgb, max_drift, vec3<f32>(0.0, 0.0, 1.0), 0.0);
    }
    if (radius >= escape_radius && radial_coordinate_speed(position.yzw, momentum, spin, mass) > 0.0) {
      let sky_direction = escaped_sky_direction(position.yzw, momentum, spin, mass);
      let sky_rgb = legacy_background_color(sky_direction);
      if (brightest_disk_weight > 1.0e-8) {
        return finish_composite(4.0, steps, radius, brightest_disk_radius, accumulated_disk + sky_rgb, accumulated_disk, max_drift, sky_direction, 1.0);
      }
      return finish_composite(0.0, steps, radius, -1.0, sky_rgb, vec3<f32>(0.0), max_drift, sky_direction, 1.0);
    }

    let adaptive_step = adaptive_trace_step(position.yzw, momentum, input, radius, horizon);

    if (steps % 2u == 0u) {
      let emission = disk_volume_emission(position.yzw, momentum, input.observerVelocity, input, adaptive_step * 2.0);
      if (emission.weight > 0.0) {
        accumulated_disk = accumulated_disk + emission.color;
        if (emission.weight > brightest_disk_weight) {
          brightest_disk_weight = emission.weight;
          brightest_disk_radius = emission.radius;
        }
      }
    }

    let next = rk4(position, momentum, spin, mass, adaptive_step);
    position = next.position;
    momentum = next.momentum;
    max_drift = max(max_drift, abs(hamiltonian(position.yzw, momentum, spin, mass) - h0));
  }

  let radius = ks_radius(position.yzw, spin);
  let sky_direction = escaped_sky_direction(position.yzw, momentum, spin, mass);
  let sky_rgb = legacy_background_color(sky_direction);
  if (brightest_disk_weight > 1.0e-8) {
    return finish_composite(4.0, max_steps, radius, brightest_disk_radius, accumulated_disk + sky_rgb, accumulated_disk, max_drift, sky_direction, 1.0);
  }
  return finish_composite(3.0, max_steps, radius, -1.0, sky_rgb, vec3<f32>(0.0), max_drift, sky_direction, 1.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= arrayLength(&outputData)) {
    return;
  }
  outputData[i] = trace_composite(samples[i]);
}
`;

const CAMERA_COMPOSITE_BINDINGS = `
struct CameraUniforms {
  position: vec4<f32>,
  observerVelocity: vec4<f32>,
  eTime: vec4<f32>,
  eRight: vec4<f32>,
  eUp: vec4<f32>,
  eForward: vec4<f32>,
  paramsA: vec4<f32>,
  paramsB: vec4<f32>,
  paramsC: vec4<f32>,
  paramsD: vec4<f32>,
  reservedA: vec4<f32>,
  reservedB: vec4<f32>,
  reservedC: vec4<f32>,
  reservedD: vec4<f32>,
  reservedE: vec4<f32>,
  reservedF: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<storage, read_write> outputData: array<CompositeOutput>;
`;

const CAMERA_COMPOSITE_MAIN = `
fn lower_vector(position: vec3<f32>, spin: f32, mass: f32, vector: vec4<f32>) -> vec4<f32> {
  let r = ks_radius(position, spin);
  let scalar = ks_scalar(position, spin, mass);
  let l = ks_l(position, spin, r);
  let projected = vector.x + dot(l, vector.yzw);
  return vec4<f32>(
    -vector.x + scalar * projected,
    vector.y + scalar * projected * l.x,
    vector.z + scalar * projected * l.y,
    vector.w + scalar * projected * l.z
  );
}

fn camera_local_direction(pixel_index: u32) -> vec3<f32> {
  let width = max(u32(camera.paramsB.z), 1u);
  let height = max(u32(camera.paramsB.w), 1u);
  let x = f32(pixel_index % width);
  let y = f32(pixel_index / width);
  let ndc_x = (2.0 * (x + 0.5) / f32(width) - 1.0) * camera.paramsD.w;
  let ndc_y = 1.0 - 2.0 * (y + 0.5) / f32(height);
  return normalize(vec3<f32>(ndc_x * camera.paramsD.z, ndc_y * camera.paramsD.z, 1.0));
}

fn camera_ray_sample(pixel_index: u32) -> RaySample {
  let direction = camera_local_direction(pixel_index);
  let contravariant =
    camera.eTime +
    camera.eRight * direction.x +
    camera.eUp * direction.y +
    camera.eForward * direction.z;
  let momentum = lower_vector(camera.position.yzw, camera.paramsA.x, camera.paramsB.y, contravariant);
  let trace_params_b = vec4<f32>(camera.paramsB.x, camera.paramsB.y, 0.0, 0.0);
  let trace_params_d = vec4<f32>(camera.paramsD.x, camera.paramsD.y, camera.reservedA.x, 0.0);
  return RaySample(
    camera.position,
    momentum,
    camera.observerVelocity,
    camera.paramsA,
    trace_params_b,
    camera.paramsC,
    trace_params_d
  );
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= arrayLength(&outputData)) {
    return;
  }
  outputData[i] = trace_composite(camera_ray_sample(i));
}
`;

const CAMERA_COMPOSITE_SHADER = COMPOSITE_SHADER
  .replace(
    '@group(0) @binding(0) var<storage, read> samples: array<RaySample>;\n@group(0) @binding(1) var<storage, read_write> outputData: array<CompositeOutput>;',
    CAMERA_COMPOSITE_BINDINGS.trim(),
  )
  .replace(
    `@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= arrayLength(&outputData)) {
    return;
  }
  outputData[i] = trace_composite(samples[i]);
}`,
    CAMERA_COMPOSITE_MAIN.trim(),
  );

const FLOAT_BYTES = Float32Array.BYTES_PER_ELEMENT;

interface PipelineCache {
  camera?: GPUComputePipeline;
  composite?: GPUComputePipeline;
  cameraComposite?: GPUComputePipeline;
  presentByFormat: Map<GPUTextureFormat, GPURenderPipeline>;
}

interface CanvasRenderResources {
  device: GPUDevice;
  format: GPUTextureFormat;
  width: number;
  height: number;
  rayCount: number;
  outputByteLength: number;
  context: GPUCanvasContext;
  uniformBuffer: GPUBuffer;
  renderUniformBuffer: GPUBuffer;
  output: GPUBuffer;
  readback?: GPUBuffer;
  compositeBindGroup: GPUBindGroup;
  presentBindGroup: GPUBindGroup;
}

let cachedDevicePromise: Promise<GPUDevice | null> | undefined;
const pipelineCaches = new WeakMap<GPUDevice, PipelineCache>();
const canvasRenderCaches = new WeakMap<HTMLCanvasElement, CanvasRenderResources>();

const PRESENT_SHADER = `
struct CompositeOutput {
  summary: vec4<f32>,
  color: vec4<f32>,
  sky: vec4<f32>,
  disk: vec4<f32>,
};

struct RenderUniforms {
  size: vec4<f32>,
};

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> outputData: array<CompositeOutput>;
@group(0) @binding(1) var<uniform> render: RenderUniforms;

@vertex
fn vertex_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0)
  );
  var out: VertexOutput;
  out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
  return out;
}

fn hash2(p: vec2<f32>) -> f32 {
  return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453123);
}

fn hash3(p: vec3<f32>) -> vec3<f32> {
  let q = vec3<f32>(
    dot(p, vec3<f32>(127.1, 311.7, 74.7)),
    dot(p, vec3<f32>(269.5, 183.3, 246.1)),
    dot(p, vec3<f32>(113.5, 271.9, 124.6))
  );
  return fract(sin(q) * 43758.5453123);
}

fn value_noise(p: vec2<f32>) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let a = hash2(i);
  let b = hash2(i + vec2<f32>(1.0, 0.0));
  let c = hash2(i + vec2<f32>(0.0, 1.0));
  let d = hash2(i + vec2<f32>(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm(p0: vec2<f32>) -> f32 {
  var p = p0;
  var amp = 0.5;
  var sum = 0.0;
  for (var i = 0u; i < 5u; i = i + 1u) {
    sum = sum + amp * value_noise(p);
    p = p * 2.07 + vec2<f32>(13.7, 7.1);
    amp = amp * 0.5;
  }
  return sum;
}

fn blackbody_rgb_fast(temperature: f32) -> vec3<f32> {
  let t = max(temperature, 400.0);
  var color = vec3<f32>(
    56100000.0 * pow(t, -1.5) + 148.0,
    select(100.04 * log(t) - 623.6, 35200000.0 * pow(t, -1.5) + 184.0, t > 6500.0),
    194.18 * log(t) - 1448.6
  );
  color = clamp(color, vec3<f32>(0.0), vec3<f32>(255.0)) / 255.0;
  if (t < 1000.0) {
    color = color * (t / 1000.0);
  }
  return color;
}

fn detailed_sky(direction: vec3<f32>) -> vec3<f32> {
  let d = normalize(direction);
  var color = vec3<f32>(0.0);

  for (var layer = 0u; layer < 2u; layer = layer + 1u) {
    let scale = select(22.0, 47.0, layer == 1u);
    let q = d * scale;
    let id = floor(q);
    let h = hash3(id);
    let star_position = id + 0.2 + 0.6 * h;
    let dist = length(q - star_position);
    let lit = step(0.82, hash3(id + vec3<f32>(17.0)).x);
    let core = exp(-dist * dist * 220.0);
    let temp = mix(2800.0, 14000.0, h.y * h.y);
    let mag = 0.3 + 2.2 * h.z * h.z;
    color = color + lit * core * mag * blackbody_rgb_fast(temp);
  }

  let band_normal = normalize(vec3<f32>(0.35, 0.2, 1.0));
  let band = exp(-pow(dot(d, band_normal) * 3.2, 2.0));
  let neb_uv1 = vec2<f32>(d.x + 0.37 * d.z, d.y - 0.21 * d.x) * 5.0;
  let neb_uv2 = vec2<f32>(d.z - 0.31 * d.x, d.y + 0.27 * d.z) * 5.0;
  let neb = 0.5 * fbm(neb_uv1 + vec2<f32>(3.7)) + 0.5 * fbm(neb_uv2 + vec2<f32>(11.1));
  color = color + band * (0.012 + 0.05 * neb * neb) * vec3<f32>(0.55, 0.62, 0.85);
  return color;
}

fn aces(color: vec3<f32>) -> vec3<f32> {
  return clamp((color * (2.51 * color + vec3<f32>(0.03))) / (color * (2.43 * color + vec3<f32>(0.59)) + vec3<f32>(0.14)), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fragment_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
  let width = max(u32(render.size.x), 1u);
  let height = max(u32(render.size.y), 1u);
  let x = min(u32(position.x), width - 1u);
  let y = min(u32(position.y), height - 1u);
  let index = y * width + x;
  let sample = outputData[index];
  let sky_direction = normalize(select(vec3<f32>(0.0, 0.0, 1.0), sample.sky.xyz, sample.sky.w > 0.5));
  let sky = detailed_sky(sky_direction) * sample.sky.w;
  let rgb = aces((sky + sample.disk.xyz) * 1.35);
  return vec4<f32>(pow(rgb, vec3<f32>(1.0 / 2.2)), 1.0);
}
`;

export async function runWebGpuCompositeProbe(
  samples: Float32Array<ArrayBufferLike>,
  expected: Float32Array<ArrayBufferLike>,
): Promise<WebGpuCompositeProbeResult> {
  const expectedCopy = new Float32Array(expected);
  const raw = await runWebGpuComposite(samples);
  if (!raw.supported || !raw.output) {
    return { supported: raw.supported, message: raw.message };
  }
  const comparison = compareCompositeReadback(expectedCopy, raw.output);

  return {
    supported: true,
    message: 'WebGPU composite probe matched CPU reference',
    output: raw.output,
    maxAbsDiff: comparison.maxAbsDiff,
    statusMismatches: comparison.statusMismatches,
    diskMismatches: comparison.diskMismatches,
  };
}

export async function runWebGpuComposite(
  samples: Float32Array<ArrayBufferLike>,
): Promise<WebGpuCompositeRunResult> {
  const constants = globalThis as typeof globalThis & WebGpuConstants;
  const usage = constants.GPUBufferUsage;
  const mapMode = constants.GPUMapMode;
  if (!navigator.gpu || !usage || !mapMode) {
    return { supported: false, message: 'WebGPU globals are unavailable' };
  }

  const inputCopy = new Float32Array(samples);
  const rayCount = compositeRayCount(inputCopy);
  if (!Number.isInteger(rayCount)) {
    return { supported: false, message: 'Composite sample buffer has an invalid ray stride' };
  }
  const outputByteLength = compositeOutputByteLength(inputCopy);
  const device = await requestCachedDevice();
  if (!device) return { supported: false, message: 'No WebGPU adapter available' };

  const source = device.createBuffer({
    size: inputCopy.byteLength,
    usage: usage.STORAGE | usage.COPY_DST,
  });
  const output = device.createBuffer({
    size: outputByteLength,
    usage: usage.STORAGE | usage.COPY_SRC,
  });
  const readback = device.createBuffer({
    size: outputByteLength,
    usage: usage.COPY_DST | usage.MAP_READ,
  });

  device.queue.writeBuffer(source, 0, inputCopy);

  const pipeline = getCompositePipeline(device);
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: source } },
      { binding: 1, resource: { buffer: output } },
    ],
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(rayCount / 64));
  pass.end();
  encoder.copyBufferToBuffer(output, 0, readback, 0, outputByteLength);
  device.queue.submit([encoder.finish()]);

  await readback.mapAsync(mapMode.READ);
  const outputCopy = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();

  return {
    supported: true,
    message: 'WebGPU composite renderer completed',
    output: outputCopy,
  };
}

export async function runWebGpuCompositeFromCamera(
  options: CompositeCameraSampleOptions,
): Promise<WebGpuCompositeRunResult> {
  const constants = globalThis as typeof globalThis & WebGpuConstants;
  const usage = constants.GPUBufferUsage;
  const mapMode = constants.GPUMapMode;
  if (!navigator.gpu || !usage || !mapMode) {
    return { supported: false, message: 'WebGPU globals are unavailable' };
  }

  const uniforms = createCompositeCameraUniforms(options);
  const rayCount = options.width * options.height;
  const sampleByteLength = rayCount * COMPOSITE_INPUT_FLOATS_PER_RAY * FLOAT_BYTES;
  const outputByteLength = rayCount * COMPOSITE_OUTPUT_FLOATS_PER_RAY * FLOAT_BYTES;
  const device = await requestCachedDevice();
  if (!device) return { supported: false, message: 'No WebGPU adapter available' };

  const uniformBuffer = device.createBuffer({
    size: COMPOSITE_CAMERA_UNIFORM_FLOATS * FLOAT_BYTES,
    usage: usage.UNIFORM | usage.COPY_DST,
  });
  const samples = device.createBuffer({
    size: sampleByteLength,
    usage: usage.STORAGE,
  });
  const output = device.createBuffer({
    size: outputByteLength,
    usage: usage.STORAGE | usage.COPY_SRC,
  });
  const readback = device.createBuffer({
    size: outputByteLength,
    usage: usage.COPY_DST | usage.MAP_READ,
  });

  device.queue.writeBuffer(uniformBuffer, 0, uniforms);

  const cameraPipeline = getCameraPipeline(device);
  const cameraBindGroup = device.createBindGroup({
    layout: cameraPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: samples } },
    ],
  });

  const compositePipeline = getCompositePipeline(device);
  const compositeBindGroup = device.createBindGroup({
    layout: compositePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: samples } },
      { binding: 1, resource: { buffer: output } },
    ],
  });

  const encoder = device.createCommandEncoder();
  const cameraPass = encoder.beginComputePass();
  cameraPass.setPipeline(cameraPipeline);
  cameraPass.setBindGroup(0, cameraBindGroup);
  cameraPass.dispatchWorkgroups(Math.ceil(rayCount / 64));
  cameraPass.end();

  const compositePass = encoder.beginComputePass();
  compositePass.setPipeline(compositePipeline);
  compositePass.setBindGroup(0, compositeBindGroup);
  compositePass.dispatchWorkgroups(Math.ceil(rayCount / 64));
  compositePass.end();

  encoder.copyBufferToBuffer(output, 0, readback, 0, outputByteLength);
  device.queue.submit([encoder.finish()]);

  await readback.mapAsync(mapMode.READ);
  const outputCopy = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();

  return {
    supported: true,
    message: 'WebGPU camera-generated composite renderer completed',
    output: outputCopy,
  };
}

export async function renderWebGpuCompositeFromCameraToCanvas(
  options: CompositeCameraSampleOptions,
  canvas: HTMLCanvasElement,
  canvasOptions: WebGpuCompositeCanvasOptions = {},
): Promise<WebGpuCompositeRunResult> {
  const constants = globalThis as typeof globalThis & WebGpuConstants;
  const usage = constants.GPUBufferUsage;
  const textureUsage = constants.GPUTextureUsage;
  const shouldReadBack = canvasOptions.readback ?? false;
  const mapMode = constants.GPUMapMode;
  if (!navigator.gpu || !usage || !textureUsage || (shouldReadBack && !mapMode)) {
    return { supported: false, message: 'WebGPU globals are unavailable' };
  }

  const uniforms = createCompositeCameraUniforms(options);
  const device = await requestCachedDevice();
  if (!device) return { supported: false, message: 'No WebGPU adapter available' };

  const format = navigator.gpu.getPreferredCanvasFormat();
  const limitCheck = validateCanvasRenderLimits(device, options);
  if (limitCheck) return { supported: false, message: limitCheck };

  const resources = getCanvasRenderResources(canvas, device, format, options, usage, textureUsage);
  if (!resources) return { supported: false, message: 'Canvas WebGPU context is unavailable' };

  device.queue.writeBuffer(resources.uniformBuffer, 0, uniforms);
  device.queue.writeBuffer(resources.renderUniformBuffer, 0, new Float32Array([options.width, options.height, 0, 0]));

  const compositePipeline = getCameraCompositePipeline(device);
  const presentPipeline = getPresentPipeline(device, format);

  const encoder = device.createCommandEncoder();
  const compositePass = encoder.beginComputePass();
  compositePass.setPipeline(compositePipeline);
  compositePass.setBindGroup(0, resources.compositeBindGroup);
  compositePass.dispatchWorkgroups(Math.ceil(resources.rayCount / 64));
  compositePass.end();

  const renderPass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: resources.context.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      },
    ],
  });
  renderPass.setPipeline(presentPipeline);
  renderPass.setBindGroup(0, resources.presentBindGroup);
  renderPass.draw(3);
  renderPass.end();

  if (shouldReadBack) {
    const readback = getCanvasReadbackBuffer(resources, usage);
    encoder.copyBufferToBuffer(resources.output, 0, readback, 0, resources.outputByteLength);
  }
  device.queue.submit([encoder.finish()]);

  if (!shouldReadBack) {
    return {
      supported: true,
      message: 'WebGPU camera-generated composite rendered to canvas',
    };
  }

  const readback = resources.readback;
  if (!readback || !mapMode) {
    return { supported: false, message: 'WebGPU readback buffer is unavailable' };
  }

  await readback.mapAsync(mapMode.READ);
  const outputCopy = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();

  return {
    supported: true,
    message: 'WebGPU camera-generated composite rendered to canvas',
    output: outputCopy,
  };
}

function getCanvasRenderResources(
  canvas: HTMLCanvasElement,
  device: GPUDevice,
  format: GPUTextureFormat,
  options: CompositeCameraSampleOptions,
  usage: NonNullable<WebGpuConstants['GPUBufferUsage']>,
  textureUsage: NonNullable<WebGpuConstants['GPUTextureUsage']>,
): CanvasRenderResources | null {
  const cached = canvasRenderCaches.get(canvas);
  if (
    cached &&
    cached.device === device &&
    cached.format === format &&
    cached.width === options.width &&
    cached.height === options.height
  ) {
    return cached;
  }

  const context = canvas.getContext('webgpu') as GPUCanvasContext | null;
  if (!context) return null;
  if (canvas.width !== options.width) canvas.width = options.width;
  if (canvas.height !== options.height) canvas.height = options.height;
  context.configure({
    device,
    format,
    usage: textureUsage.RENDER_ATTACHMENT,
    alphaMode: 'opaque',
  });

  const rayCount = options.width * options.height;
  const outputByteLength = rayCount * COMPOSITE_OUTPUT_FLOATS_PER_RAY * FLOAT_BYTES;
  const uniformBuffer = device.createBuffer({
    size: COMPOSITE_CAMERA_UNIFORM_FLOATS * FLOAT_BYTES,
    usage: usage.UNIFORM | usage.COPY_DST,
  });
  const renderUniformBuffer = device.createBuffer({
    size: 4 * FLOAT_BYTES,
    usage: usage.UNIFORM | usage.COPY_DST,
  });
  const output = device.createBuffer({
    size: outputByteLength,
    usage: usage.STORAGE | usage.COPY_SRC,
  });
  const compositePipeline = getCameraCompositePipeline(device);
  const presentPipeline = getPresentPipeline(device, format);
  const resources: CanvasRenderResources = {
    device,
    format,
    width: options.width,
    height: options.height,
    rayCount,
    outputByteLength,
    context,
    uniformBuffer,
    renderUniformBuffer,
    output,
    compositeBindGroup: device.createBindGroup({
      layout: compositePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: output } },
      ],
    }),
    presentBindGroup: device.createBindGroup({
      layout: presentPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: output } },
        { binding: 1, resource: { buffer: renderUniformBuffer } },
      ],
    }),
  };
  canvasRenderCaches.set(canvas, resources);
  return resources;
}

function validateCanvasRenderLimits(device: GPUDevice, options: CompositeCameraSampleOptions): string | null {
  const rayCount = options.width * options.height;
  const outputByteLength = rayCount * COMPOSITE_OUTPUT_FLOATS_PER_RAY * FLOAT_BYTES;
  const storageLimit = device.limits.maxStorageBufferBindingSize;
  const bufferLimit = device.limits.maxBufferSize;

  if (outputByteLength > bufferLimit) {
    return `render size ${options.width}x${options.height} needs ${formatBytes(outputByteLength)}, above this GPU buffer limit ${formatBytes(bufferLimit)}`;
  }
  if (outputByteLength > storageLimit) {
    return `render size ${options.width}x${options.height} needs storage binding ${formatBytes(outputByteLength)}, above this GPU limit ${formatBytes(storageLimit)}`;
  }
  return null;
}

function formatBytes(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${bytes} B`;
}

function getCanvasReadbackBuffer(
  resources: CanvasRenderResources,
  usage: NonNullable<WebGpuConstants['GPUBufferUsage']>,
): GPUBuffer {
  if (!resources.readback) {
    resources.readback = resources.device.createBuffer({
      size: resources.outputByteLength,
      usage: usage.COPY_DST | usage.MAP_READ,
    });
  }
  return resources.readback;
}

async function requestCachedDevice(): Promise<GPUDevice | null> {
  if (!cachedDevicePromise) {
    cachedDevicePromise = navigator.gpu.requestAdapter().then(async (adapter) => {
      if (!adapter) return null;
      const requiredLimits: Record<string, number> = {};
      if (adapter.limits.maxStorageBufferBindingSize > 134_217_728) {
        requiredLimits.maxStorageBufferBindingSize = adapter.limits.maxStorageBufferBindingSize;
      }
      if (adapter.limits.maxBufferSize > 268_435_456) {
        requiredLimits.maxBufferSize = adapter.limits.maxBufferSize;
      }
      return adapter.requestDevice(
        Object.keys(requiredLimits).length > 0 ? { requiredLimits } : undefined,
      );
    });
  }
  return cachedDevicePromise;
}

function getPipelineCache(device: GPUDevice): PipelineCache {
  let cache = pipelineCaches.get(device);
  if (!cache) {
    cache = { presentByFormat: new Map() };
    pipelineCaches.set(device, cache);
  }
  return cache;
}

function getCameraPipeline(device: GPUDevice): GPUComputePipeline {
  const cache = getPipelineCache(device);
  if (!cache.camera) {
    const module = device.createShaderModule({ code: CAMERA_SAMPLE_SHADER });
    cache.camera = device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
  }
  return cache.camera;
}

function getCompositePipeline(device: GPUDevice): GPUComputePipeline {
  const cache = getPipelineCache(device);
  if (!cache.composite) {
    const module = device.createShaderModule({ code: COMPOSITE_SHADER });
    cache.composite = device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
  }
  return cache.composite;
}

function getCameraCompositePipeline(device: GPUDevice): GPUComputePipeline {
  const cache = getPipelineCache(device);
  if (!cache.cameraComposite) {
    const module = device.createShaderModule({ code: CAMERA_COMPOSITE_SHADER });
    cache.cameraComposite = device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
  }
  return cache.cameraComposite;
}

function getPresentPipeline(device: GPUDevice, format: GPUTextureFormat): GPURenderPipeline {
  const cache = getPipelineCache(device);
  const cached = cache.presentByFormat.get(format);
  if (cached) return cached;
  const module = device.createShaderModule({ code: PRESENT_SHADER });
  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module, entryPoint: 'vertex_main' },
    fragment: {
      module,
      entryPoint: 'fragment_main',
      targets: [{ format }],
    },
  });
  cache.presentByFormat.set(format, pipeline);
  return pipeline;
}
