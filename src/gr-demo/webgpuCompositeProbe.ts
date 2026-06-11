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
  let radius = max(ks_radius(position, spin), 1.0);
  let eps = 1.0e-5 * radius;
  let ex = vec3<f32>(eps, 0.0, 0.0);
  let ey = vec3<f32>(0.0, eps, 0.0);
  let ez = vec3<f32>(0.0, 0.0, eps);
  return vec3<f32>(
    (hamiltonian(position + ex, momentum, spin, mass) - hamiltonian(position - ex, momentum, spin, mass)) / (2.0 * eps),
    (hamiltonian(position + ey, momentum, spin, mass) - hamiltonian(position - ey, momentum, spin, mass)) / (2.0 * eps),
    (hamiltonian(position + ez, momentum, spin, mass) - hamiltonian(position - ez, momentum, spin, mass)) / (2.0 * eps)
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

fn escaped_background_color(direction: vec3<f32>) -> vec3<f32> {
  let dir = normalize(direction);
  let band_coordinate = abs(0.72 * dir.y + 0.24 * dir.x - 0.1 * dir.z);
  let band = pow(max(1.0 - band_coordinate * 4.5, 0.0), 2.0);
  var color = vec3<f32>(
    0.006 + 0.012 * max(dir.z, 0.0),
    0.008 + 0.01 * max(dir.y, 0.0),
    0.016 + 0.018 * max(-dir.z, 0.0)
  ) + vec3<f32>(0.035, 0.032, 0.048) * band;
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
  let emitted_rgb = blackbody_rgb(temperature);
  let radial_falloff = pow(radius / inner_radius, -2.4);
  let azimuth = atan2(position.y, position.x);
  let advected_azimuth = azimuth - spin_direction * emission_phase;
  let spiral = cos(6.0 * advected_azimuth + 1.35 * log(max(radius, 1.0e-4)));
  let texture = 0.65 + 0.35 * (0.5 + 0.5 * spiral);
  let bolometric = emissivity_scale * texture * radial_falloff * pow(max(redshift, 0.0), boost_power);
  return emitted_rgb * bolometric;
}

fn disk_crossing(start_position: vec4<f32>, start_momentum: vec4<f32>, input: RaySample) -> CompositeOutput {
  let spin = input.paramsA.x;
  let step_size = input.paramsA.y;
  let mass = input.paramsB.y;
  let inner_radius = input.paramsC.x;
  let outer_radius = input.paramsC.y;
  let h0 = start_position.w;
  let end = rk4(start_position, start_momentum, spin, mass, step_size);
  let h1 = end.position.w;

  if (abs(h0) < 1.0e-12) {
    let radius = ks_radius(start_position.yzw, spin);
    if (radius >= inner_radius && radius <= outer_radius) {
      let rgb = observed_disk_rgb(start_position.yzw, start_momentum, input.observerVelocity, input);
      return CompositeOutput(vec4<f32>(4.0, 0.0, radius, radius), vec4<f32>(rgb, 0.0));
    }
  }
  if (h0 * h1 > 0.0) {
    return CompositeOutput(vec4<f32>(-1.0, 0.0, 0.0, -1.0), vec4<f32>(0.0));
  }

  var lo = 0.0;
  var hi = step_size;
  var lo_height = h0;
  var best = end;
  for (var i = 0u; i < 18u; i = i + 1u) {
    let mid = 0.5 * (lo + hi);
    let state = rk4(start_position, start_momentum, spin, mass, mid);
    let height = state.position.w;
    best = state;
    if (abs(height) < 1.0e-12) {
      lo = mid;
      hi = mid;
      break;
    }
    if (lo_height * height <= 0.0) {
      hi = mid;
    } else {
      lo = mid;
      lo_height = height;
    }
  }

  let disk_radius = ks_radius(best.position.yzw, spin);
  if (disk_radius < inner_radius || disk_radius > outer_radius) {
    return CompositeOutput(vec4<f32>(-1.0, 0.0, 0.0, -1.0), vec4<f32>(0.0));
  }
  let rgb = observed_disk_rgb(best.position.yzw, best.momentum, input.observerVelocity, input);
  return CompositeOutput(vec4<f32>(4.0, 0.5 * (lo + hi), disk_radius, disk_radius), vec4<f32>(rgb, 0.0));
}

fn diagnostic_color(status: f32, position: vec3<f32>, momentum: vec4<f32>, spin: f32, mass: f32) -> vec3<f32> {
  if (status == 1.0) {
    return vec3<f32>(0.0, 0.0, 0.0);
  }
  if (status == 2.0) {
    return vec3<f32>(0.3, 0.0, 0.5);
  }
  if (status == 3.0) {
    return vec3<f32>(0.9, 0.1, 0.1);
  }
  let velocity = coordinate_velocity(position, momentum, spin, mass).yzw;
  let dir = normalize(velocity);
  return escaped_background_color(dir);
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
  var position = input.position;
  var momentum = input.momentum;
  var max_drift = 0.0;

  for (var steps = 0u; steps < max_steps; steps = steps + 1u) {
    let radius = ks_radius(position.yzw, spin);
    if (radius <= singularity_radius) {
      let rgb = diagnostic_color(2.0, position.yzw, momentum, spin, mass);
      return CompositeOutput(vec4<f32>(2.0, f32(steps), radius, -1.0), vec4<f32>(rgb, max_drift));
    }
    if (horizon > 0.0 && radius <= horizon) {
      let rgb = diagnostic_color(1.0, position.yzw, momentum, spin, mass);
      return CompositeOutput(vec4<f32>(1.0, f32(steps), radius, -1.0), vec4<f32>(rgb, max_drift));
    }
    if (radius >= escape_radius && radial_coordinate_speed(position.yzw, momentum, spin, mass) > 0.0) {
      let rgb = diagnostic_color(0.0, position.yzw, momentum, spin, mass);
      return CompositeOutput(vec4<f32>(0.0, f32(steps), radius, -1.0), vec4<f32>(rgb, max_drift));
    }

    let crossing = disk_crossing(position, momentum, input);
    if (crossing.summary.x == 4.0 && crossing.summary.y > 1.0e-7) {
      return CompositeOutput(vec4<f32>(4.0, f32(steps), crossing.summary.z, crossing.summary.w), vec4<f32>(crossing.color.xyz, max_drift));
    }

    let next = rk4(position, momentum, spin, mass, step_size);
    position = next.position;
    momentum = next.momentum;
    max_drift = max(max_drift, abs(hamiltonian(position.yzw, momentum, spin, mass) - h0));
  }

  let radius = ks_radius(position.yzw, spin);
  let rgb = diagnostic_color(3.0, position.yzw, momentum, spin, mass);
  return CompositeOutput(vec4<f32>(3.0, f32(max_steps), radius, -1.0), vec4<f32>(rgb, max_drift));
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

const FLOAT_BYTES = Float32Array.BYTES_PER_ELEMENT;

interface PipelineCache {
  camera?: GPUComputePipeline;
  composite?: GPUComputePipeline;
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
  samples: GPUBuffer;
  output: GPUBuffer;
  readback: GPUBuffer;
  cameraBindGroup: GPUBindGroup;
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

fn display_map(value: f32) -> f32 {
  let positive = max(value, 0.0);
  let mapped = positive / (1.0 + positive);
  return pow(mapped, 1.0 / 2.2);
}

@fragment
fn fragment_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
  let width = max(u32(render.size.x), 1u);
  let height = max(u32(render.size.y), 1u);
  let x = min(u32(position.x), width - 1u);
  let y = min(u32(position.y), height - 1u);
  let index = y * width + x;
  let rgb = outputData[index].color.xyz;
  return vec4<f32>(display_map(rgb.r), display_map(rgb.g), display_map(rgb.b), 1.0);
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
  const mapMode = constants.GPUMapMode;
  if (!navigator.gpu || !usage || !textureUsage || !mapMode) {
    return { supported: false, message: 'WebGPU globals are unavailable' };
  }

  const uniforms = createCompositeCameraUniforms(options);
  const device = await requestCachedDevice();
  if (!device) return { supported: false, message: 'No WebGPU adapter available' };

  const format = navigator.gpu.getPreferredCanvasFormat();
  const resources = getCanvasRenderResources(canvas, device, format, options, usage, textureUsage);
  if (!resources) return { supported: false, message: 'Canvas WebGPU context is unavailable' };

  device.queue.writeBuffer(resources.uniformBuffer, 0, uniforms);
  device.queue.writeBuffer(resources.renderUniformBuffer, 0, new Float32Array([options.width, options.height, 0, 0]));

  const cameraPipeline = getCameraPipeline(device);
  const compositePipeline = getCompositePipeline(device);
  const presentPipeline = getPresentPipeline(device, format);

  const encoder = device.createCommandEncoder();
  const cameraPass = encoder.beginComputePass();
  cameraPass.setPipeline(cameraPipeline);
  cameraPass.setBindGroup(0, resources.cameraBindGroup);
  cameraPass.dispatchWorkgroups(Math.ceil(resources.rayCount / 64));
  cameraPass.end();

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

  const shouldReadBack = canvasOptions.readback ?? true;
  if (shouldReadBack) {
    encoder.copyBufferToBuffer(resources.output, 0, resources.readback, 0, resources.outputByteLength);
  }
  device.queue.submit([encoder.finish()]);

  if (!shouldReadBack) {
    return {
      supported: true,
      message: 'WebGPU camera-generated composite rendered to canvas',
    };
  }

  await resources.readback.mapAsync(mapMode.READ);
  const outputCopy = new Float32Array(resources.readback.getMappedRange().slice(0));
  resources.readback.unmap();

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
  const sampleByteLength = rayCount * COMPOSITE_INPUT_FLOATS_PER_RAY * FLOAT_BYTES;
  const outputByteLength = rayCount * COMPOSITE_OUTPUT_FLOATS_PER_RAY * FLOAT_BYTES;
  const uniformBuffer = device.createBuffer({
    size: COMPOSITE_CAMERA_UNIFORM_FLOATS * FLOAT_BYTES,
    usage: usage.UNIFORM | usage.COPY_DST,
  });
  const renderUniformBuffer = device.createBuffer({
    size: 4 * FLOAT_BYTES,
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
  const cameraPipeline = getCameraPipeline(device);
  const compositePipeline = getCompositePipeline(device);
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
    samples,
    output,
    readback,
    cameraBindGroup: device.createBindGroup({
      layout: cameraPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: samples } },
      ],
    }),
    compositeBindGroup: device.createBindGroup({
      layout: compositePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: samples } },
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

async function requestCachedDevice(): Promise<GPUDevice | null> {
  if (!cachedDevicePromise) {
    cachedDevicePromise = navigator.gpu.requestAdapter().then((adapter) => adapter?.requestDevice() ?? null);
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
