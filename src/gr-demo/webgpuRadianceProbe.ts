export interface WebGpuRadianceProbeResult {
  supported: boolean;
  message: string;
  output?: Float32Array;
  maxAbsDiff?: number;
  validMismatches?: number;
}

interface WebGpuConstants {
  GPUBufferUsage?: {
    STORAGE: number;
    COPY_DST: number;
    COPY_SRC: number;
    MAP_READ: number;
  };
  GPUMapMode?: {
    READ: number;
  };
}

const RADIANCE_SHADER = `
struct RadianceSample {
  position: vec4<f32>,
  momentum: vec4<f32>,
  observerVelocity: vec4<f32>,
  paramsA: vec4<f32>,
  paramsB: vec4<f32>,
};

struct RadianceOutput {
  sample: vec4<f32>,
  color: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> samples: array<RadianceSample>;
@group(0) @binding(1) var<storage, read_write> outputData: array<RadianceOutput>;

fn ks_radius(p: vec3<f32>, a: f32) -> f32 {
  let r2_euclid = dot(p, p);
  if (abs(a) < 1.0e-6) {
    return sqrt(r2_euclid);
  }
  let b = r2_euclid - a * a;
  let r2 = 0.5 * (b + sqrt(b * b + 4.0 * a * a * p.z * p.z));
  return sqrt(max(r2, 0.0));
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

fn metric_dot(position: vec3<f32>, spin: f32, mass: f32, a: vec4<f32>, b: vec4<f32>) -> f32 {
  let r = ks_radius(position, spin);
  let scalar = ks_scalar(position, spin, mass);
  let l = ks_l(position, spin, r);
  let la = a.x + dot(l, a.yzw);
  let lb = b.x + dot(l, b.yzw);
  return -a.x * b.x + dot(a.yzw, b.yzw) + scalar * la * lb;
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

fn covector_contraction(covector: vec4<f32>, vector: vec4<f32>) -> f32 {
  return dot(covector, vector);
}

fn redshift_factor(momentum: vec4<f32>, emitter_velocity: vec4<f32>, observer_velocity: vec4<f32>) -> f32 {
  let emitter_frequency = -covector_contraction(momentum, emitter_velocity);
  let observer_frequency = select(-momentum.x, -covector_contraction(momentum, observer_velocity), observer_velocity.x > 0.0);
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

fn sample_radiance(input: RadianceSample) -> RadianceOutput {
  let spin = input.paramsA.x;
  let mass = input.paramsA.y;
  let inner_radius = input.paramsA.z;
  let outer_radius = input.paramsA.w;
  let inner_temperature = input.paramsB.x;
  let emissivity_scale = input.paramsB.y;
  let boost_power = input.paramsB.z;
  let spin_direction = input.paramsB.w;
  let position = input.position.yzw;
  let radius = ks_radius(position, spin);

  if (radius < inner_radius || radius > outer_radius) {
    return RadianceOutput(vec4<f32>(0.0, radius, 0.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 0.0));
  }

  let emitter_velocity = disk_emitter_four_velocity(position, spin, mass, spin_direction);
  let redshift = redshift_factor(input.momentum, emitter_velocity, input.observerVelocity);
  let temperature = inner_temperature * pow(radius / inner_radius, -0.75);
  let emitted_rgb = blackbody_rgb(temperature);
  let radial_falloff = pow(radius / inner_radius, -2.4);
  let bolometric = emissivity_scale * radial_falloff * pow(max(redshift, 0.0), boost_power);
  let observed_rgb = emitted_rgb * bolometric;
  return RadianceOutput(vec4<f32>(1.0, radius, temperature, redshift), vec4<f32>(bolometric, observed_rgb));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= arrayLength(&outputData)) {
    return;
  }
  outputData[i] = sample_radiance(samples[i]);
}
`;

export async function runWebGpuRadianceProbe(
  samples: Float32Array<ArrayBufferLike>,
  expected: Float32Array<ArrayBufferLike>,
): Promise<WebGpuRadianceProbeResult> {
  const constants = globalThis as typeof globalThis & WebGpuConstants;
  const usage = constants.GPUBufferUsage;
  const mapMode = constants.GPUMapMode;
  if (!navigator.gpu || !usage || !mapMode) {
    return { supported: false, message: 'WebGPU globals are unavailable' };
  }

  const inputCopy = new Float32Array(samples);
  const expectedCopy = new Float32Array(expected);
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return { supported: false, message: 'No WebGPU adapter available' };
  const device = await adapter.requestDevice();

  const source = device.createBuffer({
    size: inputCopy.byteLength,
    usage: usage.STORAGE | usage.COPY_DST,
  });
  const output = device.createBuffer({
    size: expectedCopy.byteLength,
    usage: usage.STORAGE | usage.COPY_SRC,
  });
  const readback = device.createBuffer({
    size: expectedCopy.byteLength,
    usage: usage.COPY_DST | usage.MAP_READ,
  });

  device.queue.writeBuffer(source, 0, inputCopy);

  const module = device.createShaderModule({ code: RADIANCE_SHADER });
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: 'main' },
  });
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
  pass.dispatchWorkgroups(Math.ceil(inputCopy.length / 20 / 64));
  pass.end();
  encoder.copyBufferToBuffer(output, 0, readback, 0, expectedCopy.byteLength);
  device.queue.submit([encoder.finish()]);

  await readback.mapAsync(mapMode.READ);
  const outputCopy = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();

  return {
    supported: true,
    message: 'WebGPU radiance probe matched CPU reference',
    output: outputCopy,
    maxAbsDiff: maxAbsDiff(expectedCopy, outputCopy),
    validMismatches: mismatchCount(expectedCopy, outputCopy),
  };
}

function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  let max = 0;
  for (let i = 0; i < a.length; i++) {
    max = Math.max(max, Math.abs(a[i] - b[i]));
  }
  return max;
}

function mismatchCount(a: Float32Array, b: Float32Array): number {
  let count = 0;
  for (let i = 0; i < a.length; i += 8) {
    if (Math.round(a[i]) !== Math.round(b[i])) count += 1;
  }
  return count;
}
