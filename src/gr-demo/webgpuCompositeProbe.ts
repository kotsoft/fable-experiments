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
    STORAGE: number;
    COPY_DST: number;
    COPY_SRC: number;
    MAP_READ: number;
  };
  GPUMapMode?: {
    READ: number;
  };
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

fn observed_disk_rgb(position: vec3<f32>, momentum: vec4<f32>, observer_velocity: vec4<f32>, input: RaySample) -> vec3<f32> {
  let spin = input.paramsA.x;
  let mass = input.paramsB.y;
  let inner_radius = input.paramsC.x;
  let inner_temperature = input.paramsC.z;
  let emissivity_scale = input.paramsC.w;
  let boost_power = input.paramsD.x;
  let spin_direction = input.paramsD.y;
  let radius = ks_radius(position, spin);
  let emitter_velocity = disk_emitter_four_velocity(position, spin, mass, spin_direction);
  let redshift = redshift_factor(momentum, emitter_velocity, observer_velocity);
  let temperature = inner_temperature * pow(radius / inner_radius, -0.75);
  let emitted_rgb = blackbody_rgb(temperature);
  let radial_falloff = pow(radius / inner_radius, -2.4);
  let bolometric = emissivity_scale * radial_falloff * pow(max(redshift, 0.0), boost_power);
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
  return vec3<f32>(0.35 + 0.35 * dir.x, 0.35 + 0.35 * dir.y, 0.55 + 0.3 * dir.z);
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

export async function runWebGpuCompositeProbe(
  samples: Float32Array<ArrayBufferLike>,
  expected: Float32Array<ArrayBufferLike>,
): Promise<WebGpuCompositeProbeResult> {
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

  const module = device.createShaderModule({ code: COMPOSITE_SHADER });
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
  pass.dispatchWorkgroups(Math.ceil(inputCopy.length / 28 / 64));
  pass.end();
  encoder.copyBufferToBuffer(output, 0, readback, 0, expectedCopy.byteLength);
  device.queue.submit([encoder.finish()]);

  await readback.mapAsync(mapMode.READ);
  const outputCopy = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();

  return {
    supported: true,
    message: 'WebGPU composite probe matched CPU reference',
    output: outputCopy,
    maxAbsDiff: maxAbsDiff(expectedCopy, outputCopy),
    statusMismatches: mismatchCount(expectedCopy, outputCopy, 8, 0),
    diskMismatches: mismatchCount(expectedCopy, outputCopy, 8, 3),
  };
}

function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  let max = 0;
  for (let i = 0; i < a.length; i++) {
    max = Math.max(max, Math.abs(a[i] - b[i]));
  }
  return max;
}

function mismatchCount(a: Float32Array, b: Float32Array, stride: number, offset: number): number {
  let count = 0;
  for (let i = offset; i < a.length; i += stride) {
    if (Math.round(a[i]) !== Math.round(b[i])) count += 1;
  }
  return count;
}
