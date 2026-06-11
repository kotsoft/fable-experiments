export interface WebGpuTraceProbeResult {
  supported: boolean;
  message: string;
  output?: Float32Array;
  maxAbsDiff?: number;
  statusMismatches?: number;
  stepMismatches?: number;
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

const TRACE_SHADER = `
struct TraceSample {
  position: vec4<f32>,
  momentum: vec4<f32>,
  paramsA: vec4<f32>,
  paramsB: vec4<f32>,
};

struct TraceOutput {
  summary: vec4<f32>,
  position: vec4<f32>,
  momentum: vec4<f32>,
};

struct StepOutput {
  position: vec4<f32>,
  momentum: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> samples: array<TraceSample>;
@group(0) @binding(1) var<storage, read_write> outputData: array<TraceOutput>;

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

fn trace(sample: TraceSample) -> TraceOutput {
  let spin = sample.paramsA.x;
  let step_size = sample.paramsA.y;
  let escape_radius = sample.paramsA.z;
  let singularity_radius = sample.paramsA.w;
  let max_steps = u32(sample.paramsB.x);
  let mass = sample.paramsB.y;
  let horizon = horizon_radius(mass, spin);
  let h0 = hamiltonian(sample.position.yzw, sample.momentum, spin, mass);
  var position = sample.position;
  var momentum = sample.momentum;
  var max_drift = 0.0;

  for (var steps = 0u; steps < max_steps; steps = steps + 1u) {
    let radius = ks_radius(position.yzw, spin);
    if (radius <= singularity_radius) {
      return TraceOutput(vec4<f32>(2.0, f32(steps), radius, max_drift), position, momentum);
    }
    if (horizon > 0.0 && radius <= horizon) {
      return TraceOutput(vec4<f32>(1.0, f32(steps), radius, max_drift), position, momentum);
    }
    if (radius >= escape_radius && radial_coordinate_speed(position.yzw, momentum, spin, mass) > 0.0) {
      return TraceOutput(vec4<f32>(0.0, f32(steps), radius, max_drift), position, momentum);
    }

    let next = rk4(position, momentum, spin, mass, step_size);
    position = next.position;
    momentum = next.momentum;
    max_drift = max(max_drift, abs(hamiltonian(position.yzw, momentum, spin, mass) - h0));
  }

  let radius = ks_radius(position.yzw, spin);
  return TraceOutput(vec4<f32>(3.0, f32(max_steps), radius, max_drift), position, momentum);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= arrayLength(&outputData)) {
    return;
  }
  outputData[i] = trace(samples[i]);
}
`;

export async function runWebGpuTraceProbe(
  samples: Float32Array<ArrayBufferLike>,
  expected: Float32Array<ArrayBufferLike>,
): Promise<WebGpuTraceProbeResult> {
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

  const module = device.createShaderModule({ code: TRACE_SHADER });
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
  pass.dispatchWorkgroups(Math.ceil(inputCopy.length / 16 / 64));
  pass.end();
  encoder.copyBufferToBuffer(output, 0, readback, 0, expectedCopy.byteLength);
  device.queue.submit([encoder.finish()]);

  await readback.mapAsync(mapMode.READ);
  const outputCopy = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();

  return {
    supported: true,
    message: 'WebGPU trace probe matched CPU reference',
    output: outputCopy,
    maxAbsDiff: maxAbsDiff(expectedCopy, outputCopy),
    statusMismatches: mismatchCount(expectedCopy, outputCopy, 12, 0),
    stepMismatches: mismatchCount(expectedCopy, outputCopy, 12, 1),
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
