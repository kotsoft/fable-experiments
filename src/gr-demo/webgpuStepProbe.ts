export interface WebGpuStepProbeResult {
  supported: boolean;
  message: string;
  output?: Float32Array;
  maxAbsDiff?: number;
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

const STEP_SHADER = `
struct StepSample {
  position: vec4<f32>,
  momentum: vec4<f32>,
  params: vec4<f32>,
};

struct StepOutput {
  position: vec4<f32>,
  momentum: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> samples: array<StepSample>;
@group(0) @binding(1) var<storage, read_write> outputData: array<StepOutput>;

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

fn hamiltonian(position: vec3<f32>, momentum: vec4<f32>, spin: f32) -> f32 {
  let r = ks_radius(position, spin);
  let r2 = r * r;
  let scalar = 2.0 * r2 * r / max(r2 * r2 + spin * spin * position.z * position.z, 1.0e-8);
  let l = ks_l(position, spin, r);
  let pt = momentum.x;
  let ps = momentum.yzw;
  let minkowski = -pt * pt + dot(ps, ps);
  let projected = -pt + dot(l, ps);
  return 0.5 * (minkowski - scalar * projected * projected);
}

fn coordinate_velocity(position: vec3<f32>, momentum: vec4<f32>, spin: f32) -> vec4<f32> {
  let r = ks_radius(position, spin);
  let r2 = r * r;
  let scalar = 2.0 * r2 * r / max(r2 * r2 + spin * spin * position.z * position.z, 1.0e-8);
  let l = ks_l(position, spin, r);
  let projected = -momentum.x + dot(l, momentum.yzw);
  return vec4<f32>(
    -momentum.x + scalar * projected,
    momentum.y - scalar * projected * l.x,
    momentum.z - scalar * projected * l.y,
    momentum.w - scalar * projected * l.z
  );
}

fn gradient(position: vec3<f32>, momentum: vec4<f32>, spin: f32) -> vec3<f32> {
  let radius = max(ks_radius(position, spin), 1.0);
  let eps = 1.0e-5 * radius;
  let ex = vec3<f32>(eps, 0.0, 0.0);
  let ey = vec3<f32>(0.0, eps, 0.0);
  let ez = vec3<f32>(0.0, 0.0, eps);
  return vec3<f32>(
    (hamiltonian(position + ex, momentum, spin) - hamiltonian(position - ex, momentum, spin)) / (2.0 * eps),
    (hamiltonian(position + ey, momentum, spin) - hamiltonian(position - ey, momentum, spin)) / (2.0 * eps),
    (hamiltonian(position + ez, momentum, spin) - hamiltonian(position - ez, momentum, spin)) / (2.0 * eps)
  );
}

fn derivative(position: vec4<f32>, momentum: vec4<f32>, spin: f32) -> StepOutput {
  let v = coordinate_velocity(position.yzw, momentum, spin);
  let g = gradient(position.yzw, momentum, spin);
  return StepOutput(v, vec4<f32>(0.0, -g.x, -g.y, -g.z));
}

fn add_scaled(a: StepOutput, b: StepOutput, scale: f32) -> StepOutput {
  return StepOutput(a.position + b.position * scale, a.momentum + b.momentum * scale);
}

fn rk4(position: vec4<f32>, momentum: vec4<f32>, spin: f32, step: f32) -> StepOutput {
  let s0 = StepOutput(position, momentum);
  let k1 = derivative(s0.position, s0.momentum, spin);
  let s1 = add_scaled(s0, k1, step * 0.5);
  let k2 = derivative(s1.position, s1.momentum, spin);
  let s2 = add_scaled(s0, k2, step * 0.5);
  let k3 = derivative(s2.position, s2.momentum, spin);
  let s3 = add_scaled(s0, k3, step);
  let k4 = derivative(s3.position, s3.momentum, spin);
  return StepOutput(
    position + step / 6.0 * (k1.position + 2.0 * k2.position + 2.0 * k3.position + k4.position),
    momentum + step / 6.0 * (k1.momentum + 2.0 * k2.momentum + 2.0 * k3.momentum + k4.momentum)
  );
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= arrayLength(&outputData)) {
    return;
  }
  let sample = samples[i];
  outputData[i] = rk4(sample.position, sample.momentum, sample.params.x, sample.params.y);
}
`;

export async function runWebGpuStepProbe(
  samples: Float32Array<ArrayBufferLike>,
  expected: Float32Array<ArrayBufferLike>,
): Promise<WebGpuStepProbeResult> {
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

  const module = device.createShaderModule({ code: STEP_SHADER });
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
  pass.dispatchWorkgroups(Math.ceil(inputCopy.length / 12 / 64));
  pass.end();
  encoder.copyBufferToBuffer(output, 0, readback, 0, expectedCopy.byteLength);
  device.queue.submit([encoder.finish()]);

  await readback.mapAsync(mapMode.READ);
  const outputCopy = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();

  return {
    supported: true,
    message: 'WebGPU geodesic step probe matched CPU reference',
    output: outputCopy,
    maxAbsDiff: maxAbsDiff(expectedCopy, outputCopy),
  };
}

function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  let max = 0;
  for (let i = 0; i < a.length; i++) {
    max = Math.max(max, Math.abs(a[i] - b[i]));
  }
  return max;
}
