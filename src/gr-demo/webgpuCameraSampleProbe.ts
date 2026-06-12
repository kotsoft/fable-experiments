import {
  COMPOSITE_CAMERA_UNIFORM_FLOATS,
  createCompositeCameraUniforms,
  type CompositeCameraSampleOptions,
} from '../gr/compositeSamples';

export interface WebGpuCameraSampleProbeResult {
  supported: boolean;
  message: string;
  output?: Float32Array;
  maxAbsDiff?: number;
  momentumMaxAbsDiff?: number;
}

interface WebGpuConstants {
  GPUBufferUsage?: {
    UNIFORM: number;
    STORAGE: number;
    COPY_DST: number;
    COPY_SRC: number;
    MAP_READ: number;
  };
  GPUMapMode?: {
    READ: number;
  };
}

const FLOAT_BYTES = Float32Array.BYTES_PER_ELEMENT;
const INPUT_FLOATS_PER_RAY = 28;

export const CAMERA_SAMPLE_SHADER = `
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

struct RaySample {
  position: vec4<f32>,
  momentum: vec4<f32>,
  observerVelocity: vec4<f32>,
  paramsA: vec4<f32>,
  paramsB: vec4<f32>,
  paramsC: vec4<f32>,
  paramsD: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<storage, read_write> samples: array<RaySample>;

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

fn local_direction(pixel_index: u32) -> vec3<f32> {
  let width = max(u32(camera.paramsB.z), 1u);
  let height = max(u32(camera.paramsB.w), 1u);
  let x = f32(pixel_index % width);
  let y = f32(pixel_index / width);
  let ndc_x = (2.0 * (x + 0.5) / f32(width) - 1.0) * camera.paramsD.w;
  let ndc_y = 1.0 - 2.0 * (y + 0.5) / f32(height);
  return normalize(vec3<f32>(ndc_x * camera.paramsD.z, ndc_y * camera.paramsD.z, 1.0));
}

fn launch_momentum(direction: vec3<f32>) -> vec4<f32> {
  let contravariant =
    camera.eTime +
    camera.eRight * direction.x +
    camera.eUp * direction.y +
    camera.eForward * direction.z;
  return lower_vector(camera.position.yzw, camera.paramsA.x, camera.paramsB.y, contravariant);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= arrayLength(&samples)) {
    return;
  }
  let momentum = launch_momentum(local_direction(i));
  let traceParamsB = vec4<f32>(camera.paramsB.x, camera.paramsB.y, 0.0, 0.0);
  let traceParamsD = vec4<f32>(camera.paramsD.x, camera.paramsD.y, camera.reservedA.x, 0.0);
  samples[i] = RaySample(
    camera.position,
    momentum,
    camera.observerVelocity,
    camera.paramsA,
    traceParamsB,
    camera.paramsC,
    traceParamsD
  );
}
`;

export async function runWebGpuCameraSampleProbe(
  options: CompositeCameraSampleOptions,
  expected: Float32Array<ArrayBufferLike>,
): Promise<WebGpuCameraSampleProbeResult> {
  const raw = await runWebGpuCameraSamples(options);
  if (!raw.supported || !raw.output) {
    return { supported: raw.supported, message: raw.message };
  }

  return {
    supported: true,
    message: 'WebGPU camera sample probe matched CPU reference',
    output: raw.output,
    maxAbsDiff: maxAbsDiff(expected, raw.output),
    momentumMaxAbsDiff: stridedMaxAbsDiff(expected, raw.output, INPUT_FLOATS_PER_RAY, 4, 8),
  };
}

export async function runWebGpuCameraSamples(
  options: CompositeCameraSampleOptions,
): Promise<WebGpuCameraSampleProbeResult> {
  const constants = globalThis as typeof globalThis & WebGpuConstants;
  const usage = constants.GPUBufferUsage;
  const mapMode = constants.GPUMapMode;
  if (!navigator.gpu || !usage || !mapMode) {
    return { supported: false, message: 'WebGPU globals are unavailable' };
  }

  const uniforms = createCompositeCameraUniforms(options);
  const rayCount = options.width * options.height;
  const outputByteLength = rayCount * INPUT_FLOATS_PER_RAY * FLOAT_BYTES;
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return { supported: false, message: 'No WebGPU adapter available' };
  const device = await adapter.requestDevice();

  const uniformBuffer = device.createBuffer({
    size: COMPOSITE_CAMERA_UNIFORM_FLOATS * FLOAT_BYTES,
    usage: usage.UNIFORM | usage.COPY_DST,
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

  const module = device.createShaderModule({ code: CAMERA_SAMPLE_SHADER });
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: 'main' },
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
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
    message: 'WebGPU camera samples generated',
    output: outputCopy,
  };
}

function maxAbsDiff(a: Float32Array<ArrayBufferLike>, b: Float32Array<ArrayBufferLike>): number {
  let max = 0;
  const count = Math.min(a.length, b.length);
  for (let i = 0; i < count; i++) {
    max = Math.max(max, Math.abs(a[i] - b[i]));
  }
  return max;
}

function stridedMaxAbsDiff(
  a: Float32Array<ArrayBufferLike>,
  b: Float32Array<ArrayBufferLike>,
  stride: number,
  start: number,
  end: number,
): number {
  let max = 0;
  const count = Math.min(a.length, b.length);
  for (let base = 0; base < count; base += stride) {
    for (let offset = start; offset < end; offset++) {
      max = Math.max(max, Math.abs(a[base + offset] - b[base + offset]));
    }
  }
  return max;
}
